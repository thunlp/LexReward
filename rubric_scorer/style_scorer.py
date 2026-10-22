#!/usr/bin/env python3
"""Style Reward scorer v7.2 for Chinese legal documents.

Dimensions: D1 legal terminology, D2 subjectivity, D3 conjunction rate,
D4 sentence structure and D5 collocations. Reference statistics come from
CJO civil judgments (LY fields). The normalization scales are loaded from
reference_data/normalization_scales.json. Rewards use -distance / scale;
the overall reward is the arithmetic mean of the five dimensions.

Usage:
    python scorer/style_scorer.py --file input.txt --json
    python scorer/style_scorer.py --jsonl pairs.jsonl

Resources are resolved relative to this module. Environment variables can
override the legal-term JSON file and HowNet lexicon directory.
"""

import re
import sys
import json
import argparse
import math
import os
import statistics
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
import jieba
import jieba.posseg as pseg


# ============================================================
# 1. Text-processing helpers
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Split text at Chinese sentence punctuation and line breaks."""
    return [s.strip() for s in re.split(r"[。！？；\n]+", text) if s.strip()]


def preprocess_text(text: str) -> str:
    """Remove table-like and standalone markup lines, retaining prose."""
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"[\|\+\-\=\s]+", s):
            continue
        if s.count("|") >= 2 and len(s) < 300:
            continue
        if re.match(r"^\s*<[^>]+>\s*$", s):
            continue
        kept.append(s)
    return "\n".join(kept)


def load_legal_terms() -> set[str]:
    """Load a JSON array of legal terms.
    
    Use lexicons/legal_terms.json next to this module by default.
    LEXREWARD_LEGAL_TERMS_FILE can override the file location.
    """
    cache_path = os.environ.get("LEXREWARD_LEGAL_TERMS_FILE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lexicons", "legal_terms.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    raise FileNotFoundError(
        f"Legal terms lexicon not found at {cache_path}. "
        "Place a JSON array of legal terms there, or set LEXREWARD_LEGAL_TERMS_FILE.")


def clean_text_for_fmm(text: str) -> str:
    """Keep only CJK characters and ASCII letters or digits for matching."""
    return re.sub(r"[^一-鿿㐀-䶿a-zA-Z0-9]", "", text)


def fmm_match(text: str, dictionary: set[str], max_len: int = 8, min_len: int = 2) -> list[str]:
    """Return forward maximum-matching terms, including repeated occurrences."""
    cleaned = clean_text_for_fmm(text)
    total = len(cleaned)
    matched: list[str] = []
    i = 0
    while i < total:
        found = False
        for w in range(min(max_len, total - i), min_len - 1, -1):
            candidate = cleaned[i:i + w]
            if candidate in dictionary:
                matched.append(candidate)
                i += w
                found = True
                break
        if not found:
            i += 1
    return matched


def reward_to_level(reward: float) -> str:
    if reward >= -0.5:
        return "Excellent"
    elif reward >= -2.0:
        return "Good"
    elif reward >= -5.0:
        return "Fair"
    else:
        return "Poor"


# ============================================================
# 2. Result data classes
# ============================================================

@dataclass
class DimResult:
    dimension: str
    reward: float
    level: str
    defect_count: int
    defects: list[str] = field(default_factory=list)


@dataclass
class StyleScore:
    text: str
    overall_reward: float
    overall_level: str
    total_defects: int
    dimensions: dict[str, DimResult]
    strategy: str


# ============================================================
# 3. D1 legal terminology: KL divergence
# ============================================================


# Reference CJO civil-judgment term distribution P.
# Loaded from d1_term_reference_ms_ly.json.


def _load_d1_reference() -> dict:
    """Load legal-term document frequencies and corpus term counts."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reference_data", "d1_term_reference_ms_ly.json")
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_normalization_scales() -> dict[str, float]:
    """Load the per-dimension normalization scales."""
    scale_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "reference_data", "normalization_scales.json")
    if not os.path.exists(scale_path):
        return {"D1": 10.0, "D2": 10.0, "D3": 0.1, "D4": 100.0, "D5": 0.1}
    with open(scale_path, "r", encoding="utf-8") as f:
        return json.load(f).get("scales", {})


class LegalTermScorerV7:
    """Measure legal-term distribution divergence.
    
    Forward maximum matching yields an empirical term distribution Q.
    Normalize reference corpus counts into P, compute KL(P || Q) with an
    epsilon floor and return reward = -KL / scale.
    """

    name = "D1_用词专业性"
    label = "用词专业性"
    _EPSILON = 1e-10

    def __init__(self):
        self.legal_terms = load_legal_terms()
        self.scale = _load_normalization_scales().get("D1", 10.0)
        ref_data = _load_d1_reference()
        self.df: dict[str, int] = ref_data.get("df", {})
        self.ref_N: int = ref_data.get("N_docs", 971)
        ref_tf_total: dict[str, int] = ref_data.get("tf_total", {})

        # Normalize P(t) over all reference legal terms.
        total_ref_tf = sum(ref_tf_total.values())
        self.ref_freq: dict[str, float] = {}
        if total_ref_tf > 0:
            for t, c in ref_tf_total.items():
                self.ref_freq[t] = c / total_ref_tf

        # Set of terms in the reference distribution.
        self.ref_terms: set[str] = set(self.ref_freq)

    def _compute_kl(self, text: str) -> tuple[float, dict[str, int], set[str]]:
        """Return (KL divergence, matched term counts, matched term set)."""
        text = preprocess_text(text)
        matched = fmm_match(text, self.legal_terms)
        if not matched:
            return 999.0, {}, set()

        tf: dict[str, int] = {}
        for t in matched:
            tf[t] = tf.get(t, 0) + 1
        tf_set = set(tf)
        total_tf = sum(tf.values())

        # Normalize empirical Q(t) over reference legal terms.
        text_freq: dict[str, float] = {}
        freq_total = 0.0
        for t in self.ref_terms:
            f = tf.get(t, 0) / total_tf
            text_freq[t] = f
            freq_total += f
        # Normalize the frequency distribution.
        if freq_total > 0:
            for t in text_freq:
                text_freq[t] /= freq_total

        # KL(P||Q) = sum P(t) * log(P(t) / Q(t))
        kl = 0.0
        for t in self.ref_terms:
            p = self.ref_freq[t]
            q = max(text_freq.get(t, 0.0), self._EPSILON)
            if p > 0:
                kl += p * math.log(p / q)

        return kl, tf, tf_set

    def score(self, text: str) -> DimResult:
        kl, tf, tf_set = self._compute_kl(text)

        if not tf:
            return DimResult(self.name, 0.0, "Fair", 0, ["未匹配到法律术语"])

        reward = -kl / self.scale
        level = reward_to_level(reward)

        missing_terms = sorted(
            [t for t in self.ref_terms if t not in tf_set],
            key=lambda t: -self.ref_freq.get(t, 0),
        )
        defect_count = len(missing_terms)
        defects: list[str] = []

        info = (
            f"KL={kl:.4f}, reward={reward:+.3f} [{level}], "
            f"术语数={len(tf)}, 缺参考术语={defect_count}/{len(self.ref_terms)}"
        )
        defects.append(info)

        for t in missing_terms[:10]:
            defects.append(f"缺失术语: {t} (参考频率 {self.ref_freq[t]:.4f})")

        return DimResult(self.name, round(reward, 3), level, defect_count, defects)


# ============================================================
# 4. D2 subjectivity: sentiment-term KL divergence
# ============================================================

def _load_sentiment_words() -> set[str]:
    """Load the HowNet Chinese sentiment lexicon.
    
    Use lexicons/hownet next to this module by default, preserving the
    original Chinese filenames and GB2312 encoding. LEXREWARD_HOWNET_DIR
    can override the directory location.
    """
    hownet_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lexicons", "hownet")
    hownet_dir = os.environ.get("LEXREWARD_HOWNET_DIR") or hownet_dir
    words: set[str] = set()
    if not os.path.isdir(hownet_dir):
        raise FileNotFoundError(
            f"HowNet sentiment lexicon directory not found: {hownet_dir}. "
            "Place the *（中文）.txt lexicon files there, or set LEXREWARD_HOWNET_DIR.")
    for fname in os.listdir(hownet_dir):
        if not fname.endswith("（中文）.txt"):
            continue
        fpath = os.path.join(hownet_dir, fname)
        try:
            with open(fpath, "r", encoding="gb2312", errors="ignore") as f:
                for line in f:
                    w = line.strip()
                    if not w:
                        continue
                    if w[0].isdigit():
                        continue
                    if "{" in w or "|" in w:
                        continue
                    if len(w) < 2:
                        continue
                    words.add(w)
        except Exception:
            continue
    return words


def _load_d2_reference() -> dict:
    """Load sentiment-term document frequencies and corpus counts."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reference_data", "d2_sentiment_reference_ms_ly.json")
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


class SentimentScorerV7:
    """Measure sentiment-term distribution divergence.
    
    Match HowNet terms, normalize their empirical and reference frequencies,
    and return reward = -KL(P || Q) / scale with an epsilon floor for Q.
    """

    name = "D2_主观词控制"
    label = "主观词控制"
    _EPSILON = 1e-10

    def __init__(self):
        self.sentiment_words = _load_sentiment_words()
        self.scale = _load_normalization_scales().get("D2", 10.0)
        ref_data = _load_d2_reference()
        self.df: dict[str, int] = ref_data.get("df", {})
        self.ref_N: int = ref_data.get("N_docs", 971)
        ref_tf_total: dict[str, int] = ref_data.get("tf_total", {})

        # Normalize P(t) over reference sentiment terms.
        total_ref_tf = sum(ref_tf_total.values())
        self.ref_freq: dict[str, float] = {}
        if total_ref_tf > 0:
            for t, c in ref_tf_total.items():
                self.ref_freq[t] = c / total_ref_tf

        # Set of sentiment terms in the reference distribution.
        self.ref_terms: set[str] = set(self.ref_freq)

    def _compute_kl(self, text: str) -> tuple[float, dict[str, int], set[str]]:
        """Return (KL divergence, matched term counts, matched term set)."""
        text = preprocess_text(text)
        matched = fmm_match(text, self.sentiment_words)
        if not matched:
            return 999.0, {}, set()

        tf: dict[str, int] = {}
        for t in matched:
            tf[t] = tf.get(t, 0) + 1
        tf_set = set(tf)
        total_tf = sum(tf.values())

        # Normalize empirical Q(t) over reference sentiment terms.
        text_freq: dict[str, float] = {}
        freq_total = 0.0
        for t in self.ref_terms:
            f = tf.get(t, 0) / total_tf
            text_freq[t] = f
            freq_total += f
        # Normalize the frequency distribution.
        if freq_total > 0:
            for t in text_freq:
                text_freq[t] /= freq_total

        # KL(P||Q) = sum P(t) * log(P(t) / Q(t))
        kl = 0.0
        for t in self.ref_terms:
            p = self.ref_freq[t]
            q = max(text_freq.get(t, 0.0), self._EPSILON)
            if p > 0:
                kl += p * math.log(p / q)

        return kl, tf, tf_set

    def score(self, text: str) -> DimResult:
        kl, tf, tf_set = self._compute_kl(text)

        if not tf:
            return DimResult(self.name, 0.0, "Fair", 0, ["未匹配到情感词"])

        reward = -kl / self.scale
        level = reward_to_level(reward)

        missing_terms = sorted(
            [t for t in self.ref_terms if t not in tf_set],
            key=lambda t: -self.ref_freq.get(t, 0),
        )
        defect_count = len(missing_terms)
        defects: list[str] = []

        info = (
            f"KL={kl:.4f}, reward={reward:+.3f} [{level}], "
            f"情感词数={len(tf)}, 缺参考情感词={defect_count}/{len(self.ref_terms)}"
        )
        defects.append(info)

        for t in missing_terms[:10]:
            defects.append(f"缺失情感词: {t} (参考频率 {self.ref_freq[t]:.4f})")

        return DimResult(self.name, round(reward, 3), level, defect_count, defects)


# ============================================================
# 5. D3 conjunction rate: POS tagging and reference-rate deviation
# ============================================================

# jieba POS tags: c for conjunctions, r for pronouns.
_POS_CONJ = "c"
_POS_PRON = "r"

# Semantic subclasses of conjunctions.
_CONJ_SUBCLASSES: dict[str, set[str]] = {
    "并列": {"和", "及", "与", "以及", "及其", "和其", "同", "且", "并", "并且",
             "既", "亦", "乃至", "暨", "及以", "及使", "及为", "及非", "及后",
             "及所", "及至", "与以", "与与此同时", "并于", "并以", "并为", "并仅",
             "并致", "并令", "并有", "并至", "且其", "且以", "且为", "且是",
             "且如", "且仅", "且不论", "且此"},
    "选择": {"或", "或者", "还是", "要么", "不是", "与其", "抑或", "或仅",
             "或因", "或其", "或是", "或为", "或过", "或言"},
    "转折": {"但", "但是", "然而", "不过", "只是", "虽", "虽然", "尽管",
             "却", "反之", "而是", "并非", "而非", "然", "但其", "但以",
             "但所", "但仅", "但此", "但因", "但若", "但有", "但至", "但次",
             "但如", "然其", "然于", "然至", "虽以", "虽于", "虽是", "虽为",
             "虽有", "虽述", "虽说", "虽不相", "固然", "非但", "反而",
             "只不过", "只是"},
    "因果": {"因", "因为", "由于", "故", "因此", "所以", "由此", "是以",
             "因而", "故而", "这是因为", "因其", "因有", "因是", "因非", "因不",
             "因后", "因管", "所致", "故此", "故其", "故仅", "故管", "是以",
             "于是", "以致", "由其", "由", "由此", "由以", "由而", "由后",
             "由其", "之所以", "因何", "因何", "因何", "因何", "无果", "无因",
             "不致", "不因其"},
    "条件": {"如果", "若", "假如", "假设", "倘若", "假使", "即使", "即便",
             "只要", "只有", "除非", "无论", "不论", "不管", "如若", "若是",
             "既然", "既然如此", "如果说", "如有", "无论是"},
    "递进": {"而且", "此外", "另外", "同时", "进而", "从而", "不仅",
             "不仅如此", "加之", "况且", "再者", "再说", "再加之", "何况",
             "另一方面", "除此之外", "继而", "接着", "尔后", "而后",
             "然后", "此后", "此后"},
    "总结": {"综上所述", "综上", "总之", "可见", "由此可见", "据此",
             "为此", "如前所述", "如上所述", "换言之", "就是说",
             "如此说来", "如此"},
    "助连词": {"所", "可以", "而", "即", "及时", "除外", "否则",
               "以外", "以此", "而言", "以便", "以免", "即为", "即以",
               "即时", "即其", "即是", "即于", "即非", "即次", "即管",
               "即一是", "由有", "有别", "之时", "此时", "何其",
               "所于", "所说", "所的", "所对其", "所有者", "所见",
               "所非", "所对其所", "总以", "总是", "何为", "何时",
               "何其", "只", "另有", "另有"},
}

# Semantic subclasses of pronouns.
_PRON_SUBCLASSES: dict[str, set[str]] = {
    "指示代词": {"这", "那", "此", "该", "其", "上述", "前述", "该案", "该院",
                "该车", "该项", "该笔", "该款", "此项", "此款", "此案", "此法",
                "之人", "之事", "该等", "这些", "这个", "这种", "这么", "这样",
                "这笔", "这处", "这款", "这项", "这一", "这块", "这家", "这中",
                "这边", "这次", "那某", "那个", "那里", "那时", "那天", "那么",
                "该行", "该组", "该条", "该卡", "该处", "该次", "该所", "该厂",
                "该局", "该店", "该类", "该种", "该站", "该地", "该校", "该社",
                "该物", "该期", "该市", "该户", "该机", "该片", "该楼", "该段",
                "该法", "该栋", "该会", "该书", "该些", "该药",
                "此事", "此条", "此笔", "此种", "此类", "此举", "此人", "此栋",
                "此时", "此外", "此款", "此仅",
                "之妻", "之子", "之用", "之身", "之处", "之款", "之时", "之次",
                "其间", "其车", "其行", "其日", "其方", "其自", "其妻", "其父",
                "其子", "其四", "其三", "其二", "其一", "其时", "其后", "其于",
                "其以", "其若", "其仅", "其以",
                "本文", "本案", "本次", "此次", "本条", "本地", "本身", "本村",
                "本镇", "本省", "本项", "本笔", "本科", "本城", "本组", "本章",
                "换言之", "如此", "为此"},
    "人称代词": {"我", "我们", "我方", "我行", "我家", "我局", "我处", "我儿",
                "你", "你们", "您", "她", "他", "他们", "他人", "本人", "自己",
                "自身", "彼此", "各自", "己", "己身", "敝", "它", "别人",
                "有人", "有别", "有他", "有个", "有一"},
    "不定代词": {"某", "某某", "某甲", "某乙", "某某某", "某君", "某己", "某种",
                "某期", "某日", "某镇", "某村", "某些", "某项", "某自", "某次",
                "任何", "任何人", "一切", "各", "各人", "各方", "各项", "各条",
                "各款", "各省", "各地", "各户", "各类", "各案", "各个", "各级",
                "各国", "各股", "各门", "各套", "各笔", "各庄", "各派",
                "若干", "其余", "另", "其他", "另外", "其它", "各种",
                "各种", "任", "任一", "各各", "别", "别的", "别处",
                "有些", "有事", "有方", "有时", "有余", "有家", "有天", "有何",
                "有中"},
    "疑问代词": {"何", "如何", "怎么", "怎样", "哪些", "哪个", "谁", "什么",
                "多少", "几", "为何", "何种", "何方", "何处", "何人", "何国",
                "何等", "何笔", "何么", "哪里", "哪种", "怎么样", "啥",
                "为什么", "哪", "哪方"},
    "数量/时间": {"每", "每日", "每月", "每年", "每天", "每月", "之日", "之一",
                "之二", "之三", "其中", "其次", "每次", "每人", "每期", "每笔",
                "每户", "每周", "每套", "每股", "每栋", "每位", "每亩", "每份",
                "每季", "每车", "每间", "每瓶", "每桶", "每时", "每一",
                "每月", "之时", "之久"},
    "反身/范围": {"自", "自行", "自身", "自家", "自我", "自此", "自书", "自二",
                "自各"},
}


def _classify_word(word: str, subclasses: dict[str, set[str]]) -> str:
    """Map a word to a semantic subclass, using the final subclass as fallback."""
    for cat, words in subclasses.items():
        if cat.endswith("其他"):
            continue
        if word in words:
            return cat
    for cat in subclasses:
        if cat.endswith("其他"):
            return cat
    return list(subclasses.keys())[-1]


def _extract_pos_words(text: str, pos_tags: set[str]) -> list[str]:
    """Extract words with the specified jieba POS tag, retaining repeats."""
    text = preprocess_text(text)
    words: list[str] = []
    for w in pseg.cut(text):
        if w.flag in pos_tags and len(w.word.strip()) > 0:
            words.append(w.word.strip())
    return words


def _load_d3_reference() -> dict:
    """Load reference conjunction and pronoun frequencies."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reference_data", "d3_conj_pron_reference_ms_ly.json")
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ConjPronScorerV7:
    """Score the conjunction-rate deviation from the reference.
    
    Use jieba POS tagging to count conjunctions and all tokens. Compute
    rate = conjunction_count / token_count and return
    reward = -abs(rate - reference_rate) / scale.
    """

    name = "D3_连词率"
    label = "连词率"

    # CJO reference values from d3_conj_pron_reference_ms_ly.json.
    _ref_conj_rate = 0.0191

    def __init__(self):
        self.scale = _load_normalization_scales().get("D3", 0.01)
        ref_data = _load_d3_reference()
        conj_tf_total: dict[str, int] = ref_data.get("conj_tf_total", {})
        total_tokens: int = ref_data.get("total_tokens", 0)

        total_conj = sum(conj_tf_total.values())
        if total_tokens > 0:
            self._ref_conj_rate = total_conj / total_tokens

    @staticmethod
    def _count_conj_rate(text: str) -> tuple[float, int]:
        """Return (conjunction rate, total tokens) using jieba POS tagging."""
        n_total = 0
        n_conj = 0
        for w in pseg.cut(text):
            word = w.word.strip()
            if not word:
                continue
            n_total += 1
            if w.flag == _POS_CONJ:
                n_conj += 1
        if n_total == 0:
            return 0.0, 0
        return n_conj / n_total, n_total

    def score(self, text: str) -> DimResult:
        text = preprocess_text(text)
        conj_rate, n_total = self._count_conj_rate(text)

        if n_total == 0:
            return DimResult(self.name, 0.0, "Fair", 0, ["文本为空，无法分词"])

        w_dist = abs(conj_rate - self._ref_conj_rate)

        reward = -w_dist / self.scale
        level = reward_to_level(reward)

        defects: list[str] = []
        info = (
            f"conj_rate={conj_rate:.4f} (ref={self._ref_conj_rate:.4f}), "
            f"diff={w_dist:.4f}, "
            f"reward={reward:+.3f} [{level}], "
            f"total_tokens={n_total}"
        )
        defects.append(info)

        conj_diff = conj_rate - self._ref_conj_rate
        if abs(conj_diff) > 0.02:
            direction = "偏高" if conj_diff > 0 else "偏低"
            defects.append(f"连词率{direction}: {conj_rate:.4f} vs 参考 {self._ref_conj_rate:.4f}")

        defect_count = 1 if abs(conj_diff) > 0.02 else 0

        return DimResult(self.name, round(reward, 3), level, defect_count, defects)


# ============================================================
# 6. D4 sentence structure: distance between sentence-length statistics
# ============================================================

class SentenceStructureScorerV7:
    """Score deviations in sentence-length mean and standard deviation.
    
    Compute W = sqrt((mean - reference_mean)^2 + (std - reference_std)^2)
    with equal weights. Reference values are 64.7 characters for the mean
    and 45.3 for the standard deviation. Return reward = -W / scale.
    """

    name = "D4_句式结构"
    label = "句式结构"

    def __init__(self):
        self.scale = _load_normalization_scales().get("D4", 100.0)

    def score(self, text: str) -> DimResult:
        text = preprocess_text(text)
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return DimResult(self.name, 0.0, "Fair", 0, ["句子数不足，无法计算分布"])

        lengths = [len(s) for s in sentences]
        n = len(lengths)

        # Reference sentence-level statistics from CJO civil judgments.
        ref_mean = 64.7
        ref_std = 45.3

        # Equal weights for distance components.
        w_mean = 1.0
        w_std = 1.0

        # Compute statistics for the current text.
        text_mean = statistics.mean(lengths)
        text_std = statistics.stdev(lengths) if len(lengths) > 1 else 0

        # Compute the distance using the weights configured below.
        wasserstein_dist = math.sqrt(w_mean * (text_mean - ref_mean) ** 2 + w_std * (text_std - ref_std) ** 2)

        # Compute the negative distance normalized by its scale.
        reward = -wasserstein_dist / self.scale
        level = reward_to_level(reward)

        direction = "偏离参考" if wasserstein_dist > 50 else "接近参考"
        info = (
            f"Wasserstein={wasserstein_dist:.2f}, "
            f"text_mean={text_mean:.1f}, text_std={text_std:.1f}, "
            f"ref_mean={ref_mean}, ref_std={ref_std}, "
            f"w_mean={w_mean}, w_std={w_std}, "
            f"reward={reward:+.3f}  [{level}], "
            f"句数={n}"
        )

        return DimResult(self.name, round(reward, 3), level, 0, [info])


# ============================================================
# 7. D5 collocations: reference n-gram overlap rates
# ============================================================

def _load_d5_reference() -> dict:
    """Load reference n-gram sets and overlap-recall statistics."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reference_data", "d5_rouge_reference_ms_ly.json")
    if not os.path.exists(ref_path):
        return {}
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


_PUNCT_RE = re.compile('^[^\\w\\s]+$|^[，。、；：？！""（）【】《》·…—～·﹑﹔﹕﹖﹗﹝﹞﹟﹠﹡﹢﹣﹤﹥﹦％〈〉"\'\\-\\+\\=\\/\\\\<\\>\\*\\&\\^\\%\\$\\#\\@\\!\\~\\`\\[\\]\\{\\}\\|\\;\\:\\,\\.\\?]+$')


def _is_punct_token(token: str) -> bool:
    """Return whether a token consists entirely of punctuation."""
    if not token.strip():
        return True
    return bool(_PUNCT_RE.match(token))


class CollocationScorerV7:
    """Score reference n-gram overlap-rate deviations.
    
    Tokenize with jieba and remove punctuation. For each reference n-gram
    order of at least four, compute the fraction of input n-grams found in
    the reference set. Compute the Euclidean distance of these rates from
    reference means with equal weights, then return reward = -W / scale.
    """

    name = "D5_搭配正确性"
    label = "搭配正确性"

    def __init__(self):
        self.scale = _load_normalization_scales().get("D5", 0.1)
        ref_data = _load_d5_reference()
        rouge_stats = ref_data.get("rouge_stats", {})
        ref_ngrams_raw = ref_data.get("ref_ngrams", {})

        # Load reference n-grams, excluding those containing punctuation.
        self.ref_ngrams: dict[int, set[tuple[str, ...]]] = {}
        for key, grams in ref_ngrams_raw.items():
            n = int(key[1:])  # "n2" -> 2
            filtered = set()
            for g in grams:
                parts = g.split()
                if not any(_is_punct_token(p) for p in parts):
                    filtered.add(tuple(parts))
            self.ref_ngrams[n] = filtered

        # Load reference overlap-rate means and standard deviations.
        self.ref_means: dict[int, float] = {}
        self.ref_stds: dict[int, float] = {}
        self.ngram_orders: list[int] = []
        for key, stats in rouge_stats.items():
            n = int(key.replace("rouge", ""))
            self.ref_means[n] = stats["recall_mean"]
            self.ref_stds[n] = stats["recall_std"]
            self.ngram_orders.append(n)
        self.ngram_orders.sort()
        # Retain n-gram orders of at least four.
        self.ngram_orders = [n for n in self.ngram_orders if n >= 4]

        # Equal weights for distance components.
        self.weights: dict[int, float] = {}
        for n in self.ngram_orders:
            self.weights[n] = 1.0

    @staticmethod
    def _get_ngrams_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
        return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def _rouge_n_recall(self, tokens: list[str], n: int) -> float:
        text_ngrams = self._get_ngrams_set(tokens, n)
        ref_set = self.ref_ngrams.get(n, set())
        if not text_ngrams or not ref_set:
            return 0.0
        overlap = len(text_ngrams & ref_set)
        return overlap / len(text_ngrams)

    def score(self, text: str) -> DimResult:
        text = preprocess_text(text)
        tokens = [t for t in jieba.cut(text) if t.strip() and not _is_punct_token(t)]

        if len(tokens) < 2:
            return DimResult(self.name, 0.0, "Fair", 0, ["文本过短，无法计算 ROUGE"])

        # Compute the input n-gram overlap rate for each order.
        recalls = {}
        for n in self.ngram_orders:
            recalls[n] = self._rouge_n_recall(tokens, n)

        # Compute the distance using the weights configured below.
        w_dist_sq = 0.0
        for n in self.ngram_orders:
            w = self.weights.get(n, 1.0)
            diff = recalls[n] - self.ref_means.get(n, 0)
            w_dist_sq += w * diff * diff
        w_dist = math.sqrt(w_dist_sq)

        # Use the configured D5 normalization scale for the reward.
        scale = statistics.mean(list(self.ref_stds.values())) if self.ref_stds else 0.1
        reward = -w_dist / self.scale if self.scale > 0 else 0.0
        level = reward_to_level(reward)

        defects: list[str] = []
        recall_parts = ", ".join(f"R{n}={recalls[n]:.3f}" for n in self.ngram_orders)
        ref_parts = ", ".join(f"R{n}={self.ref_means[n]:.3f}" for n in self.ngram_orders)
        info = (
            f"W_dist={w_dist:.4f}, {recall_parts}, "
            f"ref=[{ref_parts}], "
            f"reward={reward:+.3f} [{level}], tokens={len(tokens)}"
        )
        defects.append(info)

        # Report deviations for each n-gram order.
        for n in self.ngram_orders:
            diff = recalls[n] - self.ref_means[n]
            std = self.ref_stds.get(n, 0.1)
            if abs(diff) > std:
                direction = "偏高" if diff > 0 else "偏低"
                defects.append(f"ROUGE-{n} {direction}: {recalls[n]:.3f} vs 参考 {self.ref_means[n]:.3f}")

        defect_count = sum(1 for n in self.ngram_orders if abs(recalls[n] - self.ref_means.get(n, 0)) > self.ref_stds.get(n, 0.1))

        return DimResult(self.name, round(reward, 3), level, defect_count, defects)


# ============================================================
# 8. Combined Style v7 scorer: D1 + D2 + D3 + D4 + D5
# ============================================================

class StyleScorerV7:
    """Combine Style v7 dimensions D1, D2, D3, D4 and D5 with equal weights."""

    def __init__(self):
        self.d1_scorer = LegalTermScorerV7()
        self.d2_scorer = SentimentScorerV7()
        self.d3_scorer = ConjPronScorerV7()
        self.d4_scorer = SentenceStructureScorerV7()
        self.d5_scorer = CollocationScorerV7()

    def score_D1(self, text: str) -> DimResult:
        return self.d1_scorer.score(text)

    def score_D2(self, text: str) -> DimResult:
        return self.d2_scorer.score(text)

    def score_D3(self, text: str) -> DimResult:
        return self.d3_scorer.score(text)

    def score_D4(self, text: str) -> DimResult:
        return self.d4_scorer.score(text)

    def score_D5(self, text: str) -> DimResult:
        return self.d5_scorer.score(text)

    def score(self, text: str) -> StyleScore:
        d1 = self.score_D1(text)
        d2 = self.score_D2(text)
        d3 = self.score_D3(text)
        d4 = self.score_D4(text)
        d5 = self.score_D5(text)

        dims = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}
        overall_reward = (d1.reward + d2.reward + d3.reward + d4.reward + d5.reward) / 5
        total_defects = d1.defect_count + d2.defect_count + d3.defect_count + d4.defect_count + d5.defect_count

        return StyleScore(
            text=text,
            overall_reward=round(overall_reward, 3),
            overall_level=reward_to_level(overall_reward),
            total_defects=total_defects,
            dimensions=dims,
            strategy="avg",
        )


# ============================================================
# 9. Output formatting
# ============================================================

def format_score(score: StyleScore) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"  Style Reward 评分结果 (v7.1 D1+D2+D3+D4+D5)")
    lines.append("=" * 65)
    lines.append(f"  综合得分: {score.overall_reward:+.3f}  ({score.overall_level})")
    lines.append(f"  总缺陷数: {score.total_defects}")
    lines.append(f"  聚合策略: {score.strategy}")
    lines.append("-" * 65)

    for key, d in score.dimensions.items():
        r = d.reward
        if r >= -0.5:
            icon = "●"
        elif r >= -2.0:
            icon = "◐"
        elif r >= -5.0:
            icon = "◑"
        else:
            icon = "○"
        dim_label = d.dimension if d.dimension else key
        lines.append(
            f"  {icon} {dim_label}   reward={d.reward:+.3f}  defects={d.defect_count:<2d}  [{d.level}]"
        )
        for defect in d.defects:
            lines.append(f"     ⤷ {defect}")
        lines.append("")

    lines.append("=" * 65)
    return "\n".join(lines)


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Style Reward scorer v7.2: D1, D2, D3, D4 and D5')
    parser.add_argument("text", nargs="?", default=None, help='Legal text to evaluate')
    parser.add_argument("--file", "-f", default=None, help='Read text from a UTF-8 file')
    parser.add_argument("--interactive", "-i", action="store_true", help='Read text interactively')
    parser.add_argument("--json", "-j", action="store_true", help='Output the result as JSON')
    parser.add_argument("--jsonl", "-l", default=None, help='Evaluate JSONL preference pairs with gold/generated fields')
    args = parser.parse_args()

    # Batch evaluation of JSONL preference pairs.
    if args.jsonl:
        samples = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if not samples:
            print("错误: JSONL 文件为空", file=sys.stderr)
            sys.exit(1)

        scorer = StyleScorerV7()

        # D1 statistics.
        d1_gw, d1_gnw, d1_tw = 0, 0, 0
        d1_gold_vals, d1_gen_vals = [], []

        # D2 statistics.
        d2_gw, d2_gnw, d2_tw = 0, 0, 0
        d2_gold_vals, d2_gen_vals = [], []

        # D3 statistics.
        d3_gw, d3_gnw, d3_tw = 0, 0, 0
        d3_gold_vals, d3_gen_vals = [], []

        # D4 statistics.
        d4_gw, d4_gnw, d4_tw = 0, 0, 0
        d4_gold_vals, d4_gen_vals = [], []

        # D5 statistics.
        d5_gw, d5_gnw, d5_tw = 0, 0, 0
        d5_gold_vals, d5_gen_vals = [], []

        # Overall statistics.
        all_gw, all_gnw, all_tw = 0, 0, 0
        all_gold_vals, all_gen_vals = [], []

        print(f"[INFO] 评估 {len(samples)} 条样本...", file=sys.stderr)
        for i, s in enumerate(samples):
            d1_gr = scorer.score_D1(s["gold"]).reward
            d1_gnr = scorer.score_D1(s["generated"]).reward
            d2_gr = scorer.score_D2(s["gold"]).reward
            d2_gnr = scorer.score_D2(s["generated"]).reward
            d3_gr = scorer.score_D3(s["gold"]).reward
            d3_gnr = scorer.score_D3(s["generated"]).reward
            d4_gr = scorer.score_D4(s["gold"]).reward
            d4_gnr = scorer.score_D4(s["generated"]).reward
            d5_gr = scorer.score_D5(s["gold"]).reward
            d5_gnr = scorer.score_D5(s["generated"]).reward
            all_gr = (d1_gr + d2_gr + d3_gr + d4_gr + d5_gr) / 5
            all_gnr = (d1_gnr + d2_gnr + d3_gnr + d4_gnr + d5_gnr) / 5

            d1_gold_vals.append(d1_gr); d1_gen_vals.append(d1_gnr)
            d2_gold_vals.append(d2_gr); d2_gen_vals.append(d2_gnr)
            d3_gold_vals.append(d3_gr); d3_gen_vals.append(d3_gnr)
            d4_gold_vals.append(d4_gr); d4_gen_vals.append(d4_gnr)
            d5_gold_vals.append(d5_gr); d5_gen_vals.append(d5_gnr)
            all_gold_vals.append(all_gr); all_gen_vals.append(all_gnr)

            if d1_gr > d1_gnr: d1_gw += 1
            elif d1_gnr > d1_gr: d1_gnw += 1
            else: d1_tw += 1

            if d2_gr > d2_gnr: d2_gw += 1
            elif d2_gnr > d2_gr: d2_gnw += 1
            else: d2_tw += 1

            if d3_gr > d3_gnr: d3_gw += 1
            elif d3_gnr > d3_gr: d3_gnw += 1
            else: d3_tw += 1

            if d4_gr > d4_gnr: d4_gw += 1
            elif d4_gnr > d4_gr: d4_gnw += 1
            else: d4_tw += 1

            if d5_gr > d5_gnr: d5_gw += 1
            elif d5_gnr > d5_gr: d5_gnw += 1
            else: d5_tw += 1

            if all_gr > all_gnr: all_gw += 1
            elif all_gnr > all_gr: all_gnw += 1
            else: all_tw += 1

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(samples)}...", file=sys.stderr)

        n = len(samples)
        print()
        for dim_name, gw, gnw, tw, gv, gnv in [
            ("D1", d1_gw, d1_gnw, d1_tw, d1_gold_vals, d1_gen_vals),
            ("D2", d2_gw, d2_gnw, d2_tw, d2_gold_vals, d2_gen_vals),
            ("D3", d3_gw, d3_gnw, d3_tw, d3_gold_vals, d3_gen_vals),
            ("D4", d4_gw, d4_gnw, d4_tw, d4_gold_vals, d4_gen_vals),
            ("D5", d5_gw, d5_gnw, d5_tw, d5_gold_vals, d5_gen_vals),
            ("综合", all_gw, all_gnw, all_tw, all_gold_vals, all_gen_vals),
        ]:
            print(f"  {dim_name} Gold 胜:       {gw} ({gw / n * 100:.1f}%)")
            print(f"  {dim_name} Generated 胜:  {gnw} ({gnw / n * 100:.1f}%)")
            print(f"  {dim_name} 平局:          {tw} ({tw / n * 100:.1f}%)")
            print(f"  {dim_name} Gold 均分:     {statistics.mean(gv):+.4f}")
            print(f"  {dim_name} Generated 均分: {statistics.mean(gnv):+.4f}")
            print()
        sys.exit(0)

    # Read the input text.
    if args.interactive:
        print("\n交互模式 — 输入法律文本后按 Ctrl+D 结束:\n", file=sys.stderr)
        lines = []
        try:
            while True:
                line = input()
                if line == "" and len(lines) > 0 and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        text = "\n".join(lines).strip()
        if not text:
            print("未输入文本，退出。", file=sys.stderr)
            sys.exit(1)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if not text:
        print("错误: 输入文本为空", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 评估中... (文本长度: {len(text)} 字)", file=sys.stderr)

    scorer = StyleScorerV7()
    result = scorer.score(text)

    if args.json:
        output = {
            "overall_reward": result.overall_reward,
            "overall_level": result.overall_level,
            "total_defects": result.total_defects,
            "strategy": result.strategy,
            "dimensions": {
                k: {
                    "reward": v.reward,
                    "level": v.level,
                    "defect_count": v.defect_count,
                    "defects": v.defects,
                }
                for k, v in result.dimensions.items()
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(format_score(result))


if __name__ == "__main__":
    main()
