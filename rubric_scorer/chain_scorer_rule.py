#!/usr/bin/env python3
"""Rule-based scoring of the structure of legal reasoning chains.

Four dimensions measure order, completeness, valid step labels and
non-redundancy. Steps are extracted from numbered markers such as
1. [step name], using the Chinese full-width brackets expected by STEP_PATTERN.
Task-specific Chinese step names are part of the scoring vocabulary.

Usage:
    python scorer/chain_scorer_rule.py --file input.txt --json
    python scorer/chain_scorer_rule.py --jsonl pairs.jsonl
"""

import re
import sys
import json
import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ============================================================
# 0. Step parsing
# ============================================================

# Extract step markers: number. 【name】
STEP_PATTERN = re.compile(
    r'(\d+)\s*[.．、]\s*【([^】]+)】[：:\s]*'
)

# Set of valid step names (7 required steps, default config)
DEFAULT_VALID_STEP_NAMES: frozenset[str] = frozenset([
    "原告", "被告", "纠纷类型", "法律依据",
    "责任划分", "原告损失总额", "判决结果",
])

# Step grouping for C1 Order (default config)
DEFAULT_GROUP_STEPS: dict[int, frozenset[str]] = {
    1: frozenset(["原告", "被告", "纠纷类型", "法律依据"]),
    2: frozenset(["责任划分", "原告损失总额"]),
    3: frozenset(["判决结果"]),
}

# Cross-group out-of-place penalty weights for C1 Order (default config)
DEFAULT_GROUP_OUT_OF_PLACE_WEIGHT: dict[int, int] = {
    1: 1,
    2: 1,
    3: 2,
}

# Retain aliases for compatibility with external callers.
VALID_STEP_NAMES = DEFAULT_VALID_STEP_NAMES
GROUP_STEPS = DEFAULT_GROUP_STEPS
GROUP_OUT_OF_PLACE_WEIGHT = DEFAULT_GROUP_OUT_OF_PLACE_WEIGHT


@dataclass
class StepConfig:
    """Configure task-specific step names, groups and order penalties.
    
    valid_names: Allowed names for C3.
    group_steps: Group-to-name mapping used by C1 and C2.
    group_out_of_place_weight: Penalty for an out-of-order group in C1.
    required_names: Names required by C2 when no group mapping is provided.
        Defaults to valid_names; a separate canonical set can support variants.
    """
    valid_names: frozenset[str] = DEFAULT_VALID_STEP_NAMES
    group_steps: dict[int, frozenset[str]] = field(default_factory=lambda: dict(DEFAULT_GROUP_STEPS))
    group_out_of_place_weight: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_GROUP_OUT_OF_PLACE_WEIGHT))
    required_names: frozenset[str] | None = None

    @property
    def required_steps(self) -> frozenset[str]:
        if self.required_names is not None:
            return self.required_names
        return self.valid_names


def _get_group(step_name: str, step_config: StepConfig = None) -> int | None:
    """Return the group identifier of a step, or None for an unknown name."""
    if step_config is None:
        step_config = StepConfig()
    name = step_name.strip()
    for g, names in step_config.group_steps.items():
        if name in names:
            return g
    return None


def extract_steps(text: str) -> list[dict]:
    """Extract numbered step markers, names, character offsets and text content."""
    steps: list[dict] = []
    for m in STEP_PATTERN.finditer(text):
        steps.append({
            "number": m.group(1),
            "name": m.group(2).strip(),
            "start": m.start(),
            "end": m.end(),
        })

    for i, step in enumerate(steps):
        content_start = step["end"]
        content_end = steps[i + 1]["start"] if i + 1 < len(steps) else len(text)
        step["content"] = text[content_start:content_end].strip()

    return steps


# ============================================================
# 1. Core data classes
# ============================================================

@dataclass
class DimResult:
    """Scoring result for one dimension."""
    dimension: str
    reward: float
    level: str
    defect_count: int
    defects: list[str] = field(default_factory=list)


@dataclass
class ChainScore:
    """Aggregated scoring result across the 4 dimensions."""
    text: str
    overall_reward: float
    overall_level: str
    total_defects: int
    dimensions: dict[str, DimResult]
    strategy: str


# ============================================================
# 2. Helper functions
# ============================================================

def level_from_defects(count: int) -> tuple[float, str]:
    """Map a defect count to a reward and quality level."""
    if count == 0:
        return +1.0, "Excellent"
    elif count <= 2:
        return +0.5, "Good"
    elif count <= 4:
        return -0.5, "Fair"
    else:
        return -1.0, "Poor"


def _reward_to_level(reward: float) -> str:
    if reward >= 0.75:
        return "Excellent"
    elif reward >= 0.25:
        return "Good"
    elif reward >= -0.25:
        return "Fair"
    else:
        return "Poor"


# ============================================================
# 3. Dimension scorers
# ============================================================

class BaseDimScorer(ABC):
    """Base class for dimension scorers."""
    name: str = ""
    label: str = ""

    def __init__(self, step_config: StepConfig = None):
        self.step_config = step_config or StepConfig()

    @abstractmethod
    def score(self, text: str, steps: list[dict] = None) -> DimResult:
        ...

    def _make_result(self, defects: list[str], defect_count: int = None) -> DimResult:
        n = defect_count if defect_count is not None else len(defects)
        reward, level = level_from_defects(n)
        return DimResult(self.name, reward, level, n, defects)


# C1: Step order

class OrderScorer(BaseDimScorer):
    """Measure ordering defects across configured groups.
    
    Count valid steps in each group and compute each expected position range.
    A step outside its group range incurs the configured penalty; the default
    penalties are one for groups 1 and 2, and two for group 3.
    """

    name = "C1_步骤顺序"
    label = "步骤顺序"

    def score(self, text: str, steps: list[dict] = None) -> DimResult:
        if steps is None:
            steps = extract_steps(text)

        if not steps:
            return DimResult(self.name, 0.0, "Fair", 0, ["无步骤标记可检测"])

        # Filter valid names before evaluating order.
        valid_steps = [(i, s) for i, s in enumerate(steps) if _get_group(s["name"], self.step_config) is not None]
        if not valid_steps:
            return DimResult(self.name, 0.0, "Fair", 0, ["无合法步骤标记"])

        # Count valid steps in each configured group.
        group_ids = sorted(self.step_config.group_steps.keys())
        group_counts: dict[int, int] = {g: 0 for g in group_ids}
        for _, s in valid_steps:
            g = _get_group(s["name"], self.step_config)
            group_counts[g] = group_counts.get(g, 0) + 1

        # Compute the expected position range for each group.
        offset = 0
        group_ranges: dict[int, tuple[int, int]] = {}
        for g in group_ids:
            n = group_counts[g]
            group_ranges[g] = (offset, offset + n)
            offset += n

        defects: list[str] = []
        defect_count = 0

        for pos, (orig_idx, s) in enumerate(valid_steps):
            g = _get_group(s["name"], self.step_config)
            lo, hi = group_ranges[g]
            if pos < lo or pos >= hi:
                weight = self.step_config.group_out_of_place_weight.get(g, 1)
                defect_count += weight
                defects.append(
                    f"步骤「{s['name']}」(G{g}) 出现在位置 {pos}，"
                    f"期望在 G{g} 区间 [{lo}, {hi})，错位权重 +{weight}"
                )

        reward, level = level_from_defects(defect_count)
        return DimResult(self.name, reward, level, defect_count, defects)


# C2: Step completeness

class CompletenessScorer(BaseDimScorer):
    """Check whether every configured group has at least one step.
    
    Each missing group counts as one defect. When no groups are configured,
    fall back to exact matching against the required step names.
    """

    name = "C2_步骤完整性"
    label = "步骤完整性"

    @property
    def REQUIRED_STEPS(self) -> frozenset[str]:
        return self.step_config.required_steps

    def score(self, text: str, steps: list[dict] = None) -> DimResult:
        if steps is None:
            steps = extract_steps(text)

        if not steps:
            defects = [f"缺失步骤「{m}」" for m in sorted(self.REQUIRED_STEPS)]
            return self._make_result(defects)

        # Require at least one step from each group.
        groups = self.step_config.group_steps
        if groups:
            covered_groups: set[int] = set()
            for s in steps:
                g = _get_group(s["name"], self.step_config)
                if g is not None:
                    covered_groups.add(g)

            # Choose a required label for the missing-group message.
            group_canonical: dict[int, str] = {}
            for g, names in groups.items():
                group_canonical[g] = next(iter(names), f"G{g}")

            defects = [
                f"缺失步骤「{group_canonical[g]}」"
                for g in sorted(groups.keys())
                if g not in covered_groups
            ]
            return self._make_result(defects)

        # Fall back to exact names when no groups are configured.
        required = self.REQUIRED_STEPS
        found_names: set[str] = {s["name"] for s in steps if s["name"] in required}
        missing = sorted(required - found_names)
        defects = [f"缺失步骤「{m}」" for m in missing]
        return self._make_result(defects)


# C3: Valid step labels

class CorrectnessScorer(BaseDimScorer):
    """Check that step labels belong to the allowed set.
    
    Each invalid name counts as one defect. Repetition is handled by C4;
    this rule does not assess the factual or legal correctness of step content.
    """

    name = "C3_步骤标记正确性"
    label = "步骤标记正确性"

    def score(self, text: str, steps: list[dict] = None) -> DimResult:
        if steps is None:
            steps = extract_steps(text)

        if not steps:
            return DimResult(self.name, 0.0, "Fair", 0, ["无步骤标记可检测"])

        valid = self.step_config.valid_names
        defects: list[str] = []

        for s in steps:
            name = s["name"]
            if name not in valid:
                defects.append(f"非法步骤名「{name}」，合法名称为: {sorted(valid)}")

        return self._make_result(defects)


# C4: Non-redundancy

class NonRedundancyScorer(BaseDimScorer):
    """Compare repeated steps with the same name.
    
    Pairs with different content incur one defect; pairs with sufficiently
    similar content do not. Similarity uses normalized edit distance, with
    separate thresholds for short and long text.
    """

    name = "C4_非冗余性"
    label = "非冗余性"

    LONG_SIMILARITY_THRESHOLD = 0.7   # Long-text similarity threshold.
    SHORT_SIMILARITY_THRESHOLD = 0.95  # Short-text similarity threshold.
    SHORT_CONTENT_THRESHOLD = 50       # Short-text length boundary.

    @staticmethod
    def _normalize(text: str) -> str:
        """Remove whitespace before content-similarity comparisons."""
        return re.sub(r'\s+', '', text)

    @staticmethod
    def _similarity(text1: str, text2: str) -> float:
        """Return normalized edit-distance similarity in the range [0, 1].
        
        Long inputs are truncated for comparison. Short inputs use a stricter
        threshold in _is_different to distinguish conflicting values.
        """
        s1 = NonRedundancyScorer._normalize(text1)
        s2 = NonRedundancyScorer._normalize(text2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)

        # Truncate long inputs to the first 500 characters for comparison.
        if max_len > 500:
            s1 = s1[:500]
            s2 = s2[:500]
            len1, len2 = len(s1), len(s2)
            max_len = max(len1, len2)

        # Edit distance.
        dp = list(range(len2 + 1))
        for i in range(1, len1 + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, len2 + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp

        distance = dp[len2]
        return 1.0 - distance / max_len

    def _is_different(self, text1: str, text2: str) -> bool:
        """Apply the short-text or long-text similarity threshold."""
        sim = self._similarity(text1, text2)
        max_len = max(len(self._normalize(text1)), len(self._normalize(text2)))
        threshold = self.SHORT_SIMILARITY_THRESHOLD if max_len < self.SHORT_CONTENT_THRESHOLD else self.LONG_SIMILARITY_THRESHOLD
        return sim < threshold

    def score(self, text: str, steps: list[dict] = None) -> DimResult:
        if steps is None:
            steps = extract_steps(text)

        if not steps:
            return DimResult(self.name, 0.0, "Fair", 0, ["无步骤标记可检测"])

        # Group steps by name.
        name_to_steps: dict[str, list[dict]] = {}
        for s in steps:
            name = s["name"]
            if name not in name_to_steps:
                name_to_steps[name] = []
            name_to_steps[name].append(s)

        defects: list[str] = []
        defect_count = 0

        for name, occurrences in name_to_steps.items():
            if len(occurrences) <= 1:
                continue

            # Compare every pair within the group.
            for i in range(len(occurrences)):
                for j in range(i + 1, len(occurrences)):
                    content_i = occurrences[i]["content"]
                    content_j = occurrences[j]["content"]
                    if self._is_different(content_i, content_j):
                        sim = self._similarity(content_i, content_j)
                        defect_count += 1
                        defects.append(
                            f"冗余冲突: 步骤「{name}」出现 {len(occurrences)} 次，"
                            f"第 {i+1}/{j+1} 次内容相似度仅 {sim:.0%}"
                        )

        return self._make_result(defects, defect_count)


# ============================================================
# 4. Combined scorer
# ============================================================

class ChainScorer:
    """Combine four rule-based chain scorers.
    
    step_config defines task-specific names, groups and penalties. The default
    configuration uses seven legal judgment fields.
    
    Aggregation strategies:
        weighted_sum: Weighted mean, with equal weights by default.
        mean: Arithmetic mean.
        min: Lowest dimensional reward.
    """

    def __init__(self, step_config: StepConfig = None):
        cfg = step_config or StepConfig()
        self.step_config = cfg
        self.dim_scorers: list[BaseDimScorer] = [
            OrderScorer(cfg),
            CompletenessScorer(cfg),
            CorrectnessScorer(cfg),
            NonRedundancyScorer(cfg),
        ]

    def score_C1(self, text: str) -> DimResult:
        steps = extract_steps(text)
        return self.dim_scorers[0].score(text, steps)

    def score_C2(self, text: str) -> DimResult:
        steps = extract_steps(text)
        return self.dim_scorers[1].score(text, steps)

    def score_C3(self, text: str) -> DimResult:
        steps = extract_steps(text)
        return self.dim_scorers[2].score(text, steps)

    def score_C4(self, text: str) -> DimResult:
        steps = extract_steps(text)
        return self.dim_scorers[3].score(text, steps)

    def score(
        self,
        text: str,
        weights: dict[str, float] = None,
        strategy: str = "weighted_sum",
    ) -> ChainScore:
        """Score the structural quality of a marked legal reasoning chain.
        
        Args:
            text: Input containing numbered step markers.
            weights: Optional C1-C4 dimension weights.
            strategy: weighted_sum, mean or min.
        
        Returns:
            A ChainScore containing all dimension results.
        """
        if weights is None:
            weights = {f"C{i}": 1.0 for i in range(1, 5)}

        steps = extract_steps(text)

        dims: dict[str, DimResult] = {
            f"C{i+1}": scorer.score(text, steps)
            for i, scorer in enumerate(self.dim_scorers)
        }

        total_defects = sum(d.defect_count for d in dims.values())

        if strategy == "min":
            overall_reward = min(d.reward for d in dims.values())
        elif strategy == "weighted_sum":
            overall_reward = sum(
                weights.get(k, 1.0) * dims[k].reward for k in dims
            ) / sum(weights.get(k, 1.0) for k in dims)
        else:  # mean
            overall_reward = sum(d.reward for d in dims.values()) / len(dims)

        if overall_reward >= 0.875:
            level = "Excellent"
        elif overall_reward >= 0.25:
            level = "Good"
        elif overall_reward >= -0.25:
            level = "Fair"
        else:
            level = "Poor"

        return ChainScore(
            text=text,
            overall_reward=round(overall_reward, 3),
            overall_level=level,
            total_defects=total_defects,
            dimensions=dims,
            strategy=strategy,
        )


# ============================================================
# 5. Convenience interface
# ============================================================

def score_chain(
    text: str,
    weights: dict[str, float] = None,
    strategy: str = "weighted_sum",
) -> ChainScore:
    """Backward-compatible functional scoring interface."""
    scorer = ChainScorer()
    return scorer.score(text, weights=weights, strategy=strategy)


# ============================================================
# 6. Output formatting
# ============================================================

def format_score(score: ChainScore) -> str:
    """Format a scoring result as readable text."""
    lines = []
    lines.append("=" * 65)
    lines.append("  Chain Reward 评分结果")
    lines.append("=" * 65)
    lines.append(f"  综合得分: {score.overall_reward:+.3f}  ({score.overall_level})")
    lines.append(f"  总缺陷数: {score.total_defects}")
    lines.append(f"  聚合策略: {score.strategy}")
    lines.append("-" * 65)

    dim_names = {
        "C1": "步骤顺序",
        "C2": "步骤完整性",
        "C3": "步骤标记正确性",
        "C4": "非冗余性",
    }

    for key in sorted(score.dimensions.keys()):
        d = score.dimensions[key]
        level_icons = {"Excellent": "+", "Good": "◐", "Fair": "◑", "Poor": "-"}
        icon = level_icons.get(d.level, "?")
        lines.append(
            f"  [{icon}] {dim_names.get(key, key):<16}  "
            f"reward={d.reward:+.1f}  defects={d.defect_count:<2d}  [{d.level}]"
        )
        for defect in d.defects:
            lines.append(f"       ⤷ {defect}")
        lines.append("")

    lines.append("-" * 65)

    worst = max(score.dimensions.values(), key=lambda d: d.defect_count)
    if worst.defect_count > 0:
        lines.append(f"  最需改进的维度: {dim_names.get(worst.dimension[:2], worst.dimension)} ({worst.defect_count} 处缺陷)")
    else:
        lines.append(f"  所有维度均无缺陷")

    lines.append("=" * 65)
    return "\n".join(lines)


# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Rule-based legal reasoning chain scorer')
    parser.add_argument(
        "text", nargs="?", default=None,
        help='Legal text supplied as a string'
    )
    parser.add_argument(
        "--file", "-f", default=None,
        help='Read legal text from a UTF-8 file'
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help='Read text interactively'
    )
    parser.add_argument(
        "--strategy", "-s", default="weighted_sum",
        choices=["weighted_sum", "mean", "min"],
        help='Aggregation strategy (default: weighted_sum)'
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help='Output the result as JSON'
    )
    parser.add_argument(
        "--weights", "-w", default=None,
        help='JSON dimension weights, for example C1=2.0 and C2=1.5'
    )
    parser.add_argument(
        "--dims", "-d", default=None,
        help='Comma-separated dimensions to evaluate, for example C1,C2'
    )
    parser.add_argument(
        "--jsonl", "-l", default=None,
        help='Evaluate JSONL preference pairs with gold/generated fields'
    )

    args = parser.parse_args()

    weights = None
    if args.weights:
        weights = json.loads(args.weights)

    scorer = ChainScorer()

    # Batch evaluation of JSONL preference pairs.
    if args.jsonl:
        import statistics as _st
        samples = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if not samples:
            print("错误: JSONL 文件为空", file=sys.stderr)
            sys.exit(1)

        dim_set = None
        if args.dims:
            dim_set = set(d.strip() for d in args.dims.split(","))

        gw, gnw, tw = 0, 0, 0
        gold_vals, gen_vals = [], []

        print(f"[INFO] 评估 {len(samples)} 条样本...", file=sys.stderr)
        for i, s in enumerate(samples):
            if dim_set:
                gr = sum(
                    getattr(scorer, f"score_{dk}")(s["gold"]).reward
                    for dk in sorted(dim_set)
                ) / len(dim_set)
                gnr = sum(
                    getattr(scorer, f"score_{dk}")(s["generated"]).reward
                    for dk in sorted(dim_set)
                ) / len(dim_set)
            else:
                gr = scorer.score(s["gold"]).overall_reward
                gnr = scorer.score(s["generated"]).overall_reward
            gold_vals.append(gr)
            gen_vals.append(gnr)
            if gr > gnr:
                gw += 1
            elif gnr > gr:
                gnw += 1
            else:
                tw += 1
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(samples)}...", file=sys.stderr)

        n = len(samples)
        print()
        print(f"  Gold 胜:       {gw} ({gw/n*100:.1f}%)")
        print(f"  Generated 胜:  {gnw} ({gnw/n*100:.1f}%)")
        print(f"  平局:          {tw} ({tw/n*100:.1f}%)")
        print(f"  Gold 均分:     {_st.mean(gold_vals):+.4f}")
        print(f"  Generated 均分: {_st.mean(gen_vals):+.4f}")
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

    # Compute scores.
    dim_set = None
    if args.dims:
        dim_set = set(d.strip() for d in args.dims.split(","))
        valid_dims = {f"C{i}" for i in range(1, 5)}
        invalid = dim_set - valid_dims
        if invalid:
            print(f"错误: 无效维度 {sorted(invalid)}，可选: C1-C4", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] 评估中... (文本长度: {len(text)} 字)", file=sys.stderr)
    if dim_set:
        dims = {}
        for dk in sorted(dim_set):
            dims[dk] = getattr(scorer, f"score_{dk}")(text)
        total_defects = sum(d.defect_count for d in dims.values())
        overall_reward = sum(d.reward for d in dims.values()) / len(dims)
        if overall_reward >= 0.875:
            level = "Excellent"
        elif overall_reward >= 0.25:
            level = "Good"
        elif overall_reward >= -0.25:
            level = "Fair"
        else:
            level = "Poor"
        result = ChainScore(
            text=text,
            overall_reward=round(overall_reward, 3),
            overall_level=level,
            total_defects=total_defects,
            dimensions=dims,
            strategy="selected_dims",
        )
    else:
        result = score_chain(text, weights=weights, strategy=args.strategy)

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
