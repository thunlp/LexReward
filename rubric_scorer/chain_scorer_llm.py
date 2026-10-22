#!/usr/bin/env python3
"""LLM-based subjective scoring of legal reasoning chains on a 0-10 scale.

The model identifies reasoning steps and scores four dimensions. Python
validates the response and aggregates the scores. This self-contained module
uses an OpenAI-compatible chat-completions API. Credentials and the base URL
must be supplied through CLI arguments or environment variables.

Examples:
    python scorer/chain_scorer_llm.py --file input.txt --json
    python scorer/chain_scorer_llm.py --jsonl data.jsonl --concurrency 8
"""

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field


# ============================================================
# Scoring dimensions and the self-contained LLM interface
# ============================================================

DIMENSION_NAMES: dict[str, str] = {
    "C1": "逻辑顺序",
    "C2": "推理完整性",
    "C3": "步骤正确性",
    "C4": "非冗余性",
}

# API credentials and endpoint configuration are supplied externally.
# Explicit arguments take precedence over environment configuration.
DEFAULT_MODEL = os.getenv("JUDGE_MODEL") or "deepseek-v4-flash"


def _get_client(api_key: str = None, base_url: str = None):
    from openai import AsyncOpenAI
    key = (api_key or os.getenv("DEEPSEEK_API_KEY")
           or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    url = (base_url or os.getenv("OPENAI_BASE_URL")
           or os.getenv("BASE_URL") or os.getenv("API_URL"))
    if not key or not url:
        missing = " and ".join(
            name for name, val in (("API Key", key), ("API Base URL", url)) if not val
        )
        raise ValueError(
            f"Missing API configuration: {missing}. Set CLI arguments "
            "--api-key / --base-url, or environment variables "
            "OPENAI_API_KEY / OPENAI_BASE_URL "
            "(DEEPSEEK_API_KEY / BASE_URL are also supported)."
        )
    return AsyncOpenAI(api_key=key, base_url=url)


async def _llm_call(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_key: str = None,
    base_url: str = None,
) -> str:
    client = _get_client(api_key, base_url)
    resp = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content


async def _llm_call_batch(
    messages_list: list[list[dict]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    concurrency: int = 2,
    api_key: str = None,
    base_url: str = None,
) -> list[str]:
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(concurrency)

    async def _limited(msgs):
        async with semaphore:
            return await _llm_call(msgs, model, temperature, max_tokens, api_key, base_url)

    tasks = [_limited(msgs) for msgs in messages_list]
    return await asyncio.gather(*tasks)


# ============================================================
# LLM prompts for identifying reasoning steps
# ============================================================

STEP_IDENTIFICATION_SYSTEM = """你是法律推理链分析专家。请根据文本实际内容，自由识别其中隐含的推理步骤。

不要套用固定模板，不要把步骤限制为原告、被告、纠纷类型等预设名称，也不要给步骤分组。
步骤名称应简短、具体并概括该段在当前推理中的作用，例如“确认合同成立”“认定违约行为”
“计算逾期利息”“排除免责事由”“得出赔偿结论”。不同案件可以有完全不同的步骤。

识别原则：
1. 全部步骤属于同一条有先后关系的推理链。
2. 事实、证据、规则适用、中间推论和最终结论均可成为独立步骤。
3. 不要把一个句子机械地当作一个步骤；连续表达同一推理作用的内容应合并。
4. 同一推理作用在不同位置重复出现时仍分别记录，以便后续检查重复或冲突。
5. 只识别文本中实际出现的内容，不补写缺失步骤。"""

STEP_IDENTIFICATION_USER = """请分析以下法律文本，识别其中包含的推理步骤。

对于每个你能识别到的步骤，输出 JSON 格式：
{{
  "steps": [
    {{
      "name": "步骤名称",
      "position": 出现位置序号（从1开始，按文本中出现顺序），
      "content_summary": "该步骤内容的简要概括（不超过100字）",
      "content_matched": "原文中对应的关键句（尽量引用原文）"
    }}
  ]
}}

注意：
1. 按文本中出现的顺序排列
2. 某个步骤可能在文本中多次出现，每次都要记录
3. name 由你根据该文本自由概括，不受固定名称集合限制
4. 如果某个步骤的内容散布在多处，记录其首次出现
5. 如果无法识别出任何推理步骤，返回空列表

--- 法律文本 ---
{text}
--- 文本结束 ---"""


# ============================================================
# LLM-based step identification
# ============================================================

async def identify_steps(text: str, model: str = DEFAULT_MODEL,
                         api_key: str = None, base_url: str = None) -> list[dict]:
    """Identify reasoning steps with an LLM. Return step dictionaries containing
    name, position, content_summary and content_matched."""
    messages = [
        {"role": "system", "content": STEP_IDENTIFICATION_SYSTEM},
        {"role": "user", "content": STEP_IDENTIFICATION_USER.format(text=text)},
    ]
    raw = await _llm_call(messages, model=model, temperature=0.1, max_tokens=4096,
                          api_key=api_key, base_url=base_url)
    return _extract_json_steps(raw)


async def identify_steps_batch(
    texts: list[str], model: str = DEFAULT_MODEL, concurrency: int = 8,
    api_key: str = None, base_url: str = None,
) -> list[list[dict]]:
    """Identify reasoning steps for a batch of texts."""
    messages_list = [
        [
            {"role": "system", "content": STEP_IDENTIFICATION_SYSTEM},
            {"role": "user", "content": STEP_IDENTIFICATION_USER.format(text=t)},
        ]
        for t in texts
    ]
    raws = await _llm_call_batch(messages_list, model=model, concurrency=concurrency,
                                 api_key=api_key, base_url=base_url)
    return [_extract_json_steps(r) for r in raws]


def _extract_json_steps(raw: str) -> list[dict]:
    """Extract a step list from a JSON object in the model response."""
    # Try a fenced JSON object.
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return _validate_steps(obj)
        except json.JSONDecodeError:
            pass

    # Try parsing the entire response as JSON.
    try:
        obj = json.loads(raw.strip())
        return _validate_steps(obj)
    except json.JSONDecodeError:
        pass

    # Try the content between the first opening and last closing brace.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return _validate_steps(obj)
        except json.JSONDecodeError:
            pass

    return []


def _validate_steps(obj: object) -> list[dict]:
    """Validate the response shape before formatting steps for the next prompt."""
    if not isinstance(obj, dict):
        raise ValueError("Step identification must return a JSON object")
    steps = obj.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("The steps field must be an array of objects")
    return steps


def _extract_json_object(raw: str) -> dict:
    """Parse a JSON object from plain JSON, fenced code or surrounding prose."""
    candidates = []
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    candidates.append(raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def _format_steps_for_prompt(steps: list[dict]) -> str:
    if not steps:
        return "（未识别出步骤）"
    return "\n".join(
        f"位置{s.get('position', i)}. 【{s.get('name', '未命名步骤')}】"
        f"{s.get('content_summary', '')}\n对应原文：{s.get('content_matched', '')}"
        for i, s in enumerate(steps, 1)
    )


SUBJECTIVE_SCORING_SYSTEM = """你是严谨的法律推理链评审专家。请阅读原文和从原文中识别出的
推理步骤，对整条推理链进行主观质量评分。

请独立评估以下四个维度：
- C1 逻辑顺序：前提、事实、规则、分析和结论的先后关系是否合理。
- C2 推理完整性：从已有前提到文本结论是否包含必要的关键推理环节。
- C3 步骤正确性：步骤概括是否忠实于原文，推论是否得到事实、前提或法律依据支持。
- C4 非冗余性：是否简洁有效，不存在无意义重复、循环论证或实质矛盾。

评分标尺（允许一位小数）：
- 9.0-10.0：优秀，几乎没有实质问题。
- 7.0-8.9：良好，存在少量问题，但不影响主要推理。
- 5.0-6.9：一般，存在明显问题，部分影响推理质量。
- 3.0-4.9：较差，存在严重或多处关键问题。
- 0.0-2.9：很差，推理链基本无效或无法成立。

要求：
1. 分数必须是 0 到 10 之间的数字，不要使用百分制或正负奖励分。
2. 每个维度必须单独判断，不要因为某一维度较差而机械地给所有维度低分。
3. 只根据给出的文本评分，不要求无关的法律文书模板字段。
4. reason 要具体说明给分依据；evidence 列出最关键的问题或优点，最多 3 条。
5. 不要自行输出综合分，综合分将由程序计算。
6. 只输出合法 JSON，不要输出 Markdown 或额外说明。"""


SUBJECTIVE_SCORING_USER = """请对以下法律文本的推理链进行 0-10 分主观评分。

严格使用以下 JSON 格式：
{{
  "scores": {{
    "C1": {{"score": 8.5, "reason": "评分理由", "evidence": ["关键依据"]}},
    "C2": {{"score": 7.0, "reason": "评分理由", "evidence": ["关键依据"]}},
    "C3": {{"score": 9.0, "reason": "评分理由", "evidence": ["关键依据"]}},
    "C4": {{"score": 8.0, "reason": "评分理由", "evidence": ["关键依据"]}}
  }}
}}

--- 原始文本 ---
{text}
--- 识别出的步骤 ---
{steps_text}
--- 结束 ---"""


@dataclass
class SubjectiveDimensionScore:
    dimension: str
    score: float
    level: str
    reason: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class SubjectiveChainScore:
    text: str
    overall_score: float
    overall_level: str
    dimensions: dict[str, SubjectiveDimensionScore]
    strategy: str
    weights: dict[str, float]
    identified_steps: list[dict] = field(default_factory=list)


def level_from_score(score: float) -> str:
    """Map a score on the 0-10 scale to a readable quality level."""
    if score >= 9.0:
        return "Excellent"
    if score >= 7.0:
        return "Good"
    if score >= 5.0:
        return "Fair"
    if score >= 3.0:
        return "Poor"
    return "Very Poor"


def _parse_subjective_scores(raw: str) -> dict[str, SubjectiveDimensionScore]:
    """Parse and strictly validate all four model-generated scores."""
    obj = _extract_json_object(raw)
    source = obj.get("scores") if isinstance(obj, dict) else None
    if not isinstance(source, dict):
        raise ValueError("LLM 评分结果缺少 scores 对象")

    result: dict[str, SubjectiveDimensionScore] = {}
    errors: list[str] = []
    for key, name in DIMENSION_NAMES.items():
        item = source.get(key)
        if not isinstance(item, dict):
            errors.append(f"{key} 缺失或不是对象")
            continue

        value = item.get("score")
        if isinstance(value, bool):
            errors.append(f"{key}.score 不是有效数字")
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            errors.append(f"{key}.score 不是有效数字")
            continue
        if not math.isfinite(score) or not 0.0 <= score <= 10.0:
            errors.append(f"{key}.score={value!r} 超出 0-10 范围")
            continue

        reason = str(item.get("reason", "")).strip()
        if not reason:
            errors.append(f"{key}.reason 为空")
            continue

        evidence = item.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            errors.append(f"{key}.evidence 不是数组")
            continue

        result[key] = SubjectiveDimensionScore(
            dimension=f"{key}_{name}",
            score=round(score, 1),
            level=level_from_score(score),
            reason=reason,
            evidence=[str(entry).strip() for entry in evidence[:3] if str(entry).strip()],
        )

    if errors:
        raise ValueError("LLM 评分结果格式无效：" + "；".join(errors))
    return result


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {key: 1.0 for key in DIMENSION_NAMES}
    if not isinstance(weights, dict):
        raise ValueError("weights 必须是包含 C1-C4 的 JSON 对象")

    normalized: dict[str, float] = {}
    for key in DIMENSION_NAMES:
        value = weights.get(key, 1.0)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重 {key} 必须是数字") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"权重 {key} 必须是非负有限数字")
        normalized[key] = value
    if sum(normalized.values()) <= 0:
        raise ValueError("维度权重之和必须大于 0")
    return normalized


def _aggregate(
    dimensions: dict[str, SubjectiveDimensionScore],
    weights: dict[str, float],
    strategy: str,
) -> float:
    if strategy == "min":
        return min(item.score for item in dimensions.values())
    return sum(weights[key] * dimensions[key].score for key in DIMENSION_NAMES) / sum(
        weights.values()
    )


def _evaluation_messages(text: str, steps: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SUBJECTIVE_SCORING_SYSTEM},
        {
            "role": "user",
            "content": SUBJECTIVE_SCORING_USER.format(
                text=text, steps_text=_format_steps_for_prompt(steps)
            ),
        },
    ]


class LLMChainSubjectiveScorer:
    """Request four subjective scores on a 0-10 scale and aggregate them."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        concurrency: int = 8,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.model = model
        self.concurrency = concurrency
        self.api_key = api_key
        self.base_url = base_url

    async def score(
        self,
        text: str,
        weights: dict[str, float] | None = None,
        strategy: str = "weighted_mean",
    ) -> SubjectiveChainScore:
        if not text.strip():
            raise ValueError("输入文本为空")
        if strategy not in {"weighted_mean", "min"}:
            raise ValueError("strategy 必须是 weighted_mean 或 min")
        normalized_weights = _normalize_weights(weights)
        steps = await identify_steps(
            text, model=self.model, api_key=self.api_key, base_url=self.base_url
        )
        raw = await _llm_call(
            _evaluation_messages(text, steps),
            model=self.model,
            temperature=0.1,
            max_tokens=4096,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        dimensions = _parse_subjective_scores(raw)
        overall = round(_aggregate(dimensions, normalized_weights, strategy), 2)
        return SubjectiveChainScore(
            text=text,
            overall_score=overall,
            overall_level=level_from_score(overall),
            dimensions=dimensions,
            strategy=strategy,
            weights=normalized_weights,
            identified_steps=steps,
        )

    async def score_batch(
        self,
        texts: list[str],
        weights: dict[str, float] | None = None,
        strategy: str = "weighted_mean",
    ) -> list[SubjectiveChainScore]:
        if any(not text.strip() for text in texts):
            raise ValueError("批量输入中存在空文本")
        if strategy not in {"weighted_mean", "min"}:
            raise ValueError("strategy 必须是 weighted_mean 或 min")
        normalized_weights = _normalize_weights(weights)
        all_steps = await identify_steps_batch(
            texts,
            model=self.model,
            concurrency=self.concurrency,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        raws = await _llm_call_batch(
            [_evaluation_messages(text, steps) for text, steps in zip(texts, all_steps)],
            model=self.model,
            temperature=0.1,
            max_tokens=4096,
            concurrency=self.concurrency,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        results = []
        for index, (text, steps, raw) in enumerate(zip(texts, all_steps, raws), 1):
            try:
                dimensions = _parse_subjective_scores(raw)
            except ValueError as exc:
                raise ValueError(f"第 {index} 条样本解析失败：{exc}") from exc
            overall = round(_aggregate(dimensions, normalized_weights, strategy), 2)
            results.append(
                SubjectiveChainScore(
                    text=text,
                    overall_score=overall,
                    overall_level=level_from_score(overall),
                    dimensions=dimensions,
                    strategy=strategy,
                    weights=normalized_weights,
                    identified_steps=steps,
                )
            )
        return results


def score_chain_subjective(
    text: str,
    model: str = DEFAULT_MODEL,
    weights: dict[str, float] | None = None,
    strategy: str = "weighted_mean",
    api_key: str | None = None,
    base_url: str | None = None,
) -> SubjectiveChainScore:
    scorer = LLMChainSubjectiveScorer(
        model=model, api_key=api_key, base_url=base_url
    )
    return asyncio.run(scorer.score(text, weights=weights, strategy=strategy))


def _result_to_dict(result: SubjectiveChainScore) -> dict:
    return asdict(result)


def format_score(result: SubjectiveChainScore) -> str:
    lines = [
        "=" * 68,
        "  法律推理链 LLM 主观评分（0-10 分）",
        "=" * 68,
        f"  综合得分: {result.overall_score:.2f}/10  [{result.overall_level}]",
        f"  聚合策略: {result.strategy}",
        "-" * 68,
        f"  识别出的推理步骤 ({len(result.identified_steps)}):",
    ]
    if result.identified_steps:
        for index, step in enumerate(result.identified_steps, 1):
            lines.append(
                f"    {step.get('position', index)}. "
                f"【{step.get('name', '未命名步骤')}】{step.get('content_summary', '')}"
            )
    else:
        lines.append("    （未识别出推理步骤）")

    lines.append("-" * 68)
    for key in DIMENSION_NAMES:
        item = result.dimensions[key]
        lines.append(
            f"  {key} {DIMENSION_NAMES[key]}: {item.score:.1f}/10  [{item.level}]"
        )
        lines.append(f"    理由: {item.reason}")
        for evidence in item.evidence:
            lines.append(f"    - {evidence}")
    lines.append("=" * 68)
    return "\n".join(lines)


def _read_jsonl(path: str) -> tuple[list[dict], list[str]]:
    samples = []
    texts = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            text = next(
                (sample[key] for key in ("text", "generated", "output") if key in sample),
                None,
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"JSONL 第 {line_number} 行缺少非空 text/generated/output 字段")
            samples.append(sample)
            texts.append(text)
    if not samples:
        raise ValueError("JSONL 文件为空")
    return samples, texts


async def _main_batch(args, weights: dict[str, float] | None) -> None:
    samples, texts = _read_jsonl(args.jsonl)
    scorer = LLMChainSubjectiveScorer(
        model=args.model,
        concurrency=args.concurrency,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    print(
        f"[INFO] 正在评估 {len(texts)} 条样本（模型: {args.model}）...",
        file=sys.stderr,
    )
    results = await scorer.score_batch(texts, weights=weights, strategy=args.strategy)

    overall_scores = [result.overall_score for result in results]
    print(f"Overall: mean={statistics.mean(overall_scores):.3f}, "
          f"std={statistics.pstdev(overall_scores):.3f}")
    for key in DIMENSION_NAMES:
        values = [result.dimensions[key].score for result in results]
        print(f"{key} {DIMENSION_NAMES[key]}: mean={statistics.mean(values):.3f}")

    stem, extension = os.path.splitext(args.jsonl)
    output_path = f"{stem}_subjective_scored{extension or '.jsonl'}"
    with open(output_path, "w", encoding="utf-8") as file:
        for sample, result in zip(samples, results):
            output = {**sample, "llm_subjective_scores": _result_to_dict(result)}
            file.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"[INFO] 详细结果已保存至: {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description='Subjective LLM scoring of legal reasoning chains (0-10)')
    parser.add_argument("text", nargs="?", help='Legal text to evaluate')
    parser.add_argument("--file", "-f", help='Read text from a UTF-8 file')
    parser.add_argument("--jsonl", "-l", help='Evaluate JSONL records with text/generated/output fields')
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help='Judge model name')
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["weighted_mean", "min"],
        default="weighted_mean",
        help='Score aggregation strategy',
    )
    parser.add_argument(
        "--weights",
        "-w",
        help='JSON weights for dimensions C1-C4',
    )
    parser.add_argument("--concurrency", "-c", type=int, default=2, help='Number of concurrent requests')
    parser.add_argument("--json", "-j", action="store_true", help='Output one result as JSON')
    parser.add_argument("--api-key", help="API key; defaults to environment configuration")
    parser.add_argument("--base-url", help="API Base URL")
    args = parser.parse_args()

    try:
        weights = json.loads(args.weights) if args.weights else None
        _normalize_weights(weights)
        if args.concurrency < 1:
            raise ValueError("concurrency 必须大于等于 1")

        if args.jsonl:
            asyncio.run(_main_batch(args, weights))
            return
        if args.file:
            with open(args.file, "r", encoding="utf-8") as file:
                text = file.read().strip()
        else:
            text = (args.text or "").strip()
        if not text:
            parser.error("请提供文本参数、--file 或 --jsonl")

        print(
            f"[INFO] 评分中（文本长度: {len(text)} 字，模型: {args.model}）...",
            file=sys.stderr,
        )
        result = score_chain_subjective(
            text,
            model=args.model,
            weights=weights,
            strategy=args.strategy,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        if args.json:
            print(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))
        else:
            print(format_score(result))
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
