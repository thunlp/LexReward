#!/usr/bin/env python3
"""Build CLASE RM pairs from a scored five-model pool at a fixed 5:3:2 ratio.

A: smallest r1-r2 gaps, pairing r1 with r2.
B: smallest r2-r4 gaps among remaining rows, pairing r2 with r4.
C: remaining rows, pairing r1 with r5.

Ranks are descending by supplied Style scores. No model or API is loaded.
"""
import argparse
import json
import math
from pathlib import Path


def score_value(value):
    if isinstance(value, dict):
        for key in ("score", "overall", "overall_reward"):
            if key in value:
                value = value[key]
                break
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("candidate scores must be finite")
    return value


def prepare(row):
    candidates, scores = row["candidates"], row["candidate_scores"]
    if len(candidates) != 5 or len(scores) != 5:
        raise ValueError("each CLASE row must contain exactly five candidates and scores")
    if isinstance(scores[0], dict) and "candidate_index" in scores[0]:
        by_index = {int(item["candidate_index"]): item for item in scores}
        if len(by_index) != 5 or set(by_index) != set(range(5)):
            raise ValueError("candidate indices must be unique and cover 0 through 4")
        scores = [by_index[i] for i in range(5)]
    values = [score_value(value) for value in scores]
    order = sorted(range(5), key=lambda i: (-values[i], i))
    prompt = row.get("query") or row.get("messages") or row.get("chosen")
    if isinstance(prompt, list):
        prompt = [dict(message) for message in prompt if message.get("role") != "assistant"]
    elif isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    else:
        text = row.get("prompt_text") or row.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing query/messages/chosen/prompt")
        prompt = [{"role": "user", "content": text}]
    if not prompt or not any(message.get("role") == "user" for message in prompt):
        raise ValueError("missing user message")
    return candidates, values, order, prompt


def build_pairs(rows, keep_ties=False):
    """Return one pair per row, with exact A/B/C proportions of 5:3:2."""
    if not rows or len(rows) % 10:
        raise ValueError("the pool size must be a positive multiple of 10 for an exact 5:3:2 ratio")
    prepared = [prepare(row) for row in rows]
    count_a, count_b = len(rows) * 5 // 10, len(rows) * 3 // 10

    def gap(index, left, right):
        _, scores, order, _ = prepared[index]
        return scores[order[left]] - scores[order[right]]

    tier_a = sorted(range(len(rows)), key=lambda i: (gap(i, 0, 1), i))[:count_a]
    selected_a = set(tier_a)
    remaining = [i for i in range(len(rows)) if i not in selected_a]
    tier_b = sorted(remaining, key=lambda i: (gap(i, 1, 3), i))[:count_b]
    selected_b = set(tier_b)
    assignments = {i: ("A", 0, 1) for i in tier_a}
    assignments.update({i: ("B", 1, 3) for i in tier_b})
    assignments.update({i: ("C", 0, 4) for i in remaining if i not in selected_b})

    result = []
    for index, row in enumerate(rows):
        candidates, values, order, prompt = prepared[index]
        tier, left, right = assignments[index]
        chosen, rejected = order[left], order[right]
        if not keep_ties and values[chosen] == values[rejected]:
            raise ValueError(
                f"row {row.get('row_id', index)} has tied selected scores; "
                "use --keep-ties to retain ties without changing the 5:3:2 ratio")

        def answer(position):
            value = candidates[position]
            if isinstance(value, dict):
                value = value["content"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("candidate answer must be nonempty text")
            return prompt + [{"role": "assistant", "content": value}]

        meta = dict(row.get("meta") or {})
        meta.update({"source_row_id": row.get("row_id", index), "pair_group": tier,
                     "tier_ratio": "5:3:2", "strategy": "tiered",
                     "chosen_index": chosen, "rejected_index": rejected,
                     "chosen_score": values[chosen], "rejected_score": values[rejected]})
        result.append({"chosen": answer(chosen), "rejected": answer(rejected), "meta": meta})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Scored CLASE five-model JSONL pool")
    parser.add_argument("--output", type=Path, required=True, help="Output RM preference pairs in JSONL")
    parser.add_argument("--keep-ties", action="store_true", help="Keep equal-score pairs instead of failing")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    try:
        with args.input.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        pairs = build_pairs(rows, keep_ties=args.keep_ties)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in pairs:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {tier: sum(row["meta"]["pair_group"] == tier for row in pairs) for tier in ("A", "B", "C")}
    print(json.dumps({"pairs": len(pairs), "ratio": "5:3:2", "tiers": counts}))


if __name__ == "__main__":
    main()
