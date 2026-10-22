#!/usr/bin/env python3
"""Build LexChain RM data from a scored five-model pool at a 5:3:2 ratio.

Shuffle the pool with a fixed seed and select 4000 rows by default.
A pairs r1/r2, B pairs r2/r4, and C pairs r1/r5. Preserve the original
LexChain model metadata, score ordering and row-ID tie-breaking rules.
"""
import argparse
import json
import math
import random
from pathlib import Path


def score_values(row):
    """Align five finite judge scores with candidate indices 0 through 4."""
    scores = row["candidate_scores"]
    if len(scores) != 5:
        raise ValueError(f"row {row['row_id']} must have five scores")
    by_index = {int(item["candidate_index"]): float(item["overall"]) for item in scores}
    if len(by_index) != 5 or set(by_index) != set(range(5)):
        raise ValueError(f"row {row['row_id']} has invalid candidate indices")
    if not all(math.isfinite(value) for value in by_index.values()):
        raise ValueError("candidate scores must be finite")
    return [by_index[i] for i in range(5)]


def pair(row, chosen_index, rejected_index, tier):
    scores = score_values(row)
    return {
        "chosen": row["query"] + [{"role": "assistant", "content": row["candidates"][chosen_index]}],
        "rejected": row["query"] + [{"role": "assistant", "content": row["candidates"][rejected_index]}],
        "chosen_model": row["candidate_models"][chosen_index],
        "rejected_model": row["candidate_models"][rejected_index],
        "chosen_score": scores[chosen_index],
        "rejected_score": scores[rejected_index],
        "meta": {"source_row_id": row["row_id"], "tier": tier,
                 "tier_ratio": "5:3:2", "chosen_index": chosen_index,
                 "rejected_index": rejected_index, "row_scores": scores,
                 "judge_model": row.get("judge_model")},
    }


def build_pairs(rows, num_pairs=4000, seed=42):
    """Select a reproducible subset and return exact 5:3:2 ranked pairs."""
    if num_pairs <= 0 or num_pairs % 10:
        raise ValueError("num_pairs must be a positive multiple of 10")
    if len(rows) < num_pairs:
        raise ValueError(f"need at least {num_pairs} pool rows, received {len(rows)}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("row_id values must be unique")
    for row in rows:
        if len(row["candidates"]) != 5 or len(row["candidate_models"]) != 5:
            raise ValueError(f"row {row['row_id']} must contain five candidates and model names")
        if not isinstance(row["query"], list) or not any(
                isinstance(message, dict) and message.get("role") == "user" for message in row["query"]):
            raise ValueError("query must contain a user message")
        if any(not isinstance(answer, str) or not answer.strip() for answer in row["candidates"]):
            raise ValueError("candidate answers must be nonempty strings")
        score_values(row)

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    ranked = []
    for row in shuffled[:num_pairs]:
        scores = score_values(row)
        order = sorted(range(5), key=lambda i: (-scores[i], i))
        ranked.append((row, scores, order))
    count_a, count_b = num_pairs * 5 // 10, num_pairs * 3 // 10
    sorted_a = sorted(ranked, key=lambda item: (
        item[1][item[2][0]] - item[1][item[2][1]], item[0]["row_id"]))
    tier_a = sorted_a[:count_a]
    sorted_b = sorted(sorted_a[count_a:], key=lambda item: (
        item[1][item[2][1]] - item[1][item[2][3]], item[0]["row_id"]))
    tier_b, tier_c = sorted_b[:count_b], sorted_b[count_b:]
    result = [pair(row, order[0], order[1], "A") for row, _, order in tier_a]
    result.extend(pair(row, order[1], order[3], "B") for row, _, order in tier_b)
    result.extend(pair(row, order[0], order[4], "C") for row, _, order in tier_c)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Scored LexChain five-model JSONL pool")
    parser.add_argument("--output", type=Path, required=True, help="Output RM preference JSONL")
    parser.add_argument("--num-pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    try:
        with args.input.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        pairs = build_pairs(rows, args.num_pairs, args.seed)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for record in pairs:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {tier: sum(row["meta"]["tier"] == tier for row in pairs) for tier in ("A", "B", "C")}
    print(json.dumps({"pairs": len(pairs), "ratio": "5:3:2", "tiers": counts, "seed": args.seed}))


if __name__ == "__main__":
    main()
