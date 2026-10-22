#!/usr/bin/env python3
"""Build Element RM pairs from scored JEC-QA and LexChain pools at 5:3:2 ratio.

This constructor differs from CLASE and LexChain in two ways:

1. Tier assignment uses score gaps, not ranks. Element rubric scores are
   discrete (0.0, 0.1, ..., 10.0), causing 66% of adjacent-rank pairs to
   tie. Tiers are assigned by gap magnitude rather than rank position to
   avoid placing all tied pairs in tier A.

2. One pair per query. When multiple candidate pairs fall in the same tier
   for a query, the constructor prefers the pair whose chosen and rejected
   responses contain different answer letters, which improves signal quality
   for legal choice questions.

Tier definitions:
  A (50%): smallest chosen-rejected gaps among eligible queries.
  B (30%): medium gaps among queries not in A.
  C (20%): largest gaps among queries not in A or B.

Balancing constraints control length ratio, on-policy share, rejected-side
model concentration, and model-family skew to mitigate spurious shortcuts.

No model or API is loaded. The pool must contain scored five-candidate data.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def answer_letters(text):
    """Extract answer option letters (A-F) from legal response text."""
    matches = re.findall(r'\b([A-F])\b', text)
    return ''.join(sorted(set(matches))) if matches else ''


def family(model):
    """Map model name to family: llama, qwen, gemma, or other."""
    m = model.lower()
    for fam in ('llama', 'qwen', 'gemma'):
        if fam in m:
            return fam
    return 'other'


def features(pair, onpolicy_model='Qwen3-8B'):
    """Extract balancing features from a candidate pair."""
    c_text = pair['chosen'][-1]['content']
    r_text = pair['rejected'][-1]['content']
    c_letters = answer_letters(c_text)
    r_letters = answer_letters(r_text)
    return {
        'len_chosen_longer': (len(c_text) > len(r_text)) if len(c_text) != len(r_text) else None,
        'onpolicy_chosen': pair['meta']['chosen_model'] == onpolicy_model,
        'rejected_model': pair['meta']['rejected_model'],
        'rejected_family': family(pair['meta']['rejected_model']),
        'letters_differ': bool(c_letters and r_letters and c_letters != r_letters),
    }


class Balancer:
    """Greedy balancer that selects one pair per query to minimize global skew."""

    def __init__(self, target_len=0.5, target_on=0.5, cap_model=0.25,
                 max_family_skew=0.10, letter_bonus=0.02):
        self.t_len = target_len
        self.t_on = target_on
        self.cap_model = cap_model
        self.max_family_skew = max_family_skew
        self.letter_bonus = letter_bonus
        self.len_hit = self.len_tot = 0
        self.on_hit = self.total = 0
        self.rej_model = Counter()
        self.rej_family = Counter()

    def cost(self, feat):
        """Compute deviation cost for adding this pair."""
        c = 0.0
        if feat['len_chosen_longer'] is not None:
            new_len = (self.len_hit + feat['len_chosen_longer']) / (self.len_tot + 1)
            c += abs(new_len - self.t_len)
        new_on = (self.on_hit + feat['onpolicy_chosen']) / (self.total + 1)
        c += abs(new_on - self.t_on)
        model_share = (self.rej_model[feat['rejected_model']] + 1) / (self.total + 1)
        if model_share > self.cap_model:
            c += (model_share - self.cap_model) * 2
        fam_share = (self.rej_family[feat['rejected_family']] + 1) / (self.total + 1)
        if fam_share > 0.5 + self.max_family_skew:
            c += (fam_share - 0.5 - self.max_family_skew) * 2
        if feat['letters_differ']:
            c -= self.letter_bonus
        return c

    def select(self, candidates):
        """Select the best candidate pair for this query."""
        if not candidates:
            raise ValueError('no candidate pairs for query')
        feat_list = [features(p) for p in candidates]
        costs = [self.cost(f) for f in feat_list]
        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        best = candidates[best_idx]
        best_feat = feat_list[best_idx]
        if best_feat['len_chosen_longer'] is not None:
            self.len_hit += best_feat['len_chosen_longer']
            self.len_tot += 1
        self.on_hit += best_feat['onpolicy_chosen']
        self.total += 1
        self.rej_model[best_feat['rejected_model']] += 1
        self.rej_family[best_feat['rejected_family']] += 1
        return best


def prepare_row(row):
    """Extract candidates, scores, order, and prompt from a pool row."""
    candidates = row['candidates']
    scores_obj = row['candidate_scores']
    if len(candidates) != 5 or len(scores_obj) != 5:
        raise ValueError('each row must have exactly five candidates and scores')
    scores = []
    for obj in scores_obj:
        if isinstance(obj, dict):
            val = obj.get('overall') or obj.get('overall_score') or obj.get('score')
            if val is None:
                raise ValueError('score object missing overall/overall_score/score')
        else:
            val = obj
        scores.append(float(val))
    order = sorted(range(5), key=lambda i: (-scores[i], i))
    prompt = row.get('query') or row.get('messages') or row.get('instruction')
    if isinstance(prompt, str):
        prompt = [{'role': 'user', 'content': prompt}]
    elif not isinstance(prompt, list):
        raise ValueError('missing query/messages/instruction')
    return candidates, scores, order, prompt, row.get('uniqid', f"row_{row.get('row_id', 0)}")


def build_candidate_pairs(rows):
    """Generate all valid candidate pairs per row with tier and gap."""
    result = []
    for idx, row in enumerate(rows):
        try:
            candidates_list, scores, order, prompt, uniqid = prepare_row(row)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"skip row {idx}: {exc}", file=sys.stderr)
            continue

        def text(i):
            val = candidates_list[i]
            return val['answer'] if isinstance(val, dict) else str(val)

        def make_pair(chosen_rank, rejected_rank):
            c_idx, r_idx = order[chosen_rank], order[rejected_rank]
            gap = scores[c_idx] - scores[r_idx]
            meta = {
                'source_index': idx,
                'chosen_model': row['candidate_models'][c_idx],
                'rejected_model': row['candidate_models'][r_idx],
                'chosen_score': scores[c_idx],
                'rejected_score': scores[r_idx],
                'score_margin': gap,
            }
            return {
                'chosen': prompt + [{'role': 'assistant', 'content': text(c_idx)}],
                'rejected': prompt + [{'role': 'assistant', 'content': text(r_idx)}],
                'meta': meta,
                'gap': gap,
                'uniqid': uniqid,
            }

        # Generate (r1,r2), (r2,r4), (r1,r5) candidate pairs
        pairs = []
        if scores[order[0]] > scores[order[1]]:
            pairs.append(('A', make_pair(0, 1)))
        if scores[order[1]] > scores[order[3]]:
            pairs.append(('B', make_pair(1, 3)))
        if scores[order[0]] > scores[order[4]]:
            pairs.append(('C', make_pair(0, 4)))

        for tier, pair in pairs:
            result.append((tier, pair))
    return result


def assign_tiers(candidate_pairs, ratio=(5, 3, 2)):
    """Assign tiers by gap magnitude, one pair per query."""
    total = sum(ratio)
    count_a = len(candidate_pairs) * ratio[0] // total
    count_b = len(candidate_pairs) * ratio[1] // total

    by_tier = {'A': [], 'B': [], 'C': []}
    for tier, pair in candidate_pairs:
        by_tier[tier].append(pair)

    # Sort each tier pool by gap, then by uniqid
    for t in by_tier:
        by_tier[t].sort(key=lambda p: (p['gap'], p['uniqid']))

    assigned = {}
    tier_a_pairs = by_tier['A'][:count_a]
    for p in tier_a_pairs:
        assigned[p['uniqid']] = ('A', p)

    remaining_b = [p for p in by_tier['B'] if p['uniqid'] not in assigned]
    tier_b_pairs = remaining_b[:count_b]
    for p in tier_b_pairs:
        assigned[p['uniqid']] = ('B', p)

    remaining_c = [p for p in by_tier['C'] if p['uniqid'] not in assigned]
    for p in remaining_c:
        assigned[p['uniqid']] = ('C', p)

    return assigned


def build_pairs(rows, seed=42, ratio=(5, 3, 2)):
    """Build balanced RM pairs from scored pool at the given ratio."""
    if not rows:
        raise ValueError('pool is empty')
    random.seed(seed)
    random.shuffle(rows)

    candidate_pairs = build_candidate_pairs(rows)
    if not candidate_pairs:
        raise ValueError('no valid candidate pairs generated')

    # Group candidates by query
    by_query = {}
    for tier, pair in candidate_pairs:
        uid = pair['uniqid']
        if uid not in by_query:
            by_query[uid] = []
        by_query[uid].append((tier, pair))

    # Assign tier to each query
    flat = [(t, p) for uid, pairs in by_query.items() for t, p in pairs]
    assigned = assign_tiers(flat, ratio=ratio)

    # Select one pair per query with balancing
    balancer = Balancer()
    result = []
    for uid in sorted(by_query.keys()):
        tier, selected = assigned.get(uid, (None, None))
        if tier is None:
            continue
        # Gather all candidate pairs from this query in this tier
        same_tier = [p for t, p in by_query[uid] if t == tier]
        if not same_tier:
            same_tier = [p for t, p in by_query[uid]]
        chosen = balancer.select(same_tier)
        chosen['meta']['tier'] = tier
        chosen['meta']['tier_ratio'] = f"{ratio[0]}:{ratio[1]}:{ratio[2]}"
        chosen['meta']['pool'] = 'element'
        del chosen['gap']
        del chosen['uniqid']
        result.append(chosen)

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True,
                   help='Scored Element five-model JSONL pool')
    p.add_argument('--output', type=Path, required=True,
                   help='Output RM preference pairs in JSONL')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for shuffling queries')
    args = p.parse_args()

    if args.input.resolve() == args.output.resolve():
        p.error('output must differ from input')

    with args.input.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]

    try:
        pairs = build_pairs(rows, seed=args.seed)
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as out:
        for pair in pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + '\n')

    counts = {tier: sum(pair['meta']['tier'] == tier for pair in pairs)
              for tier in ('A', 'B', 'C')}
    print(json.dumps({'pairs': len(pairs), 'ratio': '5:3:2', 'tiers': counts}))


if __name__ == '__main__':
    main()
