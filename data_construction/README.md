# RM Data Construction

## Overview

Construct RM preference data from pre-scored CLASE or LexChain five-model candidate pools. Both scripts use the A/B/C ratio of 5:3:2 and run without loading a model or calling an API. All commands below run from the repository root. Replace `/path/to/...` with your own files.

## Files

| Python file | Purpose |
| --- | --- |
| `build_clase_rm_pairs.py` | Convert the entire scored CLASE pool into 5:3:2 preference pairs, preserving source metadata and recording the selected candidate indices. |
| `build_element_rm_pairs.py` | Build Element preference pairs using gap-based tier assignment and balancing constraints, selecting one pair per query from JEC-QA and LexChain scored pools. |
| `build_lexchain_rm_pairs.py` | Shuffle the scored LexChain pool with a fixed seed, select 4000 rows by default, and construct 5:3:2 preference pairs while preserving model names and judge metadata. |

Only these construction methods and this README are included. Candidate generation, other pairing strategies, and dataset files are not bundled.

## Requirements

Use Python 3.11. Both scripts use only the Python standard library and do not need GPU libraries or an API configuration. Prepare scored five-candidate pools before running them.

## Inputs and Outputs

### build_clase_rm_pairs.py

Each JSONL record requires five `candidates` and five `candidate_scores`. Candidates may be strings or objects with a `content` field. Scores may be numbers or objects with `score`, `overall`, or `overall_reward`. If scores include `candidate_index`, indices must uniquely cover 0 through 4.

The prompt can be supplied in `query`, `messages`, `chosen`, `prompt_text`, or `prompt`. Existing assistant messages in the prompt sequence are excluded from the shared prefix of the new pair.

```json
{"row_id": 0, "chosen": [{"role": "user", "content": "Case details and rewriting instructions"}, {"role": "assistant", "content": "Reference answer"}], "candidates": ["Answer A", "Answer B", "Answer C", "Answer D", "Answer E"], "candidate_scores": [{"score": -0.2}, {"score": -0.4}, {"score": -0.8}, {"score": -1.2}, {"score": -1.5}]}
```

Output records contain chat-format `chosen` and `rejected` sequences plus `meta` with `pair_group`, `source_row_id`, `tier_ratio`, candidate indices, and scores. The entire input must contain a positive multiple of 10 rows.

### build_element_rm_pairs.py

Each JSONL record requires a `uniqid`, five `candidates`, five `candidate_models`, and five `candidate_scores` with `overall` or `overall_score`. The prompt can be supplied in `query`, `messages`, or `instruction`.

```json
{"uniqid": "jec_qa:0", "query": [{"role": "user", "content": "Case details and legal question"}], "candidates": [{"answer": "Response A"}, {"answer": "Response B"}, {"answer": "Response C"}, {"answer": "Response D"}, {"answer": "Response E"}], "candidate_models": ["Qwen3-4B", "Qwen3.5-4B", "Qwen3.5-9B", "Llama-3.1-8B-Instruct", "gemma-4-12B-it"], "candidate_scores": [{"overall": 8.5}, {"overall": 7.2}, {"overall": 6.8}, {"overall": 5.1}, {"overall": 3.9}]}
```

Output records contain `chosen`, `rejected`, `meta` with `tier`, `tier_ratio`, `pool`, candidate indices, model names, and scores. Element uses gap-based tier assignment: queries are grouped by the chosen-rejected score gap rather than by candidate ranks.

### build_lexchain_rm_pairs.py

Each JSONL record requires a unique `row_id`, a `query` message sequence, five text `candidates`, five `candidate_models`, and five score objects containing `candidate_index` and `overall`.

```json
{"row_id": 0, "query": [{"role": "user", "content": "Case details and legal question"}], "candidates": ["Answer A", "Answer B", "Answer C", "Answer D", "Answer E"], "candidate_models": ["Model A", "Model B", "Model C", "Model D", "Model E"], "candidate_scores": [{"candidate_index": 0, "overall": 9}, {"candidate_index": 1, "overall": 8}, {"candidate_index": 2, "overall": 7}, {"candidate_index": 3, "overall": 6}, {"candidate_index": 4, "overall": 5}], "judge_model": "JUDGE_MODEL"}
```

Output records contain `chosen`, `rejected`, model names, scores, and `meta` with `tier`, source and candidate indices, all five scores, and the judge model. The input pool must contain at least `--num-pairs` rows. The requested output count must be a positive multiple of 10.

For both methods, rank candidates by descending score as r1 through r5:

| Tier | Share | Selection | Chosen / rejected |
| --- | --- | --- | --- |
| A | 50% | Smallest r1-r2 gaps in the selected pool | r1 / r2 |
| B | 30% | Smallest r2-r4 gaps among rows outside A | r2 / r4 |
| C | 20% | Remaining rows | r1 / r5 |

Every selected case belongs to exactly one tier. A 4000-pair output contains 2000 A, 1200 B, and 800 C pairs. Each script writes the requested JSONL file and prints a count summary.

## Usage

### build_clase_rm_pairs.py

Construct pairs from all rows of a scored CLASE pool:

```bash
python data_construction/build_clase_rm_pairs.py \
    --input /path/to/clase_scored_pool.jsonl \
    --output outputs/clase_rm_pairs_532.jsonl
```

Selected equal-score pairs cause an error by default. Keep them explicitly without changing the tier quotas:

```bash
python data_construction/build_clase_rm_pairs.py \
    --input /path/to/clase_scored_pool.jsonl \
    --output outputs/clase_rm_pairs_532.jsonl \
    --keep-ties
```

### build_element_rm_pairs.py

Construct Element pairs from a scored JEC-QA and LexChain pool:

```bash
python data_construction/build_element_rm_pairs.py \
    --input /path/to/element_scored_pool.jsonl \
    --output outputs/element_rm_pairs_532.jsonl \
    --seed 42
```

The seed controls query shuffling for reproducible tier assignment. The constructor selects one pair per query and applies balancing constraints to control length ratio, on-policy share, and model-family concentration.

### build_lexchain_rm_pairs.py

Select 4000 rows using seed 42 and construct LexChain pairs:

```bash
python data_construction/build_lexchain_rm_pairs.py \
    --input /path/to/lexchain_scored_pool.jsonl \
    --output outputs/lexchain_rm_pairs_532.jsonl \
    --num-pairs 4000 --seed 42
```

For a smaller pool, request a smaller multiple of 10:

```bash
python data_construction/build_lexchain_rm_pairs.py \
    --input /path/to/lexchain_scored_pool.jsonl \
    --output outputs/lexchain_rm_pairs_100.jsonl \
    --num-pairs 100 --seed 42
```

## Notes

- Candidate-score ties are ordered by candidate index. CLASE tier-selection ties follow input order; LexChain tier-selection ties follow `row_id`.
- CLASE rejects selected equal-score pairs unless `--keep-ties` is supplied. LexChain retains them to preserve its original construction behavior and exact quotas.
- The CLASE method uses only the supplied scores; it does not reproduce the historical Style-v3-assisted C-tier selection.
- Constructed files are not automatically divided into training, validation, and test sets. Prepare those splits separately before RM training.
- Input and output paths must differ. The output file is written at the requested location, including overwriting an existing file there.
- Help commands do not load data or call models:

```bash
python data_construction/build_clase_rm_pairs.py --help
python data_construction/build_lexchain_rm_pairs.py --help
```
