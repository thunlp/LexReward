# Rubric Scoring

## Overview

Score Chinese legal text with Style v7.2, deterministic chain rules, or an LLM judge. All commands below run from the repository root, the directory containing `requirements.txt`. Replace `/path/to/...` with your own files.

## Files

| Python file | Purpose |
| --- | --- |
| `style_scorer.py` | Compute five style dimensions and their equal-weight aggregate: D1 legal terminology, D2 subjectivity, D3 conjunction rate, D4 sentence structure, and D5 collocations. |
| `element_scorer.py` | Reference-free LLM judge (DeepSeek-V4-Flash) scoring four Element dimensions on a 0-10 scale: subjects (legal parties), facts (case elements), statutes (legal grounds), and disposition (conclusions). |
| `chain_scorer_rule.py` | Parse numbered step markers and score chain order, completeness, valid step labels, and non-redundancy with deterministic rules. |
| `chain_scorer_llm.py` | Use a chat API to identify reasoning steps, request four subjective scores on a 0-10 scale, validate the response, and aggregate the scores. |

Supporting resources are stored in `lexicons/` and `reference_data/`. They are not executable Python files.

## Requirements

Use Python 3.11 and install the scoring dependencies:

```bash
pip install -r requirements.txt
```

Style scoring requires the legal-term JSON array, the HowNet lexicon files, and the reference statistics shipped beside the script. Resource locations are resolved relative to the script. Set `LEXREWARD_LEGAL_TERMS_FILE` or `LEXREWARD_HOWNET_DIR` to override the lexicon locations; preserve the original HowNet filenames and GB2312 encoding.

Element scoring requires the `MODELBEST_API_KEY` environment variable for DeepSeek-V4-Flash API access. Install `aiohttp` for async HTTP requests.

Rule-based chain scoring does not need a GPU or an API. LLM scoring needs credentials, a base URL, and a model served by an OpenAI-compatible chat API.

## Inputs and Outputs

| Script | Inputs | Outputs |
| --- | --- | --- |
| `style_scorer.py` | A text argument, a UTF-8 file, interactive input, or JSONL records containing `gold` and `generated`. | A readable report or single-text JSON with `overall_reward` and D1-D5 results. JSONL mode prints gold/generated comparison statistics. |
| `chain_scorer_rule.py` | The same input modes as Style scoring. Text must use the numbered full-width step markers and task-specific labels defined by `STEP_PATTERN` and `StepConfig` in the source. | A readable report or single-text JSON with `overall_reward` and C1-C4 results. JSONL mode prints comparison statistics. |
| `chain_scorer_llm.py` | A text argument, a UTF-8 file, or JSONL records with a nonempty `text`, `generated`, or `output` field. | Single-text JSON contains `overall_score`, dimension scores, and identified steps. Batch mode writes `<input_stem>_subjective_scored.jsonl` next to the input. |

For local JSONL evaluation, each line has this structure:

```json
{"gold": "Reference legal text", "generated": "Candidate legal text"}
```

For LLM batch scoring, use this structure:

```json
{"text": "Legal reasoning to evaluate"}
```

These examples illustrate the format; use actual Chinese legal text for the task. Local JSONL evaluation prints statistics rather than saving per-record scores. LLM batch output adds `llm_subjective_scores` to each input record.

## Usage

### style_scorer.py

Score one document with the five Style dimensions and print JSON:

```bash
python rubric_scorer/style_scorer.py \
    --file /path/to/legal_text.txt --json
```

Compare references and generated answers from a JSONL file:

```bash
python rubric_scorer/style_scorer.py \
    --jsonl /path/to/gold_generated_pairs.jsonl
```

### element_scorer.py

Score a single response with the four Element dimensions:

```bash
export MODELBEST_API_KEY='YOUR_API_KEY'

python rubric_scorer/element_scorer.py \
    --file /path/to/response.txt \
    --question "案情描述和法律问题" \
    --json
```

Batch score a JSONL file of question-answer pairs:

```bash
python rubric_scorer/element_scorer.py \
    --jsonl /path/to/candidates.jsonl \
    --output /path/to/scored_output.jsonl \
    --concurrency 6
```

The JSONL input requires `question` or `instruction` fields for the question text, and `answer` or `response` for the candidate text. The scorer is reference-free and should not be used when comparing gold answers against model outputs in the same batch.

### chain_scorer_rule.py

Score a document containing the expected numbered step markers:

```bash
python rubric_scorer/chain_scorer_rule.py \
    --file /path/to/marked_reasoning.txt --json
```

Select dimensions, or compare gold/generated pairs in batch:

```bash
python rubric_scorer/chain_scorer_rule.py \
    --file /path/to/marked_reasoning.txt --dims C1,C2 --json

python rubric_scorer/chain_scorer_rule.py \
    --jsonl /path/to/marked_gold_generated_pairs.jsonl
```

C3 checks label validity; it does not establish the factual or legal correctness of the reasoning content.

### chain_scorer_llm.py

Configure a judge service and score one document:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://example.invalid/v1'

python rubric_scorer/chain_scorer_llm.py \
    --model YOUR_JUDGE_MODEL \
    --file /path/to/legal_reasoning.txt --json
```

The key, endpoint, and model above are placeholders. Replace them locally before running. Batch scoring uses the same configuration:

```bash
python rubric_scorer/chain_scorer_llm.py \
    --model YOUR_JUDGE_MODEL \
    --jsonl /path/to/legal_reasoning.jsonl \
    --concurrency 4 --strategy weighted_mean
```

The alternative `--strategy min` returns the lowest dimension score. `--weights` accepts a JSON object with C1-C4 weights for the weighted mean.

## Notes

- API settings come from explicit `--api-key` / `--base-url` arguments or environment variables. Supported key variables are `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, and `API_KEY`; endpoint variables are `OPENAI_BASE_URL`, `BASE_URL`, and `API_URL`. `--model` overrides the model selected by `JUDGE_MODEL`.
- No local `config.py` or `.env` file is loaded automatically. Export configuration values in your shell.
- Comments, docstrings, CLI help, and this README are in English. Chinese task prompts, legal terms, scoring labels, and diagnostics remain part of the scoring implementation.
- D1-D5 use consecutive numbering. Historical archived data may use the former D1/D2/D4/D5/D6 labels.
- Help commands do not call an API:

```bash
python rubric_scorer/style_scorer.py --help
python rubric_scorer/chain_scorer_rule.py --help
python rubric_scorer/chain_scorer_llm.py --help
```
