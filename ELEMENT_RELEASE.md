# Element Dimension Release Checklist

## Code Components

### Rubric Scorer
- **Path**: `rubric_scorer/element_scorer.py`
- **Description**: DeepSeek-V4-Flash LLM judge scoring 4 dimensions (subjects, facts, statutes, disposition)
- **Requirements**: `MODELBEST_API_KEY` environment variable, `aiohttp`
- **Aligned with**: Paper Table 1 Element evaluation methodology

### Preference Pair Constructor
- **Path**: `data_construction/build_element_rm_pairs.py`
- **Description**: Gap-based 5:3:2 tiered pair construction with balancing constraints
- **Input**: Scored five-candidate JEC-QA + LexChain pool
- **Output**: 2,244 preference pairs (from 2,736 source instances)
- **Aligned with**: Paper §4.2 preference construction, Table 1 experimental setup

### Documentation
- **Paths**:
  - `README.md` (main, updated with Element)
  - `rubric_scorer/README.md` (usage examples)
  - `data_construction/README.md` (input/output schemas)

## Training Data

### Preference Pairs (for HuggingFace dataset upload)
- **Source path**: `/slow_share2/ultrafast_share/daixin/LexReward/reward_modeling/data/element_tiered_inv_gen25_20260918.jsonl`
- **Statistics**:
  - Total pairs: 2,992
  - Pairs with tier labels: 2,244 (A: 449, B: 673, C: 1,122)
  - Pairs without tier: 748 (construction metadata rows)
- **Schema**: `{chosen: [...], rejected: [...], meta: {tier, chosen_model, rejected_model, chosen_score, rejected_score, ...}}`
- **Aligned with**: Paper "2,736 instances yielding 2,244 preference pairs"

## Reward Model Checkpoint

### LexRM-Element (for HuggingFace model upload)
- **Source path**: `/slow_share2/ultrafast_share/daixin/LexReward/reward_modeling/runs/qwen3_8b_element_rm_tiered_inv_gen25_20260918/final/`
- **Files**:
  - `model.safetensors` (15GB, full-parameter Qwen3-8B backbone)
  - `reward_head.pt` (18KB, external reward head)
  - `config.json`, `rm_config.json`
  - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- **Architecture**: Full-parameter BT reward model, bfloat16, max_length 6144
- **Training**: 748 global steps, lr 1e-6 (backbone) + 1e-5 (head), seed 42, 4 GPUs
- **Aligned with**: Paper Table 1 Element block results (77.72/57.80/44.70/59.20/92.20)

## Evaluation Data Reference

### LegalDelta In-Domain Test Set
- **Path**: `/home/dx/legaldelta_bo8_eval_20260924/api/dsv4_elemrubric_blind_tts_cross.uid.jsonl`
- **Size**: 3,000 instances
- **Format**: TTS pool with Element rubric scores (subjects/facts/statutes/disposition/overall)
- **Usage**: Produces Table 1 Element scores with `/home/dx/report_legaldelta_bycol_20260924.py`
- **Aligned with**: Paper "3,000 LegalDelta in-domain test instances"

## Verification

### Paper Alignment Checks
- [x] Element RM checkpoint produces Table 1 numbers (77.72, 57.80, 44.70, 59.20, 92.20, avg 66.32)
- [x] Training data: 2,244 pairs from 2,736 source instances (matches paper §4.2)
- [x] Five-model candidate pool: Qwen3-4B, Qwen3.5-4B/9B, Llama-3.1-8B-Instruct, gemma-4-12B-it
- [x] RM backbone: Qwen3-8B full-parameter tuning
- [x] Evaluation: LegalDelta in-domain 3,000 instances (SPP-F, CCP F1, SLP/CAS/CAC Acc)
- [x] Rubric scorer: DeepSeek-V4-Flash reference-free blind judge
- [x] 5:3:2 tiered structure with gap-based tier assignment

### Code Quality
- [x] Element scorer matches Style/Chain code patterns
- [x] Element constructor matches CLASE/LexChain style
- [x] Documentation updated across all READMEs
- [x] CLI help messages and docstrings complete
- [x] No hardcoded API keys (environment variables only)

## HuggingFace Upload Structure

```
LexReward-Paper/
├── README.md (overview, citation, quick start)
├── rubric_scorer/
│   ├── element_scorer.py
│   ├── style_scorer.py
│   ├── chain_scorer_rule.py
│   ├── chain_scorer_llm.py
│   ├── README.md
│   ├── lexicons/
│   └── reference_data/
├── data_construction/
│   ├── build_element_rm_pairs.py
│   ├── build_clase_rm_pairs.py
│   ├── build_lexchain_rm_pairs.py
│   └── README.md
├── rm_training/
│   ├── train_rm.py
│   ├── launch_rm.sh
│   ├── ds_z3.json
│   └── README.md
├── requirements.txt
└── requirements-training.txt

LexReward-Element-Data/ (HF dataset)
└── element_rm_pairs_532.jsonl (2,244 pairs with tier labels)

LexRM-Element/ (HF model)
├── model.safetensors
├── reward_head.pt
├── config.json
├── rm_config.json
└── tokenizer files
```

## Next Steps

1. Copy filtered 2,244-pair training data to `/home/dx/LexReward-main/` for upload staging
2. Verify all file permissions and encodings
3. Create HuggingFace dataset card and model card
4. Upload to https://huggingface.co/papers/2609.39071

## Notes

- Training data file contains 2,992 total pairs: 2,244 labeled (A/B/C) + 748 unlabeled
- Only the 2,244 labeled pairs should be uploaded as the public dataset
- The 748 unlabeled pairs are construction metadata not used in RM training
- Element rubric is reference-free: gold answers must NOT be included in TTS pool evaluation
