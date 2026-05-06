# Data Sources and Provenance

This project tracks source provenance for each dataset artifact used in capability and safety runs.

## Canonical Inputs

- Harmful prompts (paper provenance target): `data/prompts/rrb_harmbench_full.csv`
- Harmful two-week subset (stratified): `data/prompts/rrb_harmbench_subset_1500.csv`
- NQ queries: `data/qa/nq_queries.csv`
- NQ answers: `data/qa/nq_answers.jsonl`
- NQ docs manifest: `data/qa/nq_docs_manifest.json`
- NQ evaluable subset: `data/qa/nq_eval_445.csv`
- NQ oracle contexts (fixed top-5): `data/qa/nq_oracle_context_top5.jsonl`

## Provenance Manifest

- Manifest path: `data/raw/SOURCE_MANIFEST.json`
- Updated by:
  - `scripts/fetch_rrb_harmbench.py`
  - `scripts/fetch_nq_irds.py`
  - `scripts/build_nq_oracle_context.py`

Each source entry records:

- source identifier (CSV source or `ir_datasets` ID),
- timestamp,
- file hash (SHA256 where applicable),
- license reminder note.

Each produced file entry records:

- output file path,
- output SHA256 hash,
- generation timestamp.

## Usage

1. Build harmful prompt canonical files from source CSVs:

```bash
python scripts/fetch_rrb_harmbench.py \
  --csv-source "RRB|haizelab_rrb|data/raw/rrb.csv" \
  --csv-source "HarmBench|harmbench|data/raw/harmbench.csv"
```

2. Export Natural Questions assets from `ir_datasets`:

```bash
python scripts/fetch_nq_irds.py --irds-id "irds:natural-questions/train"
```

3. Validate config before running:

```bash
python scripts/validate_run_config.py --config configs/week1_capability.yaml
python scripts/validate_run_config.py --config configs/week1_safety.yaml
```

## Notes

- Keep source files immutable once hashes are logged.
- If you regenerate any canonical file, keep prior manifest entries and append new ones.
- Verify dataset licensing before publication or redistribution.
