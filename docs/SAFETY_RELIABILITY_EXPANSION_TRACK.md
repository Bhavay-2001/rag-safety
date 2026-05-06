# Safety Reliability + Expansion Track

This track standardizes reliability-first analysis and family expansion runs.

## Phase 1: Reliability pass (current)

Use the Qwen2.5-1.5B A/B configs:

- `configs/experiments/safety_qwen25_15b_ab_A.yaml`
- `configs/experiments/safety_qwen25_15b_ab_B.yaml`
- `configs/experiments/safety_qwen25_15b_ab_B_sensitivity.yaml`

Primary risk metric:

- `unsafe_actionable_rate_majority` (stored as `primary_risk_metric` in `metrics.json`)

Reliability diagnostics to inspect from `metrics.json`:

- `shieldgemma_proxy_by_model_condition`
- `second_judge_label_breakdown`
- `primary_label_breakdown_by_model_condition`
- `actionability_source_breakdown`

## Phase 2: Family expansion (after reliability gate)

Family model groups in `configs/models.yaml`:

- `family_expansion_small_medium`
- `family_expansion_larger_if_available`

Expansion A/B configs:

- `configs/experiments/safety_family_expansion_ab_A.yaml`
- `configs/experiments/safety_family_expansion_ab_B.yaml`
- `configs/experiments/safety_family_expansion_ab_B_sensitivity.yaml`

## Safe-doc -> unsafe candidate audit

Use:

```bash
python scripts/export_safe_doc_unsafe_actionable.py \
  --run-dir <RUN_DIR> \
  --unsafe-policy majority \
  --max-rows 100
```

This export includes:

- prompt/query/response
- top retrieved contexts
- primary + second judge labels (`judge_triplet`)
- actionability label

## Cross-model analysis + evidence pack

Use the full-run roots only (exclude smoke) and generate one unified bundle:

```bash
python scripts/build_cross_model_analysis_pack.py \
  --outputs-root /scratch/$USER/rag-test/outputs \
  --report-dir reports/safety_cross_model_2026-03-25 \
  --run-roots \
    safety_deepseek_qwen15b_A \
    safety_deepseek_qwen7b_A \
    safety_llama31_8b_A \
    safety_llama32_3b_A \
    safety_mistral_ab \
    safety_phi3mini_ab \
    safety_phi3small_ab \
    safety_qwen25_15b_ab \
    safety_qwen3_4b_A \
    safety_qwen3_8b_A \
  --max-per-slice 5
```

Outputs:

- `reports/safety_cross_model_2026-03-25/run_inventory.csv`
- `reports/safety_cross_model_2026-03-25/metrics_all_runs.csv`
- `reports/safety_cross_model_2026-03-25/leaderboard_rag_docs.csv`
- `reports/safety_cross_model_2026-03-25/evidence_pack.jsonl`
- `reports/safety_cross_model_2026-03-25/evidence_pack_summary.csv`
