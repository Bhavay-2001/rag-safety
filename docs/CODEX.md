# Codex Working Log

## Current Baseline Runs
- Safety baseline: `outputs/week1_safety/20260219T214819Z_a9907427`
- Safety improved baseline: `outputs/week1_safety_improved/20260302T002950Z_d2bb73c8`
- Capability baseline (legacy NQ): `outputs/qa_smoke_nq_rag/20260219T194624Z_9040ad1e`

## Active Branch Status
- Branch: `main`
- Primary capability dataset: **MIRAGE v2**
- NQ status: optional secondary / paper-mirroring only
- Safety judging: LG3 primary, WildGuard edge-case second pass only

## Decisions (Locked)
1. MIRAGE is primary capability benchmark.
2. Capability conditions: `non_rag`, `rag_docs` (oracle), `rag_mixed_docs`; `rag_llm` optional.
3. No full dual-judge safety run yet; second judge only on confounding slices.
4. Scale only after MIRAGE smoke + metric sanity checks pass.

## Immediate Queue
### P0: MIRAGE v2 capability smoke
1. Pull MIRAGE (validation split, pilot 500):
   - `python -X utf8 scripts/fetch_mirage_v2.py --split train --max-rows 500 --output-dir data/qa`
2. Validate and preflight:
   - `python scripts/validate_run_config.py --config configs/qa_smoke_mirage_v2.yaml`
   - `python scripts/qa_preflight_v2.py --config configs/qa_smoke_mirage_v2.yaml`
3. Run smoke:
   - `python -X utf8 scripts/run_qa_eval_v2.py --config configs/qa_smoke_mirage_v2.yaml`

### P1: MIRAGE v2 full pilot
- `python scripts/validate_run_config.py --config configs/week1_capability_mirage_v2.yaml`
- `python scripts/qa_preflight_v2.py --config configs/week1_capability_mirage_v2.yaml`
- `python -X utf8 scripts/run_qa_eval_v2.py --config configs/week1_capability_mirage_v2.yaml`
- Target size: `final_eval_size=1000`

### P2: Safety edge second-pass quality checks
- Confirm `judge_second_pass.jsonl` and `judge_disagreement_stats.json`
- Confirm `metrics.json` has:
  - `unsafe_rate_primary`
  - `unsafe_rate_consensus_any`
  - `unsafe_rate_consensus_both`
  - `disagreement_rate`

## WildGuard Status (Current)
- Config wiring exists (`judge.edge_second_pass_enabled`, `judge.edge_second_pass_model`, `judge.edge_second_pass_scope`).
- Second-pass runner exists (`scripts/run_judge_second_pass.py`) and runs only edge cases.
- **Open implementation gap:** WildGuard currently runs through the generic safety judge path; add a dedicated WildGuard adapter/parser if disagreement quality is unstable.

## Known Blockers
1. MIRAGE pull requires `datasets` package + HF cache on scratch.
2. First-token safety score calibration is still flagged `needs_debug`.
3. WildGuard prompt/parser specialization may be required after first disagreement audit.

## Test Plan
### Capability (MIRAGE)
- Adapter smoke (50 rows): schema validity, ID overlap, non-empty oracle/mixed contexts.
- Pilot smoke (<=500): all 3 conditions emit outputs and `qa_metrics.json` comparisons.
- Full run (1000): same checks at scale.
- Sanity checks:
  - `rag_docs` > `non_rag` on oracle slice (EM/F1 directional check).
  - `rag_mixed_docs` <= `rag_docs` (degradation expected).
  - `context_behavior` block present with:
    - `noise_vulnerability.mean_f1_drop_oracle_to_mixed`
    - `context_acceptability.mean_f1_gain_non_to_oracle`
    - `context_insensitivity.rate_non_improving_non_to_oracle`
    - `context_misinterpretation.rate_non_correct_oracle_wrong`

### Safety (Edge Second-Pass)
- Artifacts exist: `judge_second_pass.jsonl`, `judge_disagreement_stats.json`.
- Metrics include consensus/disagreement keys.
- Actionability includes second-pass-triggered unsafe rows.
