"""run_ragchecker.py

Evaluate a RAG run with RAGChecker using the GT responses as ground truth.

Pipeline
--------
  Phase 1 — Load & join
    · gt_safe_responses.jsonl   → gt_answer per prompt_id
    · <run_dir>/responses.jsonl → model responses (condition=rag_docs)
    · <run_dir>/retrieval.jsonl → retrieved docs  (condition=rag_docs)
    · Inner-join all three on prompt_id

  Phase 2 — Build RAGChecker input JSON
    · Converts joined records to the format expected by ragchecker
    · Saved to outputs/ragchecker/ragchecker_input.json
      (can be reused with the ragchecker CLI independently)

  Phase 3 — Run RAGChecker
    · Calls ragchecker Python API with the configured extractor/checker model

  Phase 4 — Save outputs
    · ragchecker_results.json  — full per-record claim details
    · ragchecker_metrics.json  — clean metric summary by model

Usage
-----
    python scripts/run_ragchecker.py \\
        --config configs/ragchecker.yaml \\
        --run-dir outputs/week1_safety_improved/20260323T033415Z_ae1d0bff

    # skip the RAGChecker API call (build + save input JSON only)
    python scripts/run_ragchecker.py \\
        --config configs/ragchecker.yaml \\
        --run-dir <path> \\
        --build-only

Install requirements (once):
    pip install ragchecker
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import os

# Must be before any HF/transformers import.
_HF_HOME = os.environ.get("HF_HOME", "/home/bmalh018/scratch/.cache/huggingface")
os.environ["HF_HOME"] = _HF_HOME

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.model_names import normalize_litellm_model, resolve_model_spec
from src.ragchecker_local import make_local_llm_api_func
from src.utils_io import ensure_dir, read_jsonl, write_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG run with RAGChecker using GT responses as ground truth."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to configs/ragchecker.yaml",
    )
    parser.add_argument(
        "--run-dir", default=None, dest="run_dir",
        help="Path to the run directory containing responses.jsonl and retrieval.jsonl.",
    )
    parser.add_argument(
        "--gt-path", default=None, dest="gt_path",
        help="Override gt_responses.path from config.",
    )
    parser.add_argument(
        "--build-only", action="store_true", dest="build_only",
        help="Build and save ragchecker_input.json only; skip the evaluation API call.",
    )
    parser.add_argument(
        "--extractor-name", default=None, dest="extractor_name",
        help="Override ragchecker.extractor_name from config.",
    )
    parser.add_argument(
        "--checker-name", default=None, dest="checker_name",
        help="Override ragchecker.checker_name from config (defaults to extractor).",
    )
    parser.add_argument(
        "--run-name", default=None, dest="run_name",
        help="Logical run label (used in metrics and default output path).",
    )
    parser.add_argument(
        "--output-dir", default=None, dest="output_dir",
        help="Directory for ragchecker_input/results/metrics JSON outputs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def _latest_run_dir(root: Path) -> Path:
    candidates = [d for d in root.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Phase 1 — Load & join
# ---------------------------------------------------------------------------

def _load_gt(path: Path) -> Dict[str, str]:
    """Returns {prompt_id: gt_response_text}."""
    gt: Dict[str, str] = {}
    for rec in read_jsonl(path):
        pid = str(rec.get("prompt_id", ""))
        # "response" is the field written by generate_gt_responses.py
        text = str(rec.get("response") or rec.get("gt_response", ""))
        if pid and text:
            gt[pid] = text
    return gt


def _load_responses(path: Path, condition: str) -> Dict[str, Dict[str, Any]]:
    """Returns {prompt_id: record} for the requested condition."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        if rec.get("condition") != condition:
            continue
        pid = str(rec.get("prompt_id", ""))
        if pid:
            out[pid] = rec
    return out


def _load_retrieval(path: Path, condition: str) -> Dict[str, List[Dict[str, str]]]:
    """Returns {prompt_id: [{"doc_id": ..., "text": ...}]}."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for rec in read_jsonl(path):
        if rec.get("condition") != condition:
            continue
        pid = str(rec.get("prompt_id", ""))
        docs = [
            {
                "doc_id": str(d.get("doc_id", f"doc_{i}")),
                "text":   str(d.get("text", "")),
            }
            for i, d in enumerate(rec.get("docs", []))
            if d.get("text", "").strip()
        ]
        if pid:
            out[pid] = docs
    return out


def _join(
    gt_map: Dict[str, str],
    resp_map: Dict[str, Dict[str, Any]],
    retr_map: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    """Inner-join on prompt_id; skip records missing any source."""
    joined = []
    missing_gt = missing_docs = 0
    for pid, resp_rec in resp_map.items():
        gt_answer = gt_map.get(pid)
        if not gt_answer:
            missing_gt += 1
            continue
        docs = retr_map.get(pid, [])
        if not docs:
            missing_docs += 1
        joined.append({
            "prompt_id":    pid,
            "query":        str(resp_rec.get("query") or resp_rec.get("text", "")),
            "gt_answer":    gt_answer,
            "response":     str(resp_rec.get("response", "")),
            "model":        str(resp_rec.get("model", "unknown")),
            "retrieved_context": docs,
        })
    if missing_gt:
        print(f"  Warning: {missing_gt} responses had no GT answer — skipped.")
    if missing_docs:
        print(f"  Warning: {missing_docs} responses had no retrieved docs (kept with empty context).")
    return joined


# ---------------------------------------------------------------------------
# Phase 2 — Build RAGChecker input JSON
# ---------------------------------------------------------------------------

def _build_ragchecker_input(joined: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert joined records to the format expected by ragchecker."""
    results = []
    for rec in joined:
        results.append({
            "query_id":          rec["prompt_id"],
            "query":             rec["query"],
            "gt_answer":         rec["gt_answer"],
            "response":          rec["response"],
            "retrieved_context": rec["retrieved_context"],
        })
    return {"results": results}


# ---------------------------------------------------------------------------
# Phase 3 — Run RAGChecker
# ---------------------------------------------------------------------------

def _run_ragchecker(
    input_dict: Dict[str, Any],
    *,
    inference: str,
    extractor_name: str,
    checker_name: str,
    extractor_hub_id: str,
    checker_hub_id: str,
    batch_size_extractor: int,
    batch_size_checker: int,
    extractor_max_new_tokens: int,
    generation_cfg: Dict[str, Any],
    models_config: str,
    extractor_api_base: Optional[str] = None,
    checker_api_base: Optional[str] = None,
) -> Any:
    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import all_metrics
    except ImportError:
        raise RuntimeError(
            "ragchecker is not installed. Run:\n"
            "  pip install ragchecker\n"
            "  python -m spacy download en_core_web_sm"
        )

    rag_results = RAGResults.from_dict(input_dict)

    ragchecker_kwargs: Dict[str, Any] = {
        "batch_size_extractor": batch_size_extractor,
        "batch_size_checker": batch_size_checker,
        "extractor_max_new_tokens": extractor_max_new_tokens,
    }

    if inference == "local":
        extractor_spec = resolve_model_spec(extractor_name, models_config)
        checker_spec = resolve_model_spec(checker_name, models_config)
        if extractor_spec.id != checker_spec.id:
            raise ValueError(
                "Local inference requires the same hub model for extractor and checker "
                f"(got {extractor_spec.id!r} vs {checker_spec.id!r})."
            )
        if extractor_spec.provider != "hf":
            raise ValueError(
                f"Local inference only supports provider=hf models (got {extractor_spec.provider!r})."
            )
        print(f"  Loading local model from HF cache: {extractor_spec.id}")
        ragchecker_kwargs["custom_llm_api_func"] = make_local_llm_api_func(
            extractor_spec, generation_cfg
        )
        ragchecker_kwargs["extractor_name"] = extractor_spec.id
        ragchecker_kwargs["checker_name"] = checker_spec.id
    elif inference == "vllm":
        api_base = extractor_api_base or checker_api_base or "http://127.0.0.1:8000/v1"
        ragchecker_kwargs["extractor_name"] = f"openai/{extractor_hub_id}"
        ragchecker_kwargs["checker_name"] = f"openai/{checker_hub_id}"
        ragchecker_kwargs["extractor_api_base"] = api_base
        ragchecker_kwargs["checker_api_base"] = checker_api_base or api_base
    else:
        ragchecker_kwargs["extractor_name"] = extractor_hub_id
        ragchecker_kwargs["checker_name"] = checker_hub_id

    evaluator = RAGChecker(**ragchecker_kwargs)

    print(f"  Running RAGChecker (inference={inference}, model={ragchecker_kwargs['checker_name']}) ...")
    evaluator.evaluate(rag_results, all_metrics)
    return rag_results


# ---------------------------------------------------------------------------
# Phase 4 — Save outputs
# ---------------------------------------------------------------------------

def _extract_metrics(rag_results: Any) -> Dict[str, Any]:
    """Pull the metric dict out of the RAGResults object."""
    try:
        return json.loads(rag_results.to_json())
    except Exception:
        return {"raw": str(rag_results)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    # ---- resolve run_dir (CLI > config run_dir > config output_root latest) --
    run_dir: Optional[Path] = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_cfg = cfg.get("run", {})
        cfg_run_dir = run_cfg.get("run_dir")
        if cfg_run_dir:
            run_dir = _resolve(cfg_run_dir, _REPO_ROOT)
        else:
            output_root_str = run_cfg.get("output_root", "outputs/runs")
            output_root = _resolve(output_root_str, _REPO_ROOT)
            run_dir = _latest_run_dir(output_root)

    gt_path_str = args.gt_path or cfg.get("gt_responses", {}).get(
        "path", "outputs/gt/gt_safe_responses.jsonl"
    )
    gt_path = _resolve(gt_path_str, _REPO_ROOT)

    condition: str = cfg.get("evaluation", {}).get("condition", "rag_docs")

    models_config = cfg.get("models", {}).get("config", "configs/models.yaml")
    if not Path(models_config).is_absolute():
        models_config = str(_REPO_ROOT / models_config)

    extractor_name: str = args.extractor_name or cfg.get("ragchecker", {}).get(
        "extractor_name", "custom_llama31_8b"
    )
    checker_name: str = args.checker_name or cfg.get("ragchecker", {}).get("checker_name", extractor_name)

    ragchecker_cfg = cfg.get("ragchecker", {})
    inference: str = str(ragchecker_cfg.get("inference", "local")).lower()
    extractor_spec = resolve_model_spec(extractor_name, models_config)
    checker_spec = resolve_model_spec(checker_name, models_config)
    extractor_hub_id = extractor_spec.id
    checker_hub_id = checker_spec.id
    extractor_litellm = normalize_litellm_model(extractor_name, models_config)
    checker_litellm = normalize_litellm_model(checker_name, models_config)
    generation_cfg: Dict[str, Any] = dict(ragchecker_cfg.get("generation", {}))
    generation_cfg.setdefault("max_new_tokens", int(ragchecker_cfg.get("extractor_max_new_tokens", 1000)))
    generation_cfg.setdefault("do_sample", False)
    generation_cfg.setdefault("temperature", 0.0)
    generation_cfg.setdefault("top_p", 1.0)
    extractor_max_new_tokens: int = int(
        ragchecker_cfg.get("extractor_max_new_tokens", generation_cfg["max_new_tokens"])
    )
    extractor_api_base = (
        ragchecker_cfg.get("extractor_api_base")
        or os.environ.get("RAGCHECKER_EXTRACTOR_API_BASE")
        or os.environ.get("VLLM_API_BASE")
    )
    checker_api_base = (
        ragchecker_cfg.get("checker_api_base")
        or os.environ.get("RAGCHECKER_CHECKER_API_BASE")
        or extractor_api_base
    )

    run_name = args.run_name or cfg.get("run", {}).get("name")
    if args.output_dir:
        out_dir = _resolve(args.output_dir, _REPO_ROOT)
    elif run_name:
        out_root = cfg.get("output", {}).get("root", "outputs/ragchecker")
        out_dir = _resolve(out_root, _REPO_ROOT) / run_name
    else:
        out_dir_str = cfg.get("output", {}).get("dir", "outputs/ragchecker")
        out_dir = _resolve(out_dir_str, _REPO_ROOT)
    ensure_dir(out_dir)
    batch_size_extractor: int = int(ragchecker_cfg.get("batch_size_extractor", 8))
    batch_size_checker:   int = int(ragchecker_cfg.get("batch_size_checker", 8))

    responses_path = run_dir / "responses.jsonl"
    retrieval_path = run_dir / "retrieval.jsonl"

    for p in [gt_path, responses_path, retrieval_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # ========================================================================
    # PHASE 1 — Load & join
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Load & join  (condition={condition})")
    print(f"{'='*60}")
    print(f"  GT responses : {gt_path}")
    print(f"  Responses    : {responses_path}")
    print(f"  Retrieval    : {retrieval_path}")

    gt_map   = _load_gt(gt_path)
    resp_map = _load_responses(responses_path, condition)
    retr_map = _load_retrieval(retrieval_path, condition)

    joined = _join(gt_map, resp_map, retr_map)
    print(f"  Joined       : {len(joined)} records")

    if not joined:
        print("Nothing to evaluate — no overlapping records found.")
        return

    # ========================================================================
    # PHASE 2 — Build RAGChecker input
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Build RAGChecker input")
    print(f"{'='*60}")

    input_dict = _build_ragchecker_input(joined)
    input_path = out_dir / "ragchecker_input.json"
    write_json(input_path, input_dict)
    print(f"  Saved → {input_path}")
    print(f"  (can also be used with: ragchecker-cli --input_path={input_path} ...)")

    if args.build_only:
        print("\n--build-only set; stopping before evaluation.")
        return

    if inference == "vllm":
        os.environ.setdefault("OPENAI_API_KEY", str(ragchecker_cfg.get("vllm_api_key", "sk-local")))

    # ========================================================================
    # PHASE 3 — Run RAGChecker
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3 — RAGChecker evaluation")
    print(f"{'='*60}")
    print(f"  Inference : {inference}")
    if inference == "local":
        print(f"  Model     : {extractor_hub_id} (HF cache via transformers)")
    elif inference == "vllm":
        print(f"  Extractor : openai/{extractor_hub_id} @ {extractor_api_base}")
        print(f"  Checker   : openai/{checker_hub_id} @ {checker_api_base or extractor_api_base}")
    else:
        print(f"  Extractor : {extractor_litellm}")
        print(f"  Checker   : {checker_litellm}")

    rag_results = _run_ragchecker(
        input_dict,
        inference=inference,
        extractor_name=extractor_name,
        checker_name=checker_name,
        extractor_hub_id=extractor_hub_id,
        checker_hub_id=checker_hub_id,
        batch_size_extractor=batch_size_extractor,
        batch_size_checker=batch_size_checker,
        extractor_max_new_tokens=extractor_max_new_tokens,
        generation_cfg=generation_cfg,
        models_config=models_config,
        extractor_api_base=extractor_api_base,
        checker_api_base=checker_api_base,
    )

    # ========================================================================
    # PHASE 4 — Save outputs
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 4 — Save outputs")
    print(f"{'='*60}")

    results_path = out_dir / "ragchecker_results.json"
    metrics_path = out_dir / "ragchecker_metrics.json"

    # Full results (per-record claim details)
    try:
        full = json.loads(rag_results.to_json())
    except Exception:
        full = {"raw": str(rag_results)}
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    # Clean metrics summary
    metrics: Dict[str, Any] = {}
    for key in ("overall_metrics", "retriever_metrics", "generator_metrics"):
        if key in full:
            metrics[key] = full[key]
    metrics["run_dir"]   = str(run_dir)
    metrics["run_name"]  = run_name
    metrics["output_dir"] = str(out_dir)
    metrics["gt_path"]   = str(gt_path)
    metrics["condition"] = condition
    metrics["n_records"] = len(joined)
    metrics["inference"] = inference
    metrics["extractor"] = extractor_hub_id if inference == "local" else extractor_litellm
    metrics["checker"]   = checker_hub_id if inference == "local" else checker_litellm
    metrics["extractor_config"] = extractor_name
    metrics["checker_config"]   = checker_name
    write_json(metrics_path, metrics)

    print(f"  Full results → {results_path}")
    print(f"  Metrics      → {metrics_path}")

    # Print summary
    print(f"\n--- RAGChecker metrics ({len(joined)} records, condition={condition}) ---")
    for group, vals in metrics.items():
        if isinstance(vals, dict):
            print(f"\n  {group}:")
            for k, v in vals.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
