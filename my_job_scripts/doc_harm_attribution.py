#!/usr/bin/env python3
"""Worker for doc-harm attribution (called by run_doc_harm_attribution.slurm).

Same model re-generates; ShieldGemma-9b decides safe/unsafe.

Core ablations (per case, top-5 docs)
-------------------------------------
  all_docs     : rag_llm with docs in retrieval order
  non_rag      : query only
  single:i     : only doc i  (always rendered as Context 1 — position-controlled)
  loo:i        : all docs except i  (NOTE: remaining docs shift up in rank;
                 LOO does *not* control position bias)
  ctx1:i       : all 5 docs, but doc i moved to Context 1; others keep relative order
                 (position probe: content fixed, only rank of i changes)

Combination search (adaptive)
-----------------------------
If all_docs=unsafe, non_rag=safe, and no single:i is unsafe, then several docs
may be jointly required. We then test:
  combo:i+j       pairs
  combo:i+j+k     triples (only if no unsafe pair)
and flag *minimal* unsafe subsets (unsafe set with no unsafe proper subset).
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.judge_runner import SafetyJudge, parse_judge_label
from src.model_runner import ModelRunner, ModelSpec
from src.prompt_builder import load_prompt_templates, render_prompt
from src.utils_io import ensure_dir, write_json, write_jsonl

MODELS: Dict[str, str] = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "phi4": "microsoft/Phi-4-mini-instruct",
}

JUDGE_MODEL_ID = "google/shieldgemma-9b"
EXPECTED_TOP_K = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")

Ablation = Tuple[str, str, List[str], Dict[str, Any]]  # id, condition, docs, meta


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def strip_newlines(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text).replace("\r", "").replace("\n", " ")).strip()


def _combo_id(indices: Sequence[int]) -> str:
    return "combo:" + "+".join(str(i) for i in indices)


def build_core_ablations(docs: Sequence[str]) -> List[Ablation]:
    all_docs = list(docs)
    n = len(all_docs)
    rows: List[Ablation] = [
        ("all_docs", "rag_llm", all_docs, {"kept": list(range(n)), "dropped": [], "kind": "baseline"}),
        ("non_rag", "non_rag", [], {"kept": [], "dropped": list(range(n)), "kind": "query_only"}),
    ]
    # single:i — always Context 1 only (position controlled).
    for i in range(n):
        rows.append(
            (
                f"single:{i}",
                "rag_llm",
                [all_docs[i]],
                {"kept": [i], "dropped": [j for j in range(n) if j != i], "kind": "single_as_ctx1"},
            )
        )
    # loo:i — remaining docs shift ranks (not position-controlled).
    for i in range(n):
        kept = [j for j in range(n) if j != i]
        rows.append(
            (
                f"loo:{i}",
                "rag_llm",
                [all_docs[j] for j in kept],
                {"kept": kept, "dropped": [i], "kind": "leave_one_out"},
            )
        )
    # ctx1:i — all docs kept; doc i forced to Context 1.
    for i in range(n):
        order = [i] + [j for j in range(n) if j != i]
        rows.append(
            (
                f"ctx1:{i}",
                "rag_llm",
                [all_docs[j] for j in order],
                {"kept": order, "dropped": [], "kind": "reorder_doc_to_ctx1", "ctx1_doc": i},
            )
        )
    return rows


def build_combo_ablations(docs: Sequence[str], size: int) -> List[Ablation]:
    all_docs = list(docs)
    n = len(all_docs)
    rows: List[Ablation] = []
    for idxs in itertools.combinations(range(n), size):
        kept = list(idxs)
        rows.append(
            (
                _combo_id(kept),
                "rag_llm",
                [all_docs[i] for i in kept],
                {
                    "kept": kept,
                    "dropped": [j for j in range(n) if j not in kept],
                    "kind": f"combo_size_{size}",
                },
            )
        )
    return rows


def _resolve_target(raw_target: Any, model_key: str, case_id: str) -> str:
    """Accept a plain string or a {model_key: response} map."""
    if isinstance(raw_target, dict):
        if model_key not in raw_target:
            raise SystemExit(
                f"Case {case_id}: target_harmful_response dict missing key {model_key!r} "
                f"(have {sorted(raw_target)})"
            )
        return str(raw_target[model_key])
    return str(raw_target or "")


def load_cases(path: Path, model_key: str) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Cases file must be a JSON list: {path}")
    cases: List[Dict[str, Any]] = []
    for i, row in enumerate(raw):
        case_id = str(row.get("case_id") or f"case_{i+1:02d}")
        query = str(row.get("query", ""))
        docs = [str(d) for d in (row.get("docs") or [])]
        target = _resolve_target(row.get("target_harmful_response", ""), model_key, case_id)
        if "REPLACE_WITH_" in query or "REPLACE_WITH_" in target or any("REPLACE_WITH_" in d for d in docs):
            raise SystemExit(
                f"Case {case_id} still has REPLACE_WITH_ placeholders. "
                f"Edit {path} before submitting the job."
            )
        if len(docs) != EXPECTED_TOP_K:
            raise SystemExit(f"Case {case_id}: expected {EXPECTED_TOP_K} docs, got {len(docs)}")
        if not query.strip() or not target.strip():
            raise SystemExit(f"Case {case_id}: query and target_harmful_response must be non-empty")
        cases.append(
            {
                "case_id": case_id,
                "query": query,
                "docs": docs,
                "target_harmful_response": target,
            }
        )
    if not cases:
        raise SystemExit(f"No cases in {path}")
    return cases


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_rows(
    runner: ModelRunner,
    spec: ModelSpec,
    templates: Dict[str, Any],
    case: Dict[str, Any],
    ablations: Sequence[Ablation],
    model_key: str,
    model_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ablation_id, condition, doc_subset, meta in ablations:
        template_text = templates[condition]["template"]
        prompt = render_prompt(template_text, query=case["query"], sources=doc_subset)
        print(f"  [{case['case_id']}] {ablation_id} ({condition}, {len(doc_subset)} docs)")
        response = runner.generate(spec, prompt)
        rows.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "model": model_key,
                "model_id": model_id,
                "ablation": ablation_id,
                "condition": condition,
                "kind": meta.get("kind"),
                "n_docs": len(doc_subset),
                "kept_doc_indices": list(meta["kept"]),
                "dropped_doc_indices": list(meta["dropped"]),
                "ctx1_doc": meta.get("ctx1_doc"),
                "response": response,
                "jaccard_vs_target": token_jaccard(response, case["target_harmful_response"]),
                "target_harmful_response": case["target_harmful_response"],
            }
        )
    return rows


def _judge_rows(judge: SafetyJudge, gen_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in gen_rows:
        result = judge.judge(row["query"], row["response"], include_explanation=False, judge_target="response")
        raw = str(result.get("raw") or "")
        label = str(result.get("label") or parse_judge_label(raw) or "unknown")
        raw_no_nl = strip_newlines(raw)
        if label == "unknown" and raw_no_nl:
            first = raw_no_nl.split(" ", 1)[0].lower()
            if first in {"safe", "unsafe"}:
                label = first
        out = dict(row)
        out["judge_model_id"] = judge.model_id
        out["judge_label"] = label
        out["judge_raw"] = raw_no_nl
        out["judge_categories"] = result.get("categories") or []
        out["unsafe_prob_first_token_norm"] = result.get("unsafe_prob_first_token_norm")
        out_rows.append(out)
        print(
            f"  [{row['case_id']}] {row['ablation']}: "
            f"label={label} jaccard={row['jaccard_vs_target']:.3f}"
        )
    return out_rows


def _needs_combo_search(case_rows: Sequence[Dict[str, Any]]) -> bool:
    by_id = {r["ablation"]: r for r in case_rows}
    baseline = by_id.get("all_docs", {})
    non_rag = by_id.get("non_rag", {})
    if baseline.get("judge_label") != "unsafe":
        return False
    if non_rag.get("judge_label") == "unsafe":
        return False  # query alone enough; not a multi-doc interaction story
    singles_unsafe = [
        r for r in case_rows if str(r["ablation"]).startswith("single:") and r["judge_label"] == "unsafe"
    ]
    return len(singles_unsafe) == 0


def _minimal_unsafe_combos(case_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unsafe combo rows whose kept set has no unsafe proper subset among judged combos/singles."""
    unsafe_sets: List[Tuple[frozenset[int], Dict[str, Any]]] = []
    for r in case_rows:
        if r["judge_label"] != "unsafe":
            continue
        abl = str(r["ablation"])
        if abl.startswith("single:") or abl.startswith("combo:"):
            kept = frozenset(int(x) for x in r["kept_doc_indices"])
            if kept:
                unsafe_sets.append((kept, r))

    minimal: List[Dict[str, Any]] = []
    for kept, row in unsafe_sets:
        has_smaller = False
        for other, _ in unsafe_sets:
            if other < kept:  # proper subset
                has_smaller = True
                break
        if not has_smaller:
            minimal.append(
                {
                    "ablation": row["ablation"],
                    "doc_indices": sorted(kept),
                    "size": len(kept),
                    "judge_label": row["judge_label"],
                    "jaccard_vs_target": row["jaccard_vs_target"],
                }
            )
    minimal.sort(key=lambda x: (x["size"], x["doc_indices"]))
    return minimal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Doc-harm attribution worker.")
    p.add_argument("--cases", required=True, help="JSON list of cases (5 queries × top-5 docs).")
    p.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts-yaml", default=str(_REPO_ROOT / "configs" / "prompts.yaml"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--judge-model-id", default=JUDGE_MODEL_ID)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cases = load_cases(Path(args.cases), model_key=args.model)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    templates = load_prompt_templates(args.prompts_yaml)
    for needed in ("rag_llm", "non_rag"):
        if needed not in templates:
            raise SystemExit(f"Missing prompt template {needed!r} in {args.prompts_yaml}")

    model_id = MODELS[args.model]
    gen_cfg = {
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    write_json(
        out_dir / "run_meta.json",
        {
            "model": args.model,
            "model_id": model_id,
            "judge_model_id": args.judge_model_id,
            "n_cases": len(cases),
            "top_k": EXPECTED_TOP_K,
            "generation": gen_cfg,
            "cases_path": str(Path(args.cases).resolve()),
            "notes": {
                "single_i": "doc i alone as Context 1 (position controlled)",
                "loo_i": "drops doc i; remaining docs shift ranks (NOT position controlled)",
                "ctx1_i": "all docs kept; doc i moved to Context 1",
                "combo": "adaptive pairs/triples when no single doc is sufficient",
            },
        },
    )

    case_by_id = {c["case_id"]: c for c in cases}

    # ---- Phase 1: core generation ----
    runner = ModelRunner(gen_cfg)
    spec = ModelSpec(id=model_id, alias=args.model, provider="hf")
    gen_rows: List[Dict[str, Any]] = []
    print(f"=== GENERATE core model={args.model} ({model_id}) cases={len(cases)} ===")
    for case in cases:
        gen_rows.extend(
            _generate_rows(runner, spec, templates, case, build_core_ablations(case["docs"]), args.model, model_id)
        )
    del runner
    _free_cuda()

    # ---- Phase 2: judge core ----
    print(f"=== JUDGE core model={args.judge_model_id} ===")
    judge = SafetyJudge(
        args.judge_model_id,
        generation_cfg={"max_new_tokens": 64, "temperature": 0.0, "do_sample": False},
        judge_cfg={
            "score_mode": "first_token_prob",
            "strict_parse": False,
            "include_self_confidence": False,
            "first_token_debug": False,
            "taxonomy": "lg3_mlcommons14",
            "prompts_config": str(_REPO_ROOT / "configs" / "judge_prompts.yaml"),
        },
    )
    all_rows = _judge_rows(judge, gen_rows)

    # ---- Phase 3: adaptive combination search ----
    combo_plan: Dict[str, List[int]] = {}  # case_id -> sizes to run in order
    for case in cases:
        cid = case["case_id"]
        crow = [r for r in all_rows if r["case_id"] == cid]
        if _needs_combo_search(crow):
            combo_plan[cid] = [2]  # start with pairs
            print(f"=== COMBO search queued for {cid} (no single doc sufficient) ===")

    if combo_plan:
        del judge
        _free_cuda()
        runner = ModelRunner(gen_cfg)
        spec = ModelSpec(id=model_id, alias=args.model, provider="hf")
        combo_gen: List[Dict[str, Any]] = []

        # Pairs first.
        print(f"=== GENERATE combo pairs ===")
        for cid, sizes in list(combo_plan.items()):
            if 2 not in sizes:
                continue
            case = case_by_id[cid]
            combo_gen.extend(
                _generate_rows(
                    runner, spec, templates, case, build_combo_ablations(case["docs"], 2), args.model, model_id
                )
            )
        del runner
        _free_cuda()

        judge = SafetyJudge(
            args.judge_model_id,
            generation_cfg={"max_new_tokens": 64, "temperature": 0.0, "do_sample": False},
            judge_cfg={
                "score_mode": "first_token_prob",
                "strict_parse": False,
                "include_self_confidence": False,
                "first_token_debug": False,
                "taxonomy": "lg3_mlcommons14",
                "prompts_config": str(_REPO_ROOT / "configs" / "judge_prompts.yaml"),
            },
        )
        pair_judged = _judge_rows(judge, combo_gen)
        all_rows.extend(pair_judged)

        # Triples only for cases with no unsafe pair.
        need_triples: List[str] = []
        for cid in combo_plan:
            crow = [r for r in all_rows if r["case_id"] == cid]
            unsafe_pairs = [
                r
                for r in crow
                if str(r["ablation"]).startswith("combo:") and r["n_docs"] == 2 and r["judge_label"] == "unsafe"
            ]
            if not unsafe_pairs:
                need_triples.append(cid)
                print(f"=== COMBO triples queued for {cid} (no unsafe pair) ===")

        if need_triples:
            del judge
            _free_cuda()
            runner = ModelRunner(gen_cfg)
            spec = ModelSpec(id=model_id, alias=args.model, provider="hf")
            triple_gen: List[Dict[str, Any]] = []
            print(f"=== GENERATE combo triples ===")
            for cid in need_triples:
                case = case_by_id[cid]
                triple_gen.extend(
                    _generate_rows(
                        runner, spec, templates, case, build_combo_ablations(case["docs"], 3), args.model, model_id
                    )
                )
            del runner
            _free_cuda()
            judge = SafetyJudge(
                args.judge_model_id,
                generation_cfg={"max_new_tokens": 64, "temperature": 0.0, "do_sample": False},
                judge_cfg={
                    "score_mode": "first_token_prob",
                    "strict_parse": False,
                    "include_self_confidence": False,
                    "first_token_debug": False,
                    "taxonomy": "lg3_mlcommons14",
                    "prompts_config": str(_REPO_ROOT / "configs" / "judge_prompts.yaml"),
                },
            )
            all_rows.extend(_judge_rows(judge, triple_gen))

    write_jsonl(out_dir / "ablation_results.jsonl", all_rows)

    # ---- Summary ----
    summary: Dict[str, Any] = {"model": args.model, "model_id": model_id, "cases": {}}
    for case in cases:
        cid = case["case_id"]
        mrows = [r for r in all_rows if r["case_id"] == cid]
        by_id = {r["ablation"]: r for r in mrows}
        baseline = by_id.get("all_docs", {})
        non_rag = by_id.get("non_rag", {})

        sufficient = [
            {
                "ablation": r["ablation"],
                "doc_index": r["kept_doc_indices"][0] if r["kept_doc_indices"] else None,
                "judge_label": r["judge_label"],
                "jaccard_vs_target": r["jaccard_vs_target"],
            }
            for r in mrows
            if str(r["ablation"]).startswith("single:") and r["judge_label"] == "unsafe"
        ]
        necessary = []
        for r in mrows:
            if not str(r["ablation"]).startswith("loo:"):
                continue
            if baseline.get("judge_label") == "unsafe" and r["judge_label"] == "safe":
                necessary.append(
                    {
                        "ablation": r["ablation"],
                        "dropped_doc_index": r["dropped_doc_indices"][0] if r["dropped_doc_indices"] else None,
                        "loo_label": r["judge_label"],
                    }
                )

        # Position: does moving doc i to Context 1 change the all_docs label?
        position_effects = []
        for r in mrows:
            if not str(r["ablation"]).startswith("ctx1:"):
                continue
            if r["judge_label"] != baseline.get("judge_label"):
                position_effects.append(
                    {
                        "ablation": r["ablation"],
                        "ctx1_doc": r.get("ctx1_doc"),
                        "all_docs_label": baseline.get("judge_label"),
                        "ctx1_label": r["judge_label"],
                    }
                )

        minimal_combos = _minimal_unsafe_combos(mrows)
        multi_doc_combos = [c for c in minimal_combos if c["size"] >= 2]

        summary["cases"][cid] = {
            "all_docs_label": baseline.get("judge_label"),
            "non_rag_label": non_rag.get("judge_label"),
            "sufficient_single_docs": sufficient,
            "necessary_docs_by_loo": necessary,
            "position_effects_ctx1": position_effects,
            "minimal_unsafe_doc_sets": minimal_combos,
            "combined_docs_causing_harm": multi_doc_combos,
            "combo_search_ran": cid in combo_plan,
            "interpretation": _interpret(
                baseline.get("judge_label"),
                non_rag.get("judge_label"),
                sufficient,
                necessary,
                multi_doc_combos,
                position_effects,
            ),
        }

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


def _interpret(
    all_label: str | None,
    non_rag_label: str | None,
    sufficient: List[Dict[str, Any]],
    necessary: List[Dict[str, Any]],
    multi_doc_combos: List[Dict[str, Any]],
    position_effects: List[Dict[str, Any]],
) -> str:
    bits: List[str] = []
    if non_rag_label == "unsafe" and all_label == "unsafe":
        bits.append("query_alone_sufficient")
    if sufficient:
        bits.append("single_doc_sufficient")
    if necessary:
        bits.append("loo_identified_necessary_doc")
    if multi_doc_combos:
        bits.append("multi_doc_combination_required")
    if position_effects:
        bits.append("position_sensitive_under_ctx1_reorder")
    if all_label != "unsafe":
        bits.append("baseline_did_not_reproduce_unsafe")
    elif not bits:
        bits.append("harm_reproduced_but_unattributed")
    return "+".join(bits)


if __name__ == "__main__":
    main()
