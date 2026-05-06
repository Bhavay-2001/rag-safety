from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build markdown report tables from run artifacts.")
    parser.add_argument("--qa-run-dir", required=True, help="Run dir containing qa_metrics.json.")
    parser.add_argument("--safety-run-dir", required=True, help="Run dir containing metrics.json.")
    parser.add_argument(
        "--domain-run",
        action="append",
        default=[],
        help="Optional domain run mapping: domain=run_dir_with_metrics_json",
    )
    parser.add_argument("--output-dir", default="reports/tables", help="Output table directory.")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capability_table(qa: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Capability Baseline")
    lines.append("")
    lines.append(f"- dataset: `{qa.get('dataset')}`")
    lines.append(f"- variant: `{qa.get('variant')}`")
    lines.append(f"- subset_size: `{qa.get('subset_size')}`")
    lines.append("")
    lines.append("| model | condition | em | f1 | contains_answer_rate | refusal_rate | faithfulness | context_precision | hallucination_proxy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, conds in qa.get("by_model_condition", {}).items():
        for condition, vals in conds.items():
            lines.append(
                "| {model} | {condition} | {em} | {f1} | {contains} | {refusal} | {faith} | {ctx} | {hall} |".format(
                    model=model,
                    condition=condition,
                    em=_fmt(vals.get("em")),
                    f1=_fmt(vals.get("f1")),
                    contains=_fmt(vals.get("contains_answer_rate")),
                    refusal=_fmt(vals.get("refusal_rate")),
                    faith=_fmt(vals.get("faithfulness_score")),
                    ctx=_fmt(vals.get("context_precision")),
                    hall=_fmt(vals.get("hallucination_proxy")),
                )
            )
    lines.append("")
    lines.append("## Doc Reliance Gap")
    lines.append("")
    lines.append("| model | em_gap_rag_minus_random | f1_gap_rag_minus_random | contains_gap_rag_minus_random |")
    lines.append("|---|---:|---:|---:|")
    for model, vals in qa.get("doc_reliance_gap", {}).items():
        lines.append(
            "| {model} | {em_gap} | {f1_gap} | {contains_gap} |".format(
                model=model,
                em_gap=_fmt(vals.get("em_gap_rag_minus_random")),
                f1_gap=_fmt(vals.get("f1_gap_rag_minus_random")),
                contains_gap=_fmt(vals.get("contains_gap_rag_minus_random")),
            )
        )
    return "\n".join(lines) + "\n"


def _tradeoff_table(qa: Dict[str, Any], safety: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Safety Utility Tradeoff")
    lines.append("")
    lines.append("| model | condition | qa_f1 | qa_em | unsafe_rate | refusal_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    qa_map = qa.get("by_model_condition", {})
    safe_map = safety.get("by_model_condition", {})
    all_models = sorted(set(qa_map.keys()) | set(safe_map.keys()))
    for model in all_models:
        qa_conds = qa_map.get(model, {})
        safe_conds = safe_map.get(model, {})
        all_conditions = sorted(set(qa_conds.keys()) | set(safe_conds.keys()))
        for condition in all_conditions:
            qvals = qa_conds.get(condition, {})
            svals = safe_conds.get(condition, {})
            lines.append(
                "| {model} | {condition} | {f1} | {em} | {unsafe} | {refusal} |".format(
                    model=model,
                    condition=condition,
                    f1=_fmt(qvals.get("f1", 0.0)),
                    em=_fmt(qvals.get("em", 0.0)),
                    unsafe=_fmt(svals.get("unsafe_rate", 0.0)),
                    refusal=_fmt(svals.get("refusal_rate", 0.0)),
                )
            )
    return "\n".join(lines) + "\n"


def _parse_domain_runs(values: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --domain-run value: {item}. Expected domain=path.")
        domain, path = item.split("=", maxsplit=1)
        parsed.append((domain.strip(), Path(path.strip())))
    return parsed


def _domain_matrix(domain_runs: List[Tuple[str, Path]]) -> str:
    lines: List[str] = []
    lines.append("# Domain Diagnostic Matrix")
    lines.append("")
    lines.append("| domain | model | non_rag_unsafe_rate | rag_docs_unsafe_rate | rag_llm_unsafe_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for domain, run_dir in domain_runs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            lines.append(f"| {domain} | n/a | n/a | n/a | n/a |")
            continue
        metrics = _read_json(metrics_path)
        for model, conds in metrics.get("by_model_condition", {}).items():
            lines.append(
                "| {domain} | {model} | {non_rag} | {rag_docs} | {rag_llm} |".format(
                    domain=domain,
                    model=model,
                    non_rag=_fmt(conds.get("non_rag", {}).get("unsafe_rate", 0.0)),
                    rag_docs=_fmt(conds.get("rag_docs", {}).get("unsafe_rate", 0.0)),
                    rag_llm=_fmt(conds.get("rag_llm", {}).get("unsafe_rate", 0.0)),
                )
            )
    if not domain_runs:
        lines.append("| finance | pending | pending | pending | pending |")
        lines.append("| cybersecurity | pending | pending | pending | pending |")
        lines.append("| law | pending | pending | pending | pending |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    qa = _read_json(Path(args.qa_run_dir) / "qa_metrics.json")
    safety = _read_json(Path(args.safety_run_dir) / "metrics.json")
    domains = _parse_domain_runs(args.domain_run)

    _write(out_dir / "capability_baseline.md", _capability_table(qa))
    _write(out_dir / "safety_utility_tradeoff.md", _tradeoff_table(qa, safety))
    _write(out_dir / "domain_diagnostic_matrix.md", _domain_matrix(domains))
    print(f"Wrote report tables to {out_dir}")


if __name__ == "__main__":
    main()
