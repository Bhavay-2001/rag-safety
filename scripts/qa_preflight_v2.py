from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths
from src.utils_io import read_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for MIRAGE capability v2.")
    parser.add_argument("--config", required=True, help="Path to v2 config.")
    return parser.parse_args()


def _load_queries(path: str | Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("prompt_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if pid and text:
                out.append({"prompt_id": pid, "text": text})
    return out


def _load_answers(path: str | Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in read_jsonl(path):
        pid = str(row.get("prompt_id", "")).strip()
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers] if answers.strip() else []
        if pid:
            out[pid] = [str(a).strip() for a in answers if str(a).strip()]
    return out


def _load_contexts(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_jsonl(path):
        pid = str(row.get("prompt_id", "")).strip()
        contexts = row.get("contexts") or []
        docs = []
        for c in contexts:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            docs.append(c)
        if pid:
            out[pid] = docs
    return out


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)
    qa = cfg.get("qa_v2", {})
    if not qa:
        raise ValueError("Config must include qa_v2 section.")

    req = [
        "queries_path",
        "answers_path",
        "oracle_contexts_path",
        "mixed_contexts_path",
    ]
    for key in req:
        path = qa.get(key)
        if not path:
            raise ValueError(f"qa_v2.{key} is required.")
        if not Path(path).exists():
            raise FileNotFoundError(f"qa_v2.{key} missing: {path}")

    queries = _load_queries(qa["queries_path"])
    answers = _load_answers(qa["answers_path"])
    oracle = _load_contexts(qa["oracle_contexts_path"])
    mixed = _load_contexts(qa["mixed_contexts_path"])

    q_ids: Set[str] = {q["prompt_id"] for q in queries}
    a_ids = set(answers.keys())
    o_ids = set(oracle.keys())
    m_ids = set(mixed.keys())
    overlap = q_ids & a_ids & o_ids & m_ids
    if not overlap:
        raise ValueError("No prompt_id overlap across queries/answers/oracle/mixed.")

    answers_nonempty = sum(1 for pid in overlap if answers.get(pid))
    if answers_nonempty == 0:
        raise ValueError("All overlapping rows have empty answers.")

    oracle_nonempty = sum(1 for pid in overlap if oracle.get(pid))
    mixed_nonempty = sum(1 for pid in overlap if mixed.get(pid))
    if oracle_nonempty == 0 or mixed_nonempty == 0:
        raise ValueError("Oracle or mixed contexts are empty for all overlapping rows.")

    print("MIRAGE v2 preflight passed.")
    print(f"- overlap_rows: {len(overlap)}")
    print(f"- answers_nonempty: {answers_nonempty}")
    print(f"- oracle_nonempty: {oracle_nonempty}")
    print(f"- mixed_nonempty: {mixed_nonempty}")


if __name__ == "__main__":
    main()
