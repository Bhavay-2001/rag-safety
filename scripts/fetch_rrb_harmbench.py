from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils_io import ensure_dir, write_json


TEXT_KEYS = ("text", "prompt", "question", "query", "instruction", "Base_Question")
CATEGORY_KEYS = (
    "category",
    "harm_category",
    "label",
    "taxonomy",
    "type",
    "SemanticCategory",
    "FunctionalCategory",
)
ATTACK_STYLE_KEYS = ("attack_style", "style", "prompt_style", "variant", "Tags")
DATASET_KEYS = ("dataset", "source_dataset", "subset", "split_name")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-provenance harmful prompt CSV from CSV sources."
    )
    parser.add_argument(
        "--csv-source",
        action="append",
        default=[],
        help=(
            "CSV source spec: source_benchmark|source_dataset|path. "
            "Example: RRB|haizelab_rrb|data/raw/rrb.csv"
        ),
    )
    parser.add_argument(
        "--output-full",
        default="data/prompts/rrb_harmbench_full.csv",
        help="Canonical full CSV output path.",
    )
    parser.add_argument(
        "--output-subset",
        default="data/prompts/rrb_harmbench_subset_1500.csv",
        help="Stratified subset output path.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1500,
        help="Target row count for stratified subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--manifest-path",
        default="data/raw/SOURCE_MANIFEST.json",
        help="Path to provenance manifest JSON.",
    )
    parser.add_argument(
        "--split",
        default="run",
        help="split column value to set in canonical outputs.",
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pick_first(
    row: Dict[str, Any],
    keys: Iterable[str],
    default: str = "",
    allow_value_fallback: bool = False,
) -> str:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    if allow_value_fallback:
        for val in row.values():
            if val is not None and str(val).strip():
                return str(val).strip()
    return default


def _normalize_row(
    row: Dict[str, Any],
    source_benchmark: str,
    source_dataset: str,
    split: str,
) -> Dict[str, str]:
    text = _pick_first(row, TEXT_KEYS, allow_value_fallback=True)
    category = _pick_first(row, CATEGORY_KEYS, default="unknown")
    attack_style = _pick_first(row, ATTACK_STYLE_KEYS, default="unspecified")
    row_dataset = _pick_first(row, DATASET_KEYS, default=source_dataset)

    return {
        "text": text,
        "source_benchmark": source_benchmark,
        "source_dataset": row_dataset or source_dataset,
        "category": category,
        "attack_style": attack_style,
        "split": split,
    }


def _parse_csv_spec(spec: str) -> Tuple[str, str, Path]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid --csv-source spec: {spec}")
    source_benchmark, source_dataset, path = parts
    return source_benchmark.strip(), source_dataset.strip(), Path(path.strip())


def _load_csv_source(
    source_benchmark: str,
    source_dataset: str,
    path: Path,
    split: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = _normalize_row(row, source_benchmark, source_dataset, split)
            if not normalized["text"]:
                continue
            rows.append(normalized)
    return rows


def _load_json_source(
    source_benchmark: str,
    source_dataset: str,
    path: Path,
    split: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    def _emit(record: Dict[str, Any]) -> None:
        normalized = _normalize_row(record, source_benchmark, source_dataset, split)
        if normalized["text"]:
            rows.append(normalized)

    # Common RRB/HarmBench patterns.
    if isinstance(payload, dict):
        if isinstance(payload.get("goals"), list):
            for goal in payload["goals"]:
                _emit({"text": str(goal), "category": source_dataset, "attack_style": "direct"})
            return rows
        if isinstance(payload.get("behaviors"), list):
            for item in payload["behaviors"]:
                if isinstance(item, dict):
                    _emit(item)
                else:
                    _emit({"text": str(item)})
            return rows
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict):
                    _emit(item)
                else:
                    _emit({"text": str(item)})
            return rows

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                _emit(item)
            else:
                _emit({"text": str(item)})
        return rows

    # Generic nested structures (e.g., harmfulqs hierarchy).
    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, str):
            _emit(
                {
                    "text": obj,
                    "category": path[-1] if path else source_dataset,
                    "source_dataset": source_dataset,
                }
            )
            return
        if isinstance(obj, list):
            for item in obj:
                _walk(item, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, path + [str(k)])

    _walk(payload, [])
    return rows


def _load_jsonl_source(
    source_benchmark: str,
    source_dataset: str,
    path: Path,
    split: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            normalized = _normalize_row(record, source_benchmark, source_dataset, split)
            if normalized["text"]:
                rows.append(normalized)
    return rows


def _load_source(
    source_benchmark: str,
    source_dataset: str,
    path: Path,
    split: str,
) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_source(source_benchmark, source_dataset, path, split)
    if suffix == ".json":
        return _load_json_source(source_benchmark, source_dataset, path, split)
    if suffix == ".jsonl":
        return _load_jsonl_source(source_benchmark, source_dataset, path, split)
    raise ValueError(f"Unsupported source file type: {path}")


def _assign_prompt_ids(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    canonical: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        with_id = dict(row)
        with_id["prompt_id"] = f"hb_{i:06d}"
        canonical.append(with_id)
    return canonical


def _stratified_subset(rows: List[Dict[str, str]], subset_size: int, seed: int) -> List[Dict[str, str]]:
    if subset_size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["source_dataset"], row["category"])
        buckets.setdefault(key, []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    subset: List[Dict[str, str]] = []

    # Round-robin quota so smaller strata are retained.
    while len(subset) < subset_size:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if not bucket:
                continue
            subset.append(bucket.pop())
            progressed = True
            if len(subset) >= subset_size:
                break
        if not progressed:
            break
    return subset


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "prompt_id",
        "text",
        "source_benchmark",
        "source_dataset",
        "category",
        "attack_style",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_manifest(
    manifest_path: Path,
    source_entries: List[Dict[str, Any]],
    produced_files: List[Path],
) -> None:
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    manifest.setdefault("generated_at_utc", _now_utc())
    manifest.setdefault("sources", [])
    manifest.setdefault("produced_files", [])

    manifest["generated_at_utc"] = _now_utc()
    manifest["sources"].extend(source_entries)
    for p in produced_files:
        manifest["produced_files"].append(
            {
                "path": str(p),
                "sha256": _sha256_file(p),
                "timestamp_utc": _now_utc(),
            }
        )
    write_json(manifest_path, manifest)


def main() -> None:
    args = _parse_args()
    if not args.csv_source:
        raise ValueError("At least one --csv-source is required for this script.")

    full_rows: List[Dict[str, str]] = []
    source_entries: List[Dict[str, Any]] = []

    for spec in args.csv_source:
        source_benchmark, source_dataset, path = _parse_csv_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        rows = _load_source(source_benchmark, source_dataset, path, split=args.split)
        full_rows.extend(rows)
        source_entries.append(
            {
                "source_type": "csv",
                "source_benchmark": source_benchmark,
                "source_dataset": source_dataset,
                "path": str(path),
                "sha256": _sha256_file(path),
                "timestamp_utc": _now_utc(),
                "license_note": "Verify upstream dataset license before publication.",
            }
        )

    canonical_rows = _assign_prompt_ids(full_rows)
    subset_rows = _stratified_subset(canonical_rows, args.subset_size, args.seed)

    output_full = Path(args.output_full)
    output_subset = Path(args.output_subset)
    _write_csv(output_full, canonical_rows)
    _write_csv(output_subset, subset_rows)

    manifest_path = Path(args.manifest_path)
    _update_manifest(manifest_path, source_entries, [output_full, output_subset])

    print(f"Wrote canonical prompts to {output_full} ({len(canonical_rows)} rows)")
    print(f"Wrote stratified subset to {output_subset} ({len(subset_rows)} rows)")
    print(f"Updated source manifest at {manifest_path}")


if __name__ == "__main__":
    main()
