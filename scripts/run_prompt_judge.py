from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import load_config, resolve_paths
from src.judge_runner import SafetyJudge
from src.utils_io import write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safety judge on user prompts (optional).")
    parser.add_argument("--config", required=True, help="Path to config.")
    parser.add_argument("--output", default=None, help="Optional output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)

    prompts_path = cfg.get("prompts", {}).get("data_path") or cfg.get("prompts", {}).get("path")
    judge_cfg = cfg.get("judge", {})
    model_id = judge_cfg.get("response_model", "meta-llama/Llama-Guard-3-8B")
    out_path = Path(args.output) if args.output else Path(cfg.get("output", {}).get("root", "outputs/runs")) / "prompt_judge.jsonl"

    judge = SafetyJudge(model_id, generation_cfg=judge_cfg.get("generation", {}), judge_cfg=judge_cfg)
    rows = []
    with Path(prompts_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt_id = str(row.get("prompt_id") or idx)
            text = row.get("text") or row.get("question") or row.get("prompt") or row.get("query") or ""
            if not str(text).strip():
                continue
            result = judge.judge(str(text), "", judge_target="prompt")
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "label": result.get("label"),
                    "categories": result.get("categories") or [],
                    "judge_parse_error": result.get("judge_parse_error", False),
                    "judge_parse_error_reason": result.get("judge_parse_error_reason"),
                    "raw": result.get("raw"),
                }
            )
    write_jsonl(out_path, rows)
    print(f"Wrote prompt judge labels to {out_path}")


if __name__ == "__main__":
    main()
