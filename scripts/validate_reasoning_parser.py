from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_runner import ModelRunner, ModelSpec


def _run_case(name: str, raw: str, expected_response: str, expected_status: str) -> None:
    spec = ModelSpec(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        alias="DeepSeek-R1D-Qwen-7B",
        provider="hf",
        reasoning_mode="reasoning",
        reasoning_parser="think_tags",
    )
    out = ModelRunner._normalize_response(raw, spec)
    ok = out["response"] == expected_response and out["response_parse_status"] == expected_status
    if not ok:
        raise AssertionError(
            f"{name} failed: response={out['response']!r} status={out['response_parse_status']!r}"
        )
    print(f"PASS {name}: status={out['response_parse_status']} response={out['response']!r}")


def main() -> None:
    _run_case(
        "think_then_final",
        "<think>reasoning</think>final answer",
        "final answer",
        "ok",
    )
    _run_case(
        "think_only_fallback",
        "<think>reasoning only</think>",
        "<think>reasoning only</think>",
        "empty_final_fallback_raw",
    )
    _run_case(
        "no_tag_passthrough",
        "plain answer",
        "plain answer",
        "no_tag",
    )
    _run_case(
        "multiple_close_tags_last_split",
        "<think>a</think>middle</think>final",
        "final",
        "ok",
    )
    print("All parser checks passed.")


if __name__ == "__main__":
    main()
