"""Local HuggingFace inference backend for RAGChecker (bypasses HF Inference API)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.model_runner import ModelRunner, ModelSpec


def make_local_llm_api_func(
    model_spec: ModelSpec,
    generation_cfg: Dict[str, Any],
) -> Callable[[List[str]], List[str]]:
    """
    Build a RAGChecker-compatible batch function that runs via transformers + HF cache.

    RAGChecker/RefChecker call this as ``custom_llm_api_func(prompts) -> responses``.
    """
    runner = ModelRunner(generation_cfg)

    def _generate_batch(prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        print(f"  [local HF] generating {len(prompts)} prompt(s) with {model_spec.id} ...")
        return [runner.generate(model_spec, prompt) for prompt in prompts]

    return _generate_batch
