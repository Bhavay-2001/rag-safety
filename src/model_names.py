"""Shared model naming: resolve aliases → hub ids → LiteLLM provider strings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.model_runner import ModelSpec, load_models_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_CONFIG = _REPO_ROOT / "configs" / "models.yaml"

_PROVIDER_PREFIXES = (
    ("huggingface/", "huggingface"),
    ("hf/", "hf"),
    ("openai/", "openai"),
    ("anthropic/", "anthropic"),
    ("bedrock/", "bedrock"),
)

_LITELLM_PROVIDER = {
    "hf": "huggingface",
    "openai": "openai",
    "anthropic": "anthropic",
    "bedrock": "bedrock",
}


def _split_provider_prefix(name: str) -> tuple[Optional[str], str]:
    n = (name or "").strip()
    for prefix, provider in _PROVIDER_PREFIXES:
        if n.startswith(prefix):
            return provider, n[len(prefix) :]
    return None, n


def index_models(models_cfg: Dict[str, Any]) -> Dict[str, ModelSpec]:
    """Map each model alias and hub id to its ModelSpec."""
    index: Dict[str, ModelSpec] = {}
    for entries in models_cfg.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            spec = ModelSpec(
                id=entry["id"],
                alias=entry["alias"],
                provider=entry["provider"],
            )
            index[spec.id] = spec
            index[spec.alias] = spec
    return index


def resolve_model_spec(
    name: str,
    models_config_path: str | Path | None = None,
) -> ModelSpec:
    """
    Resolve a model reference to a ModelSpec.

    Accepted forms:
      - alias from configs/models.yaml (e.g. Llama-3-8B-Instruct)
      - hub id org/repo (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
      - prefixed hub id (e.g. hf/Qwen/Qwen3-30B-A3B-Instruct-2507)
    """
    if not (name or "").strip():
        raise ValueError("Model name is required.")

    prefix_provider, model_part = _split_provider_prefix(name)
    cfg_path = Path(models_config_path) if models_config_path else DEFAULT_MODELS_CONFIG
    models_cfg = load_models_config(cfg_path)
    index = index_models(models_cfg)

    if model_part in index:
        return index[model_part]

    if "/" in model_part:
        provider = prefix_provider or "hf"
        if provider == "huggingface":
            provider = "hf"
        return ModelSpec(id=model_part, alias=model_part, provider=provider)

    raise ValueError(
        f"Unknown model {name!r}. Use hf/<org>/<repo> or an alias defined in {cfg_path}."
    )


def normalize_litellm_model(
    name: str,
    models_config_path: str | Path | None = None,
) -> str:
    """Return provider-prefixed model string for LiteLLM (e.g. huggingface/org/repo)."""
    spec = resolve_model_spec(name, models_config_path)
    litellm_provider = _LITELLM_PROVIDER.get(spec.provider, spec.provider)
    if litellm_provider == "huggingface" and "/" not in spec.id:
        raise ValueError(
            f"HuggingFace model id must be org/repo for LiteLLM (got {spec.id!r} from {name!r})."
        )
    return f"{litellm_provider}/{spec.id}"


def model_slug(
    name: str,
    models_config_path: str | Path | None = None,
) -> str:
    """Filesystem-safe slug from resolved hub id."""
    spec = resolve_model_spec(name, models_config_path)
    slug = spec.id.lower().replace("/", "_").replace(":", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in slug)
