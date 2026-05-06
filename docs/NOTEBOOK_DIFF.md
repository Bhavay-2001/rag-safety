# Notebook vs Repo Pipeline (CSI4900)

This repo implements a scripted pipeline that is **conceptually similar** to the notebook,
but it is not a 1:1 port. The table below highlights the differences and how the tiny
config aligns with the notebook's intent.

## Key differences

### Storage and paths
- Notebook: uses Google Drive paths under `/content/drive/MyDrive/RAG_Safety_Project/`.
- Repo: uses local relative paths (e.g., `data/`, `indexes/`, `outputs/`).

### Corpus build
- Notebook: filters Wikipedia to **finance/legal/cybersecurity** using keyword counts.
- Repo: supports optional filtering, but **defaults to no filtering**.
  - To align: enable `corpus.filtering.enabled` and provide a keywords map.

### Retriever
- Both: BM25 with paragraph‑level chunks and top‑k retrieval.
- Notebook: uses NLTK tokenization.
- Repo: uses a simple regex tokenizer.

### Models
- Notebook: 1–2B models (TinyLlama, SmolLM, etc).
- Repo: default configs focus on 7B+ models.
  - Tiny config uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

### Prompt modes
- Notebook: **three modes** (non‑RAG, RAG‑only, ALL).
- Repo: supports the same three core modes, plus extra controls
  (length‑match, random docs, hard‑negative).
  - Tiny config uses only the three core modes for speed.

### Safety judge
- Notebook: Llama Guard 3‑1B.
- Repo: default is Llama Guard 2 (paper‑aligned).
  - Tiny config switches to **Llama Guard 3‑1B** for similarity.

## How the tiny config mirrors the notebook (scaled down)
- Small model: TinyLlama‑1.1B.
- Small prompt set: `data/prompts/tiny_prompts.csv` (3 prompts).
- Small corpus: `--max-docs 2000`.
- Same three modes: non‑RAG, RAG‑docs, RAG‑LLM.
- Same safety judge class: Llama Guard 3‑1B.

## What to run
```bash
bash scripts/run_all.sh configs/tiny.yaml 2000
```

Outputs are saved under `outputs/tiny/<run_id>/`.
