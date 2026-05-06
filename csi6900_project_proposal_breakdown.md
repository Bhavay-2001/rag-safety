# CSI6900 Graduate Project Proposal - Technical Breakdown

## Working title
Diagnosing safety regressions in RAG: safe documents, unsafe answers, and retrieval noise

## Core problem framing
- RAG is assumed to improve safety via grounding, but evidence shows unsafe outputs can increase.
- Open question: are regressions driven by unsafe content in retrieved documents, or by context-length and prompt-framing effects that bypass non-RAG safety alignment?
- Additional risk: irrelevant retrieval may confuse the model and contribute to unsafe behavior.

## Scope and objectives
### Core objectives
- Build a reproducible BM25-RAG safety evaluation harness with logging (queries, retrieved passages, scores, prompts, outputs, judge labels).
- Quantify safe-document to unsafe-answer failures and identify common patterns (e.g., disclaimer + procedural steps).
- Compare cases where unsafe procedures are present in retrieved passages vs absent.
- Test retrieval relevance/noise effects with controlled conditions (length-matched padding, random retrieval, hard-negative retrieval).

### Stretch objectives
- Domain-focused corpus or simple document filtering to test if safer retrieval reduces failures.
- Retriever comparison (dense or hybrid) to test sensitivity beyond BM25.

### Out of scope
- End-to-end safety fine-tuning or training large models.
- Large-scale human annotation as the primary evaluation method.
- Literature-only project without experiments.

## Workplan and deliverables
### Deliverables
- D1 (Weeks 1-3): BM25 corpus prep, indexing, end-to-end harness, pilot results.
- D2 (Weeks 4-8): Controlled experiments (non-RAG, BM25-RAG, length-matched, retrieval-noise), sensitivity to top-k and context size.
- D3 (Weeks 7-11): Present-vs-absent analysis on a targeted subset + structured taxonomy of failure modes.
- D4 (Weeks 12-14): Final report, cleaned repo with instructions, and presentation/demo.

### Schedule (high-level)
- Weeks 1-2: setup, corpus, indexing, baseline runs.
- Weeks 3-6: context-length controls + retrieval noise; interim reporting.
- Weeks 7-10: present-vs-absent labeling + deep error analysis.
- Weeks 11-14: consolidate results, write report, prep presentation, finalize repo.

## Methods and practical details
### Data and prompts
- Harmful prompt set derived from red-teaming datasets.
- Adapt a portion to reduce retrieval mismatch (mirroring honors report approach).

### Retrieval setup
- BM25 baseline, top-k passages (default k=5), configurable.
- Log retrieved text and scores for relevance analysis.

### Models
- Small set of instruction-tuned open models chosen for compute feasibility.
- Include at least one RAG-conservative and one RAG-fragile model (per honors report).

### Prompt conditions and controls
- Non-RAG baseline.
- BM25-RAG baseline (docs-only prompting).
- Length-matched non-RAG baseline (padding without useful info).
- Irrelevant retrieval control (random passages, length-matched).
- Hard-negative retrieval control (keyword overlap without useful content).

### Safety evaluation and metrics
- Primary evaluator: automated judge (e.g., Llama Guard) with calibration and spot checks.
- Metrics: unsafe rate, refusal rate, category breakdowns (where possible).
- Report safe-doc to unsafe-answer rates and present-vs-absent results on labeled subset.

## Risks and mitigations
- Compute limits: keep model set small; cache and batch runs.
- Judge reliability: calibration, spot checks, transparent reporting.
- Retrieval mismatch: measure and treat as a controlled variable, not a confounder.

## Evaluation criteria (marking scheme)
- Implementation & reproducibility (30%)
- Experiments & results (40%)
- Analysis (20%)
- Communication & execution (10%)

## Learning objectives
- Design controlled evaluations separating retrieval-content effects from context-length effects.
- Build a reproducible BM25-based RAG pipeline for safety analysis.
- Apply automated safety judges responsibly with calibration.
- Develop defensible failure-mode analysis tied to retrieval relevance and answer presence.
- Communicate methods/results clearly in report and presentation.

