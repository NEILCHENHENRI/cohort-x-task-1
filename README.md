# CohortX Task 1 — Cohort Criteria Extraction

Extracts 6 structured fields from PMC biomedical articles: `conditions`, `study_type`, `sex`, `minimum_age`, `maximum_age`, `eligibility_criteria`.

## Repository layout

Files are grouped by category; every folder below is a Python package, so all
scripts run in module form from the repo root (`python -m <pkg>.<module>`).

```
run_submission.py             bare-script entry point for graders (offline Path B)
common/                       shared by everything
  config.py                   constants (model names, queries, metric weights)
  parser.py                   NXMLParser + HierarchicalNXMLParser (PMC JATS/NXML -> dicts)
  evaluate.py                 competition metrics (number sim, BioBERT cosine, FM3S)
finetuned_models/             Path A — week-1 fine-tuned experiments (HISTORICAL)
  models.py                   all training components and CohortXPipeline
  train.py                    CLI entry point for training
  predict.py                  inference with the fine-tuned model chain
local_llm/                    Path B — THE LIVE SUBMISSION
  predict_ollama.py           LIVE offline inference via Ollama (Qwen 1.5B)
  predict_qwen.py             alt backend via Transformers (Qwen3-4B, 8-bit)
retrieval/                    block-based BGE retrieval subsystem (Neil)
  retrieval.py                field-specific evidence retrieval over parsed blocks
  RETRIEVAL_ALGO_v1.md        the retrieval algorithm, step by step
gepa_opt/                     dev-time GEPA prompt optimization (never on the offline path;
                              named gepa_opt to avoid shadowing the installed `gepa` library)
  data_split.py               deterministic train/val/holdout split (results/splits.json)
  dspy_program.py             DSPy signature + program + GEPA metric (field-specific feedback)
  optimize_gepa.py            run GEPA (Claude reflection) -> optimized instruction
  run_eval.py                 score the offline path (baseline / optimized) on a split
  check_transfer.py           re-score the shipped prompt vs the baseline
  scenario_test.py            qualitative pred-vs-gold inspection on sample papers
notebooks/                    explore.ipynb, explore_failure_mode.ipynb, cohortx_gepa_colab.ipynb
tests/                        test_refactor.py
data/                         download_data.py (dataset itself is gitignored)
results/                      baseline.json / transfer_check.json (before / after scores),
                              seed_instruction.txt / optimized_instruction.txt (the two prompts),
                              splits.json, holdout_ids.json, gepa_optimized.json, findings.md
```

## About the project

Given a full-text PMC article (`.nxml`), predict the 6 clinical-trial eligibility fields
above. Scoring is a weighted blend across the fields (eligibility_criteria is half the
score) that rewards correct **meaning**, not exact strings — see
[metric weights](#evaluation-metric-weights).

Two solution paths live in this repo:

- **Path A — fine-tuned specialist chain** (`finetuned_models/`): a pipeline of small
  fine-tuned models. Week-1 experimentation, kept for history.
- **Path B — single-model generative pipeline** (`local_llm/predict_ollama.py`): **the live
  submission.** One local model (Qwen 1.5B via Ollama) extracts all 6 fields from a prompt,
  fully offline.

Two additions build on Path B:

- **Retrieval subsystem** (`retrieval/`, Neil): a more advanced way to select which
  paragraphs the model reads (block-based parsing + BGE field-specific retrieval).
- **GEPA prompt optimization** (`gepa_opt/`, Evan): automatically rewrites the Path-B
  instruction to raise the score. This is a dev-time tool; it never runs on the offline
  submission path.

---

# Base pipeline (Neil)

## Setup

```bash
pip install transformers sentence-transformers datasets torch lxml \
            pandas openpyxl scikit-learn joblib tqdm word2number
```

For evaluation only (no training):
```bash
pip install sentence-transformers spacy word2number nltk tqdm
python -m spacy download en_core_web_sm
```

## Training

```bash
python -m finetuned_models.train \
  --data_dir /path/to/data \
  --nxml_dir /path/to/PMC_NXML_Archives \
  --gpu
```

## Inference

**Ollama (local, Qwen 1.5B) — THE SUBMISSION:**
```bash
ollama serve
ollama pull qwen2.5:1.5b
python -m local_llm.predict_ollama \
	--data_dir /path/to/data \
	--nxml_dir /path/to/nxml \
	--test
# equivalently, the bare-script entry point graders can use:
python run_submission.py --data_dir /path/to/data --nxml_dir /path/to/nxml --test
```

**Transformers (Qwen3-4B, 8-bit):**

```bash
pip install accelerate bitsandbytes
python -m local_llm.predict_qwen \
  --data_dir /path/to/data \
  --nxml_dir /path/to/PMC_NXML_Archives \
  --test
```

## Retrieval subsystem

A newer, more precise way to build the model's context (`retrieval/retrieval.py`), fully
described in [retrieval/RETRIEVAL_ALGO_v1.md](retrieval/RETRIEVAL_ALGO_v1.md): parse each
paper into a section tree + atomic blocks, embed the blocks with `BAAI/bge-small-en-v1.5`,
retrieve field-specific evidence (condition/study-type, demographics, eligibility), rerank
by section metadata, and pack a per-field context.

It is a library-style module (import its functions, e.g. `export_retrieved_contexts(...)`;
it reads parsed JSON and writes to gitignored output dirs). It is **not yet wired into the
shipped Path B** — combining it with the GEPA-optimized prompt is a listed next step.

## Evaluation

```python
from common.evaluate import evaluate_fast
scores_df = evaluate_fast(preds_df, gold_df)
```

## Key findings from exploration

**Article structure:** Eligibility criteria, age, and sex are almost always in a section titled "Participants", "Methods", or "Eligibility" — rarely in the abstract. Conditions are usually in the abstract and keywords.

**minimum_age failure mode:** 344/416 training documents return "Not Specified". Most of these are genuine data gaps — the article text never states the minimum age explicitly. The gold labels come from ClinicalTrials.gov registrations, not the paper itself, so no extraction method can recover them from the NXML alone.

**MiniLM section filtering:** Using max(title score, text score) rather than concatenated snippet score significantly improves recall for sections with informative titles but generic opening sentences (e.g. a "Participants" section that starts with imaging protocol text).

**sex metric:** Always scores 1.0 because "ALL/MALE/FEMALE" contain no numbers, so both pred and ref produce empty number sets — the number similarity metric returns 1.0 for two empty sets by definition. This field is effectively unscored.

## Evaluation metric weights

| Field | Weight | Metric |
|---|---|---|
| eligibility_criteria | 0.50 | FM3S |
| conditions | 0.15 | BioBERT cosine |
| minimum_age | 0.10 | Number similarity |
| maximum_age | 0.10 | Number similarity |
| study_type | 0.10 | BioBERT cosine |
| sex | 0.05 | Number similarity |

---

# GEPA prompt optimization (Evan)

GEPA (DSPy's reflective optimizer) automatically rewrites the Path-B **instruction** using
a strong **Claude** reflection model at *dev time*, then ships the single optimized
instruction string into the **offline** Ollama path. Claude is never on the inference path.
Full write-up: [results/findings.md](results/findings.md).

### Prerequisites (one-time)

```bash
# 1. Install everything (adds dspy, ollama, python-dotenv on top of the base deps)
pip install -r requirements.txt

# 2. Start the local student model (leave `ollama serve` running in its own terminal)
ollama serve
ollama pull qwen2.5:1.5b

# 3. Add your Claude key for the reflection model (dev-time only, gitignored)
cp .env.example .env          # then edit .env:  ANTHROPIC_API_KEY=sk-ant-...

# 4. Download the dataset (via kagglehub)
python data/download_data.py
```

### Run the pipeline

```bash
# 1. BEFORE — score the default hand-written prompt        (~20-30 min, needs Ollama running)
python -m gepa_opt.run_eval --split holdout \
    --instruction_file results/seed_instruction.txt --output results/baseline.json

# 2. OPTIMIZE — GEPA rewrites the prompt via Claude         (~30 min, ~$0.12 of credits, needs .env key)
python -m gepa_opt.optimize_gepa --max_metric_calls 150
#    -> writes results/optimized_instruction.txt

# 3. AFTER — score the optimized prompt on the same papers  (~20-30 min, needs Ollama running)
python -m gepa_opt.run_eval --split holdout \
    --instruction_file results/optimized_instruction.txt --output results/transfer_check.json
```

Both evals score the **same** 60 held-out papers (`results/holdout_ids.json`) on the real
offline path, swapping only the instruction — an honest, isolated before/after. The first
run auto-creates the frozen train/val/holdout split (`results/splits.json`).

### Ship it

Paste `results/optimized_instruction.txt` into `local_llm/predict_ollama.py`'s
`INSTRUCTION_PROSE`. The offline path then runs that fixed string on local Qwen with **no
API, no internet, no GPU**.

### Result

Held-out (n=60): composite **0.629 → 0.685 (+0.056, +9% rel)** for **$0.118**. Biggest
wins on `conditions` (+0.30) and `study_type` (+0.11), plus `eligibility_criteria` (+0.04,
the 50%-weight field); `minimum_age` regressed (a metric artifact — gold ages come from
ClinicalTrials.gov, so an honest "Not Specified" scores 0 against a gold of "18 Years").
