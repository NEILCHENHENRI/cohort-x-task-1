# CohortX Task 1 — Cohort Criteria Extraction

Extracts 6 structured fields from PMC biomedical articles: `conditions`, `study_type`, `sex`, `minimum_age`, `maximum_age`, `eligibility_criteria`.

## Structure

```
config.py                 constants (model names, queries, metric weights)
parser.py                 NXMLParser — parses PMC JATS/NXML to structured dicts
models.py                 all training components and CohortXPipeline (Path A)
evaluate.py               competition metrics (number sim, BioBERT cosine, FM3S)
train.py                  CLI entry point for training (Path A)
predict_ollama.py         LIVE offline inference via Ollama (Qwen 1.5B) — Path B
predict_qwen.py           local/Colab inference via Transformers (Qwen3-4B, 8-bit)
explore.ipynb             NXML exploration and failure analysis

# ── GEPA prompt optimization (dev-time) ──────────────────────────────
download_data.py          fetch competition data via kagglehub
data_split.py             deterministic train/val/holdout split (results/splits.json)
dspy_program.py           DSPy signature + program + GEPA metric (field-specific feedback)
optimize_gepa.py          run GEPA (Claude reflection) -> optimized instruction
run_eval.py               score the offline path (baseline / optimized) on a split
scenario_test.py          qualitative pred-vs-gold inspection on sample papers
cohortx_gepa_colab.ipynb  end-to-end GPU-aware notebook for demos
```

Two solution paths live in this repo: **Path A** (fine-tuned specialist chain:
`train.py`, `models.py`, `predict.py`) and **Path B** (the live single-model generative
pipeline: `predict_ollama.py`). The GEPA work optimizes **Path B only**.

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
python train.py \
  --data_dir /path/to/data \
  --nxml_dir /path/to/PMC_NXML_Archives \
  --gpu
```

## Inference

**Ollama (local, Qwen 1.5B):**
```bash
ollama pull qwen2.5:1.5b
python predict_ollama.py \
	--data_dir /path/to/data
	--nxml_dir /path/to/nxml
	--test
```

**Transformers (Qwen3-4B, 8-bit):**

```bash
pip install accelerate bitsandbytes
python predict_qwen.py \
  --data_dir /path/to/data \
  --nxml_dir /path/to/PMC_NXML_Archives \
  --test
```

## GEPA Prompt Optimization (Path B)

GEPA (DSPy's reflective optimizer) rewrites the Path-B **instruction** using a strong
**Claude** reflection model at *dev time*, then ships the single optimized instruction
into the **offline** Ollama path. The reflection model is never on the inference path.

```bash
python download_data.py                                              # data via kagglehub
python run_eval.py --split holdout --output results/baseline.json    # "before"
python optimize_gepa.py --max_metric_calls 150                       # GEPA (needs .env key)
python run_eval.py --split holdout \
    --instruction_file results/optimized_instruction.txt \
    --output results/optimized.json                                  # "after"
```

`.env` (gitignored) holds `ANTHROPIC_API_KEY`. To **ship** an optimized prompt, paste
`results/optimized_instruction.txt` into `predict_ollama.py`'s `INSTRUCTION_PROSE`.
Full write-up in [results/findings.md](results/findings.md).

**Result (held-out n=60):** composite **0.629 → 0.686 (+0.057, +9% rel)** for **$0.13**.
Big wins on `conditions` (+0.29) and `eligibility_criteria` (+0.06, the 50%-weight
field); `minimum_age` regressed (a metric artifact — gold ages come from
ClinicalTrials.gov, so honest "Not Specified" scores 0 against a gold of "18 Years").

## Evaluation

```python
from evaluate import evaluate_fast
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
