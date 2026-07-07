# CohortX Task 1 — Cohort Criteria Extraction

Extracts 6 structured fields from PMC biomedical articles: `conditions`, `study_type`, `sex`, `minimum_age`, `maximum_age`, `eligibility_criteria`.

## Structure

Files are grouped by category; every folder below is a Python package, so all
scripts run in module form from the repo root (`python -m <pkg>.<module>`).

```
run_submission.py             bare-script entry point for graders (offline Path B)
common/                       shared by everything
  config.py                   constants (model names, queries, metric weights)
  parser.py                   NXMLParser — parses PMC JATS/NXML to structured dicts
  evaluate.py                 competition metrics (number sim, BioBERT cosine, FM3S)
finetuned_models/             Path A — week-1 fine-tuned experiments (HISTORICAL)
  models.py                   all training components and CohortXPipeline
  train.py                    CLI entry point for training
  predict.py                  inference with the fine-tuned model chain
local_llm/                    Path B — THE LIVE SUBMISSION
  predict_ollama.py           LIVE offline inference via Ollama (Qwen 1.5B)
  predict_qwen.py             alt backend via Transformers (Qwen3-4B, 8-bit)
gepa_opt/                     dev-time GEPA prompt optimization (never on the offline path;
                              named gepa_opt to avoid shadowing the installed `gepa` library)
  data_split.py               deterministic train/val/holdout split (results/splits.json)
  dspy_program.py             DSPy signature + program + GEPA metric (field-specific feedback)
  optimize_gepa.py            run GEPA (Claude reflection) -> optimized instruction
  run_eval.py                 score the offline path (baseline / optimized) on a split
  scenario_test.py            qualitative pred-vs-gold inspection on sample papers
notebooks/                    explore.ipynb, explore_failure_mode.ipynb, cohortx_gepa_colab.ipynb
tests/                        test_refactor.py
data/                         download_data.py (dataset itself is gitignored)
results/                      baseline.json, gepa_optimized.json, optimized_instruction.txt, splits.json
```

Two solution paths live in this repo: **Path A** (fine-tuned specialist chain in
`finetuned_models/`) and **Path B** (the live single-model generative pipeline in
`local_llm/predict_ollama.py`). The GEPA work optimizes **Path B only**.

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

## GEPA Prompt Optimization (Path B)

GEPA (DSPy's reflective optimizer) rewrites the Path-B **instruction** using a strong
**Claude** reflection model at *dev time*, then ships the single optimized instruction
into the **offline** Ollama path. The reflection model is never on the inference path.

```bash
python data/download_data.py                                               # data via kagglehub
# BEFORE — default hand-written prompt (-> 0.629)
python -m gepa_opt.run_eval --split holdout \
    --instruction_file results/seed_instruction.txt --output results/baseline.json
python -m gepa_opt.optimize_gepa --max_metric_calls 150                    # GEPA (needs .env key)
# AFTER — GEPA-optimized prompt (-> 0.685, the shipped one)
python -m gepa_opt.run_eval --split holdout \
    --instruction_file results/optimized_instruction.txt --output results/transfer_check.json
```

`.env` (gitignored) holds `ANTHROPIC_API_KEY`. To **ship** an optimized prompt, paste
`results/optimized_instruction.txt` into `local_llm/predict_ollama.py`'s `INSTRUCTION_PROSE`.
Full write-up in [results/findings.md](results/findings.md).

**Result (held-out n=60):** composite **0.629 → 0.685 (+0.056, +9% rel)** for **$0.118**.
Big wins on `conditions` (+0.30) and `study_type` (+0.11), plus `eligibility_criteria`
(+0.04, the 50%-weight field); `minimum_age` regressed (a metric artifact — gold ages come
from ClinicalTrials.gov, so an honest "Not Specified" scores 0 against a gold of "18 Years").

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
