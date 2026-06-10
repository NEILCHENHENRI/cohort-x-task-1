# CohortX Task 1 — Cohort Criteria Extraction

Extracts 6 structured fields from PMC biomedical articles: `conditions`, `study_type`, `sex`, `minimum_age`, `maximum_age`, `eligibility_criteria`.

## Structure

```
config.py           constants (model names, queries, metric weights)
parser.py           NXMLParser — parses PMC JATS/NXML to structured dicts
models.py           all training components and CohortXPipeline
evaluate.py         competition metrics (number sim, BioBERT cosine, FM3S)
train.py            CLI entry point for training
predict_ollama.py   local inference via Ollama (Qwen 1.5B default)
explore.ipynb       NXML exploration and failure analysis
```

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
python predict_ollama.py --data_dir /path/to/data --nxml_dir /path/to/nxml --test
```

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
