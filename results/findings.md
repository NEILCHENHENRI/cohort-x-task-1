# GEPA Prompt Optimization — Findings

**Date:** 2026-06-30 · **Student model:** Qwen2.5-1.5B (Ollama, offline) ·
**Reflection model:** Claude Sonnet 4.6 (dev-time only) · **Optimizer:** DSPy GEPA 3.2.1

## Verdict

**GEPA helped.** On 60 held-out papers (never seen by the optimizer), the composite
score rose **0.629 → 0.686 (+0.057, +9% relative)** — for **$0.13** of Claude credits
and one ~55-min dev-time run. The optimizer turned a 2-sentence seed prompt into a
71-line structured instruction that independently rediscovered every field strategy we
had identified by hand.

## Before / after (held-out, n=60, shipped offline path)

| Field | Weight | Baseline | Optimized | Δ |
|---|---:|---:|---:|---:|
| **eligibility_criteria** | 0.50 | 0.781 | **0.837** | **+0.056** |
| conditions | 0.15 | 0.251 | **0.540** | **+0.289** |
| study_type | 0.10 | 0.775 | 0.836 | +0.060 |
| minimum_age | 0.10 | 0.267 | 0.050 | **−0.217** |
| maximum_age | 0.10 | 0.467 | 0.483 | +0.017 |
| sex | 0.05 | 1.000 | 1.000 | 0.000 |
| **OVERALL** | 1.00 | **0.6288** | **0.6862** | **+0.0574** |

Weighted contribution to the +0.057 gain: conditions +0.043, eligibility +0.028,
study_type +0.006, max_age +0.002, **min_age −0.022**.

## What worked

- **conditions (+0.289, more than doubled)** — the single biggest driver. GEPA's rule
  *"output the SPECIFIC disease, not a symptom, imaging finding, broad category, or
  technology name"* (with concrete examples like `acute ischemic stroke`, not
  `nanoradiopharmaceuticals`) fixed the model's habit of naming methods/symptoms.
- **eligibility_criteria (+0.056, the 50%-weight field)** — GEPA added *"extract BOTH
  inclusion AND exclusion criteria; actively search methods sections — they're easy to
  overlook; a short response likely means you missed criteria."*
- **study_type (+0.060)** — GEPA listed the cue words (*randomized/placebo →
  INTERVENTIONAL; retrospective/registry/review → OBSERVATIONAL*).

## The minimum_age regression is a metric artifact, not a model failure

GEPA correctly taught the model **not to hallucinate ages** (*"'adults' does NOT mean
'18 Years' unless the number 18 is explicitly written → output Not Specified"*). That is
the **right behavior**, and it *helped* maximum_age (+0.017). But it **hurt**
minimum_age (−0.217). Why:

- The **gold ages come from ClinicalTrials.gov registrations, not the paper text.** The
  article usually never states the age, yet **293/416 (70%) of gold minimum_age =
  "18 Years."**
- The number-similarity metric scores an honest *"Not Specified"* as **0** against a gold
  of *"18 Years."* The baseline accidentally scored higher by **guessing** a number that
  often matched the "18" mode.
- So the optimized prompt is **more correct** (no fabrication) but scores **worse** on
  minimum_age. Under this metric, the score-optimal minimum_age policy is to always emit
  the mode *"18 Years"* — which is metric-gaming, not extraction.

This is the headline nuance: **GEPA optimized honestly and still won overall**; the one
loss comes from a field whose labels are not recoverable from the article and whose
metric rewards guessing a constant.

## Cost & runtime

- Smoke run (validate end-to-end): $0.042, 2 reflections, ~6 min.
- Full run (train 60 / val 40, 150 metric calls): **$0.088, 3 reflections, ~55 min** on
  the fanless M2 (CPU/Metal). A Colab GPU runtime would cut this to minutes.
- Validation score during optimization: seed 0.666 → best candidate **0.675**.

## Reproduce

```bash
python run_eval.py --split holdout --output results/baseline.json          # before
python optimize_gepa.py --max_metric_calls 150                             # GEPA (needs .env key)
python run_eval.py --split holdout \
    --instruction_file results/optimized_instruction.txt \
    --output results/optimized.json                                        # after
```
Or run `cohortx_gepa_colab.ipynb` end-to-end (GPU-aware).

## Next steps

1. **Recover minimum_age cheaply:** add a field rule to default minimum_age to the mode
   *"18 Years"* when unstated (metric-gaming but valid), or feed GEPA that signal. Could
   alone add ~+0.02–0.03 to the composite.
2. **Scale the search:** `auto="medium"` and/or more train data — the light run improved
   val by only +0.009, so there is likely more to find.
3. **Ship it:** paste `results/optimized_instruction.txt` into `predict_ollama.py`'s
   `INSTRUCTION_PROSE` (the offline path stays API-free).
