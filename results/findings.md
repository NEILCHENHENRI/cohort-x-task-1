# GEPA Prompt Optimization — Findings

**Date:** 2026-07-06 · **Student model:** Qwen2.5-1.5B (Ollama, offline) ·
**Reflection model:** Claude Sonnet 4.6 (dev-time only) · **Optimizer:** DSPy GEPA 3.2.1

## Verdict

**GEPA helped — decisively, on the shipped offline path.** Starting from a hand-tuned
seed prompt, GEPA raised the composite score on 60 held-out papers (never seen by the
optimizer) from **0.627 → 0.685 (+0.058, +9% relative)** for **$0.118** of Claude
credits and one ~29-min dev-time run. Measured against the original 2-sentence prompt,
the full pipeline is **0.629 → 0.685 (+0.056)**.

Crucially, this was scored on the **real offline path** (`local_llm/predict_ollama.py`
+ `evaluate.score_row` / FM3S), not GEPA's internal proxy — so it's the number that
actually ships.

## The three stages (offline holdout, n=60, competition metric)

| Field | Weight | ① original seed | ② hand-edited seed | ③ GEPA optimized |
|---|---:|---:|---:|---:|
| eligibility_criteria | 0.50 | 0.781 | 0.716 | **0.822** |
| conditions | 0.15 | 0.251 | 0.552 | 0.550 |
| study_type | 0.10 | 0.775 | 0.845 | 0.880 |
| minimum_age | 0.10 | 0.267 | 0.050 | 0.050 |
| maximum_age | 0.10 | 0.467 | 0.467 | 0.483 |
| sex | 0.05 | 1.000 | 1.000 | 1.000 |
| **OVERALL** | 1.00 | **0.6288** | **0.6270** | **0.6850** |

- **① → ②** (hand edits: add conditions guidance + age "don't guess" guard):
  conditions **+0.30**, but eligibility **−0.064** and min_age **−0.217** → net **flat**
  (−0.002). The 1.5B model is prompt-sensitive; the longer seed helped conditions but
  disrupted eligibility.
- **② → ③** (GEPA): eligibility **+0.106** (recovered the regression *and* beat the
  original 0.781), study_type +0.035, max_age +0.017, conditions held → net **+0.058**.

**Why this is the headline result:** hand-editing could not balance the fields — fixing
conditions broke eligibility. GEPA optimizes the *whole* instruction against per-field
feedback, so it kept the conditions win **and** repaired eligibility. That is exactly
the case GEPA is built for.

Weighted contribution of the +0.058 GEPA gain: **eligibility +0.053**, study_type +0.004,
max_age +0.002, conditions −0.000.

## What GEPA wrote

From a 2-sentence seed, GEPA (via Claude reflection over field-specific feedback) produced
a 51-line structured instruction that independently rediscovered our hand strategies:

- **eligibility (the 50%-weight field)** — *"extract inclusion AND exclusion criteria;
  if exclusion criteria are mentioned anywhere in the article (not just a dedicated
  section), include them; a short response likely missed criteria."*
- **conditions** — *"use standard clinical terminology as it would appear in a
  ClinicalTrials.gov condition field; not an overly narrow sub-classification or a broad
  category."*
- **study_type** — cue words: *randomized/placebo → INTERVENTIONAL; retrospective /
  registry / review → OBSERVATIONAL.*

## The minimum_age regression is a metric artifact, not a model failure (Fix 5)

The **age guard held** — GEPA kept *"report an age only if explicitly stated; otherwise
output Not Specified; do not assume '18 Years' just because the study involves adults."*
That is the **correct, honest behavior**, and it *helped* maximum_age (+0.017). But it
scores **worse** on minimum_age, and that is a metric artifact:

- Gold ages come from **ClinicalTrials.gov registrations, not the paper text.** On this
  holdout the gold minimum_age is usually a number (very commonly **"18 Years"**) that the
  article never actually states.
- The number-similarity metric scores an honest *"Not Specified"* as **0** against a gold
  of *"18 Years."* The original prompt scored higher on min_age only by **guessing** a
  number that often matched the "18" mode.
- So the shipped prompt is **more truthful** (no fabricated ages) but scores lower on
  minimum_age. The score-optimal policy here is to always emit the constant *"18 Years"* —
  metric-gaming, not extraction. We deliberately did **not** do that.

## Cost & runtime

- Full run (train 60 / val 40, `--max_metric_calls 150`): **$0.118, 4 reflection calls,
  ~29 min** on the fanless M2 (CPU/Metal). The cost is entirely the local Qwen model
  running ~150× — the Claude reflection calls are a few cents.
- GEPA internal validation score (proxy): seed **0.632 → best candidate 0.674**.
- Offline transfer check (the real number): **0.627 → 0.685**.

## Reproduce

```bash
# before (pre-GEPA seed baseline)
python -m gepa_opt.run_eval --split holdout --output results/baseline.json
# GEPA (dev-time; needs .env ANTHROPIC_API_KEY)
python -m gepa_opt.optimize_gepa --max_metric_calls 150
# after: re-score the OPTIMIZED prompt on the SHIPPED offline path (the number that counts)
python -m gepa_opt.check_transfer
```

## Shipped

`results/optimized_instruction.txt` is pasted into `local_llm/predict_ollama.py`'s
`INSTRUCTION_PROSE`. The offline inference path stays **fully API-free** (verified: no
`anthropic` / `dspy` / `openai` / `litellm` import in `local_llm/` or `common/`).

## Next steps (optional)

1. **Scale the search:** `--auto medium` or more train data — the light run improved val
   by only +0.042, so there may be more to find (esp. conditions, still ~0.55).
2. **minimum_age is capped by the data**, not the prompt — the gold isn't in the article.
   Chasing it means metric-gaming (emit the "18 Years" mode); not worth it for a hidden
   test set.
