# CLAUDE.md — CohortX Task 1: GEPA Prompt Optimization

## What this project is

This repo is our entry for the **CohortX Task 1** challenge: read full-text
biomedical articles (PMC articles stored as `.nxml` XML files) and extract 6
structured fields per article:

- `conditions` (diseases studied)
- `study_type` (interventional / observational / etc.)
- `sex`
- `minimum_age`
- `maximum_age`
- `eligibility_criteria` (inclusion/exclusion rules, free text)

**My job (this task):** add **GEPA** prompt optimization to the current
generative pipeline and measure whether it beats the current hand-written
prompt. GEPA = Genetic-Pareto, the DSPy prompt optimizer
(https://github.com/gepa-ai/gepa). This is NOT "JEPA" (an unrelated vision
architecture). If a tool call or doc says JEPA, treat it as a typo for GEPA.

## Scoring (this drives every decision)

The composite score is a weighted blend across the 6 fields. Weights live in
`config.py` (`WEIGHTS`):

- `eligibility_criteria` = **0.50** (half the score — this is the priority)
- `conditions` 0.15, `study_type` 0.10, `minimum_age` 0.10, `maximum_age` 0.10, `sex` 0.05

Scoring methods (already implemented in `evaluate.py`):
- ages: extract numbers from prediction vs gold, compare the sets
- `conditions` + `study_type`: BioBERT embedding cosine similarity (semantic, no exact-match needed)
- `eligibility_criteria`: FM3S fuzzy sentence similarity (noun + verb + word-order overlap)

The metric rewards correct **meaning**, not exact strings.

## The codebase — what to touch and what NOT to touch

The repo has two historical solution paths. **We are only working on Path B.**

- **Path A (IGNORE for this task):** `train.py`, `models.py`, `predict.py` —
  a chain of fine-tuned specialist models (MiniLM ranker, SciFive, DistilBERT
  classifiers, NER, QA). This was week-1 experimentation. It has no prompts, so
  GEPA cannot optimize it. Do not modify these files.
- **Path B (THE LIVE SYSTEM — our target):** `predict_ollama.py` — one local
  generative model (Qwen2.5 1.5B via Ollama) extracts all 6 fields from a
  prompt. This is what we optimize.

Other relevant files:
- `config.py` — model names, `ELIGIBILITY_QUERY`, `WEIGHTS`, regex/constants. Reuse, do not rewrite.
- `parser.py` — `NXMLParser`, parses `.nxml` to a dict. Reuse as is.
- `evaluate.py` — competition metrics. This becomes our GEPA metric. Do not change its scoring logic.

## What GEPA actually optimizes here

Inside `predict_ollama.py`, the prompt has two parts:
1. **Article context** — MiniLM section filtering + assembly of title / abstract /
   keywords / relevant sections. **Keep this exactly as is.** It is the input.
2. **Instruction** — the "You are a biomedical information extraction system..."
   text that tells the model what to do. **This is the only thing GEPA rewrites.**

So the first refactor is to cleanly separate (1) from (2) so the instruction can
be swapped without touching the input-building logic.

## Hard rules (do not break)

1. **The submitted system must run fully offline, CPU, 16 GB RAM, no internet, no
   GPU.** GEPA does NOT break this. GEPA is a **dev-time** tool: it runs on my
   laptop with internet, calls a strong "reflection" model to rewrite prompts, and
   produces one optimized instruction string. The shipped `predict_ollama.py`
   then runs that fixed string on local Qwen with no API calls. Never add a
   runtime dependency on an external API to the inference path.
2. **Keep optimization separate from inference.** Do GEPA work in a new file
   (e.g. `optimize_gepa.py`). Leave `predict_ollama.py` runnable offline at all
   times. The final step is to paste the optimized instruction back into it.
3. **Baseline before optimizing.** We cannot claim GEPA helped without a
   before-number.

## Environment (my machine: MacBook Air M2)

- Apple Silicon (ARM), likely 8–16 GB RAM, fanless (may thermal-throttle on long runs).
- **Use the Ollama path only.** Do NOT use `predict_qwen.py` / bitsandbytes 8-bit —
  bitsandbytes needs CUDA and does not work on Mac. Ollama runs Qwen2.5 1.5B
  natively with Metal acceleration.
- **Student model (the one being optimized):** local Qwen2.5 1.5B via Ollama.
  In DSPy: `dspy.LM("ollama_chat/qwen2.5:1.5b", api_base="http://localhost:11434")`.
  Make sure `ollama serve` is running and `ollama pull qwen2.5:1.5b` is done.
- **Reflection model (rewrites prompts, dev-time only):** Claude via the Anthropic API.
  In DSPy: `dspy.LM("anthropic/claude-sonnet-4-6")` with `ANTHROPIC_API_KEY` set.
  (If that model string errors, confirm the current name at
  https://docs.claude.com/en/api/overview — model names change. Opus is an
  upgrade option if Sonnet's rewrites are weak.)
- **Budget:** ~$60 of Claude credits. The reflection model is called only a
  handful of times per run, so `auto="light"` is cheap (a few dollars). Start
  light, check spend before going bigger.
- Put the API key in a `.env` (gitignored). Never commit it.

## Implementation plan (phased)

**Phase 1 — Baseline.** Run `predict_ollama.py` on the Train sheet, score with
`evaluate.py`, and record composite + per-field numbers. Save to a file
(e.g. `results/baseline.json`). This is the bar to beat.

**Phase 2 — Wrap extraction as a DSPy program.** Create a DSPy `Signature` whose
inputs = the assembled article context and outputs = the 6 fields. Seed the
signature instruction (docstring) with the CURRENT prompt text from
`predict_ollama.py`. Use a simple `dspy.Predict` module first (ChainOfThought adds
tokens the 1.5B model may mishandle). Point the student LM at local Qwen. Verify
it produces parseable output. Reuse the existing article-context builder; do not
re-implement section filtering.

**Phase 3 — Wire the metric.** Wrap `evaluate.py`'s row scoring into a GEPA metric
with signature `metric(example, prediction, trace=None, pred_name=None, pred_trace=None)`
returning `dspy.Prediction(score=..., feedback=...)`. The `feedback` is GEPA's main
advantage: make it **field-specific** (e.g. "eligibility missed the exclusion
criteria", "conditions too generic", "max_age hallucinated a number not in text").
Confirm the metric's composite matches `evaluate.py` exactly on a few rows.

**Phase 4 — Run GEPA.** Split the 416 rows into train/val (keep val held-out for
honest evaluation). Configure `dspy.GEPA(metric=..., reflection_lm=<claude>,
auto="light", num_threads=2)`, then `optimizer.compile(program, trainset=..., valset=...)`.
Save the result with `optimized.save("results/gepa_optimized.json")`.
(Verify the exact GEPA constructor args against the installed DSPy version first;
the API evolves.)

**Phase 5 — Evaluate and decide.** Score the optimized program on held-out rows
vs the Phase 1 baseline. Then extract the optimized instruction string from the
saved JSON, paste it into `predict_ollama.py`, and confirm the offline path still
runs on Ollama alone (no Anthropic call). Write a short results note
(`results/findings.md`): baseline vs optimized, per-field deltas, and a verdict.
If GEPA does NOT beat baseline, that is a valid finding — document it clearly.

**Phase 6 — Scope up (only if light helped).** Try `auto="medium"`, or split
`eligibility_criteria` into its own focused DSPy module, or extend optimization to
`conditions` and `study_type`.

## Field strategy (where effort pays off)

- **`eligibility_criteria` (50%) is the priority.** It is a generation task where
  wording matters most — GEPA's strongest use case. Focus here.
- **`sex` is effectively unscored.** The number-based metric returns 1.0 for it
  every time (no numbers in MALE/FEMALE/ALL). Do not spend effort optimizing it.
- **`minimum_age` is mostly unrecoverable.** 344 of 416 training docs are
  "Not Specified" because the age is genuinely not in the article (gold labels
  came from ClinicalTrials.gov, not the paper). GEPA cannot extract absent data.
  Best case: teach the model to output a clean "Not Specified" instead of
  hallucinating. Do not chase recall here.

## Known gotchas

- **1.5B model + structured output.** Small models often break DSPy's expected
  output format. Make output parsing robust: prefer an explicit JSON instruction,
  and fall back to "Not Specified" per field on parse failure rather than crashing.
- **DSPy + Ollama backend friction.** Version/config mismatches are common. Smoke-test
  a single `dspy.Predict` call against Ollama before building anything else.
- **Cost creep.** Confirm `auto="light"` spend before scaling. Log token usage if possible.
- **Don't conflate dev-time and test-time.** The Claude reflection model is dev-time
  only. If you ever find the inference path importing the Anthropic SDK, that's a bug.

## Definition of done (first milestone)

1. `results/baseline.json` recorded.
2. DSPy program runs end-to-end on local Ollama Qwen and produces all 6 fields.
3. GEPA `auto="light"` run completes; optimized program saved.
4. Optimized vs baseline scored on held-out rows; deltas recorded.
5. Optimized instruction pasted back into `predict_ollama.py`; offline run confirmed
   (Ollama only, no API).
6. `results/findings.md` written with the verdict.
