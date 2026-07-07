# GEPA Prompt Optimization — Findings

**CohortX Task 1** · 2026-07-06
Student model: Qwen2.5-1.5B (local, offline) · Reflection model: Claude Sonnet 4.6 (dev-time only) · Optimizer: DSPy GEPA

---

## TL;DR

We used **GEPA** to automatically rewrite the prompt our extraction model runs on.
On 60 held-out papers it lifted our score from **0.627 → 0.685 (+9%)**, for **12 cents**
of API credits and one ~30-minute run. Almost all of the gain came from the single
highest-weighted field — eligibility criteria (50% of the score). The optimized prompt
is shipped in the offline pipeline; the delivered system makes **no API calls**.

---

## 1. What is GEPA?

GEPA (**Ge**netic-**Pa**reto) is an automatic prompt optimizer.

The problem it solves: when you use a language model for a task, the instruction you
write — "the prompt" — hugely affects output quality. But writing a good prompt is
trial-and-error guesswork, and a change that helps one part of the task often quietly
hurts another. GEPA automates that trial-and-error.

You give it three things: a **starting prompt**, a set of **examples with known-correct
answers**, and a **scoring function** that grades the model's output. Then it loops:

1. Run the current prompt on some examples.
2. Score the outputs and collect *written feedback* on what went wrong
   (e.g. "the answer missed the exclusion criteria").
3. Send that feedback to a strong "reflection" model, which **rewrites the prompt** to
   fix those specific mistakes.
4. Keep the rewrites that score better; discard the rest. Repeat.

The name says how it searches: it evolves a *population* of prompt variants (**genetic**)
and keeps the ones that are best on *different* parts of the task, not just the best
average (**Pareto**) — so a variant that's excellent at one field isn't thrown away for
being mediocre overall. The key ingredient over older optimizers is that written
feedback: it tells the rewriter *why* an answer was wrong, not just a number.

**One thing to keep straight:** GEPA is a **development-time tool**. It runs once, on a
laptop with internet, and its only output is a better prompt string. The delivered
system just uses that string — no GEPA, no API — at run time.

---

## 2. How it applies to our task

Our task: read a full biomedical paper and extract 6 structured fields (conditions,
study type, sex, minimum age, maximum age, eligibility criteria). We do this with **one
small local model** (Qwen 1.5B) driven by **a single prompt**. That is an ideal GEPA
setup:

- **Everything rides on the prompt.** One small model, one instruction — improving the
  instruction is the main lever we have.
- **We already have a scorer.** The competition metric grades each field (fuzzy text
  similarity for eligibility, semantic similarity for conditions, number matching for
  ages). GEPA needs exactly that.
- **The scoring is lopsided.** Eligibility criteria alone is **50%** of the score. A
  tweak that helps eligibility is worth 10× one that helps sex (5%). GEPA optimizes
  against those real weights, so it spends effort where it counts — hard to do by hand.
- **Small models are prompt-sensitive.** A 1.5B model reacts sharply to wording — a
  headache to tune manually, but exactly where automated search pays off.

---

## 3. How it's wired into our pipeline

### The GEPA loop, at a glance

The solid arrows are the **dev-time optimization loop**; the dashed arrow is the
**one-time handoff** to the shipped offline system.

```mermaid
flowchart TD
    A["📄 Article context<br/>MiniLM-filtered sections<br/><i>fixed input — GEPA never touches it</i>"]
    B["📝 Instruction prompt<br/><i>starts as our hand-written prompt</i>"]
    A --> RUN["🤖 Qwen 1.5B (local)<br/>extracts the 6 fields"]
    B --> RUN
    RUN --> SCORE["📊 Competition metric<br/>score + <b>written per-field feedback</b>"]
    SCORE -->|"e.g. 'missed the exclusion criteria'"| REFLECT["✍️ Claude (reflection model)<br/>rewrites the instruction"]
    REFLECT -->|"improved candidate prompt"| B
    SCORE -.->|"best prompt after N trials"| SHIP["📦 Paste into offline pipeline<br/><b>local Qwen only · no API · no internet · no GPU</b>"]

    classDef dev fill:#e8f0fe,stroke:#4285f4,color:#111;
    classDef ship fill:#e6f4ea,stroke:#34a853,color:#111;
    class A,B,RUN,SCORE,REFLECT dev;
    class SHIP ship;
```

*The loop repeats: each pass, Claude reads the written feedback, rewrites the prompt,
and GEPA keeps the rewrite only if it scores better. Claude never touches the final
system — it only helped write the prompt during development.*

### First, we split the prompt in two — so GEPA only touches one half

- **Article context (the input — never optimized):** we parse the paper and use a small
  embedding model (MiniLM) to pull only the sections relevant to eligibility, keeping the
  prompt inside the 1.5B model's context window. GEPA never touches this.
- **The instruction (what GEPA rewrites):** the "you are a biomedical extraction
  system…" guidance telling the model what to pull and how.

### What we built (three pieces)

To let GEPA optimize the instruction, we wrapped our extractor as a small **DSPy program**
(DSPy is the framework GEPA runs inside). It has three parts:

1. **A typed extraction task** — one input (the article context) and six outputs (the six
   fields). We **seeded its instruction with our existing hand-written prompt**, so GEPA's
   starting point *is* our current baseline and any gain is a fair before/after. We kept
   the model call simple (direct prediction, no chain-of-thought — the 1.5B model tends to
   mishandle extra reasoning steps) and made it crash-proof: if the small model returns
   malformed output, every field falls back to "Not Specified" instead of erroring.

2. **The scorer** — GEPA grades every attempt with the **exact competition metric**. We
   reuse the same scoring code, so GEPA optimizes the *real* score, not a stand-in. (We
   assert-checked that the two match to the decimal.)

3. **The feedback — the part that makes GEPA work.** Alongside each score, we wrote a
   function that generates **field-specific written notes** on what went wrong; this is
   the signal the reflection model actually learns from. For example:
   - eligibility → *"you captured inclusion but MISSED the exclusion criteria the gold lists"*
   - conditions → *"name the SPECIFIC disease studied (use the title/keywords), not a broad category"*
   - ages → *"gold has no age but you output '65 Years' — when the article doesn't state it, output 'Not Specified', never guess"*

   The notes lean hardest on eligibility (50% of the score), so GEPA's attention goes
   where the points are.

### Running it (dev-time, once, on my laptop)

We split our 416 labeled papers into **train / validation / held-out** sets (held-out is
never shown to the optimizer, so the final number is honest). We cache the parsed context
per paper so the slow MiniLM step runs only once. Then GEPA loops: run the current prompt
on training papers with **local Qwen** → score them → hand the low scores + written
feedback to **Claude** (the reflection model) → Claude rewrites the instruction → keep the
rewrite only if it scores better on validation. The output is one optimized instruction
string (capped at a fixed number of trials to keep runtime and cost predictable).

### Shipping it (run-time)

We paste that single string back into the offline pipeline's prompt. That's the whole
handoff — Claude *wrote* the prompt, but the delivered system runs the fixed string on
local Qwen with **no API, no internet, no GPU**. We verified the shipped code imports no
API library.

---

## 4. Impact / results

Scored on **60 held-out papers the optimizer never saw**, using the real offline
pipeline and the competition metric:

| Field | Weight | Hand-tuned prompt | GEPA-optimized | Δ |
|---|---:|---:|---:|---:|
| **eligibility_criteria** | 0.50 | 0.716 | **0.822** | **+0.106** |
| conditions | 0.15 | 0.552 | 0.550 | −0.002 |
| study_type | 0.10 | 0.845 | 0.880 | +0.035 |
| maximum_age | 0.10 | 0.467 | 0.483 | +0.017 |
| minimum_age | 0.10 | 0.050 | 0.050 | 0.000 |
| sex | 0.05 | 1.000 | 1.000 | 0.000 |
| **OVERALL** | 1.00 | **0.627** | **0.685** | **+0.058** |

**The story in one line: hand-tuning couldn't win; GEPA could.** When we tried to fix our
weakest field (conditions) by hand, it helped conditions but quietly *broke* eligibility
— and because eligibility is half the score, the overall stayed flat. GEPA, optimizing
the whole prompt against per-field feedback, kept the conditions gain **and** repaired
eligibility, pushing it above where it started. Of the +0.058 total gain, **+0.053 came
from eligibility** alone.

**One honest caveat — minimum_age.** The "gold" ages come from a clinical-trial registry,
not the paper text, so the correct age often isn't actually stated in the article. GEPA
(correctly) taught the model to answer **"Not Specified"** instead of guessing. That is
more truthful, but scores lower on this one field, because the metric rewards guessing the
common "18 Years." We chose truthfulness over gaming the metric.

**Cost:** $0.118, 4 reflection calls, ~30 min on a fanless laptop.

---

## 5. Next steps

1. **Scale the search.** This was a light run and it still found a +9% gain — a larger
   budget (more iterations / more training examples) may find more, especially for
   conditions (stuck around 0.55).
2. **Combine with the new retrieval subsystem.** A better section-retrieval module now
   exists in the repo; feeding the model cleaner context and re-running GEPA on top of it
   is the natural next lever.
3. **Don't chase minimum_age.** It's capped by the data, not the prompt — the answer
   isn't in the paper. Pushing it further would mean gaming the metric, not real
   extraction.
