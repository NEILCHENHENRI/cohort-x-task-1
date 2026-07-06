"""
CohortX Task 1 — DSPy program + GEPA metric (dev-time)
======================================================
Wraps the Path-B extraction as a DSPy program so GEPA can optimize the instruction
(the signature docstring, seeded from predict_ollama.INSTRUCTION_PROSE). The MiniLM
article-context builder (predict_ollama.build_article_context) is reused unchanged
as the fixed input. The metric reuses evaluate.score_row verbatim and adds
field-specific textual feedback — GEPA's main lever.

Design choices (see plan):
  - 6 native output fields + JSONAdapter: matches the shipped JSON format, so the
    optimized instruction transfers cleanly back into predict_ollama.
  - forward() is crash-proof: on any parse/LM failure every field -> "Not Specified".
  - The optimizable unit is the prose docstring; field descs are fixed scaffold and
    mirror predict_ollama's OUTPUT_SCHEMA hints so the seed is faithful.
"""

import ast
import pickle
from pathlib import Path

import dspy
from tqdm import tqdm

from local_llm import predict_ollama
from gepa_opt.data_split import FIELDS, get_splits, gold_dict, load_sheet, nxml_path
from common.evaluate import score_row
from common.parser import NXMLParser

CONTEXT_CACHE = Path("results/contexts.pkl")
NOT_SPEC      = "Not Specified"


# ---------------------------------------------------------------------------
# Student LM (local Qwen via Ollama) — deterministic, mirrors the shipped path
# ---------------------------------------------------------------------------

def get_student_lm(model: str = "qwen2.5:1.5b", max_tokens: int = 1024) -> dspy.LM:
    return dspy.LM(f"ollama_chat/{model}", api_base="http://localhost:11434",
                   temperature=0.0, max_tokens=max_tokens)


def configure_student(model: str = "qwen2.5:1.5b") -> None:
    # JSONAdapter keeps the output format close to the shipped JSON path.
    dspy.configure(lm=get_student_lm(model), adapter=dspy.JSONAdapter())


# ---------------------------------------------------------------------------
# Signature + program
# ---------------------------------------------------------------------------

class ExtractCriteria(dspy.Signature):
    article_context: str = dspy.InputField(
        desc="title / abstract / keywords / relevant sections of a biomedical article")
    conditions: str = dspy.OutputField(desc="primary medical conditions studied")
    study_type: str = dspy.OutputField(desc="INTERVENTIONAL or OBSERVATIONAL")
    sex: str = dspy.OutputField(desc="ALL or MALE or FEMALE")
    minimum_age: str = dspy.OutputField(desc="number followed by Years e.g. 18 Years")
    maximum_age: str = dspy.OutputField(desc="number followed by Years e.g. 65 Years")
    eligibility_criteria: str = dspy.OutputField(desc="full inclusion and exclusion criteria text")


# Seed the instruction with the shipped prose so the GEPA "before" == the baseline.
ExtractCriteria = ExtractCriteria.with_instructions(predict_ollama.INSTRUCTION_PROSE)


class CohortExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractCriteria)

    def forward(self, article_context: str) -> dspy.Prediction:
        try:
            return self.predict(article_context=article_context)
        except Exception:
            # 1.5B models break structured output sometimes — never crash GEPA.
            return dspy.Prediction(**{f: NOT_SPEC for f in FIELDS})


# ---------------------------------------------------------------------------
# Field normalization (DSPy output -> the clean strings score_row expects)
# ---------------------------------------------------------------------------

def norm_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):          # stringified list -> "a, b, c"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return ", ".join(str(v) for v in parsed)
        except Exception:
            pass
    return s


def pred_dict(pred: dspy.Prediction) -> dict:
    return {f: norm_field(getattr(pred, f, "")) for f in FIELDS}


# ---------------------------------------------------------------------------
# Examples (with a context cache so MiniLM filtering runs once per doc)
# ---------------------------------------------------------------------------

def build_contexts(pmcids: list) -> dict:
    cache = pickle.loads(CONTEXT_CACHE.read_bytes()) if CONTEXT_CACHE.exists() else {}
    missing = [p for p in pmcids if p not in cache]
    if missing:
        parser = NXMLParser()
        for pid in tqdm(missing, desc="Building contexts"):
            cache[pid] = predict_ollama.build_article_context(parser.parse(str(nxml_path(pid))))
        CONTEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CONTEXT_CACHE.write_bytes(pickle.dumps(cache))
    return {p: cache[p] for p in pmcids}


def build_examples(split: str) -> list:
    pmcids = get_splits()[split]
    contexts = build_contexts(pmcids)
    df = load_sheet("Train").set_index("pmcids")
    return [
        dspy.Example(pmcid=pid, article_context=contexts[pid],
                     **gold_dict(df.loc[pid])).with_inputs("article_context")
        for pid in pmcids
    ]


# ---------------------------------------------------------------------------
# GEPA metric — reuses evaluate.score_row, adds field-specific feedback
# ---------------------------------------------------------------------------

def make_feedback(p: dict, g: dict, s: dict) -> str:
    fb = ["Score %.2f (" % s["overall"]
          + ", ".join(f"{k}={s[k]:.2f}" for k in FIELDS) + ")."]

    # eligibility_criteria — 50% of the score, the priority
    if s["eligibility_criteria"] < 0.85:
        pe, ge = p["eligibility_criteria"], g["eligibility_criteria"]
        if not pe.strip() or pe.strip().upper() in ("NOT SPECIFIED", "NULL", "NONE"):
            note = ("you returned nothing — always extract the FULL inclusion AND "
                    "exclusion criteria text from the Participants/Methods/Eligibility section")
        elif "exclu" in ge.lower() and "exclu" not in pe.lower():
            note = "you captured inclusion but MISSED the exclusion criteria the gold lists"
        elif len(pe) < 0.5 * len(ge):
            note = ("your text is far shorter than gold — include every specific criterion "
                    "(diagnoses, lab thresholds, prior treatments), not a summary")
        else:
            note = ("tighten wording and coverage to match the gold's specific inclusion "
                    "AND exclusion items")
        fb.append("eligibility_criteria (HIGHEST weight 0.50): " + note + ".")

    if s["conditions"] < 0.7:
        fb.append(f"conditions: predicted '{p['conditions'][:80]}' vs gold "
                  f"'{g['conditions'][:80]}' — name the SPECIFIC disease(s) studied "
                  "(use title/keywords), not a broad category.")

    if s["study_type"] < 0.7:
        fb.append(f"study_type: predicted '{p['study_type']}' but gold '{g['study_type']}' "
                  "— INTERVENTIONAL only if a treatment/intervention is assigned, else OBSERVATIONAL.")

    for age in ("minimum_age", "maximum_age"):
        if s[age] < 1.0:
            pv, gv = str(p[age]), str(g[age])
            if not gv.strip() or gv.lower() == "nan":
                fb.append(f"{age}: gold has NO age but you output '{pv}'. When the article does "
                          f"not explicitly state the {age.split('_')[0]} age, output "
                          f"'{NOT_SPEC}' — never guess a number.")
            else:
                fb.append(f"{age}: gold is '{gv}' but you output '{pv}'. Only report an age "
                          f"explicitly stated in the text; otherwise '{NOT_SPEC}'.")
    return " ".join(fb)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    p = pred_dict(pred)
    g = {f: norm_field(getattr(gold, f, "")) for f in FIELDS}
    s = score_row(p, g)
    return dspy.Prediction(score=s["overall"], feedback=make_feedback(p, g, s))


# ---------------------------------------------------------------------------
# Smoke test (Phase 3/4 verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_student()
    exs = build_examples("holdout")[:2]
    prog = CohortExtractor()
    for ex in exs:
        pred = prog(article_context=ex.article_context)
        p, g = pred_dict(pred), {f: norm_field(getattr(ex, f, "")) for f in FIELDS}
        s = score_row(p, g)
        m = gepa_metric(ex, pred)
        # metric score MUST equal evaluate.score_row overall
        assert abs(m.score - s["overall"]) < 1e-9, (m.score, s["overall"])
        print(f"\nPMC{ex.pmcid}  metric.score={m.score:.4f}  (== score_row overall ✓)")
        print("  pred study_type:", repr(p["study_type"]), "| gold:", repr(g["study_type"]))
        print("  feedback:", m.feedback[:300])
