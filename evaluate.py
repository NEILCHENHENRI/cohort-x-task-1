"""
CohortX Task 1 — Evaluation Metrics
=====================================
Implements the three competition metrics:

  1. Number Similarity     — sex, minimum_age, maximum_age
  2. Semantic Similarity   — conditions, study_type
                             (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)
  3. FM3S                  — eligibility_criteria
                             (noun sim + verb sim + CWO, λ=0.6)

Usage:
  from evaluate import evaluate_fast, evaluate, score_row
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from word2number import w2n

from config import ALPHA, BIOBERT_EVAL_NAME, LAM, STOP_VERBS, WEIGHTS

# ---------------------------------------------------------------------------
# Lazy model loading — only instantiated when first used
# ---------------------------------------------------------------------------

_biobert = None
_nlp     = None


def _get_biobert():
    global _biobert
    if _biobert is None:
        from sentence_transformers import SentenceTransformer
        _biobert = SentenceTransformer(BIOBERT_EVAL_NAME)
    return _biobert


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# 1. Number Similarity
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> set:
    text = str(text) if text else ""
    nums = set()
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass
    words = text.lower().split()
    for i in range(len(words)):
        for j in range(i + 1, min(i + 5, len(words)) + 1):
            try:
                nums.add(float(w2n.word_to_num(" ".join(words[i:j]))))
            except Exception:
                pass
    return nums


def number_similarity(pred: str, ref: str) -> float:
    n_pred = extract_numbers(pred)
    n_ref  = extract_numbers(ref)
    if not n_pred and not n_ref:
        return 1.0
    if not n_pred or not n_ref:
        return 0.0
    return len(n_pred & n_ref) / max(len(n_pred), len(n_ref))


# ---------------------------------------------------------------------------
# 2. Semantic Similarity
# ---------------------------------------------------------------------------

def semantic_similarity(pred: str, ref: str) -> float:
    pred = str(pred).strip() if pred else ""
    ref  = str(ref).strip()  if ref  else ""
    if not pred or not ref:
        return 0.0
    if pred.upper() in ("NOT SPECIFIED", "NAN") or \
       ref.upper()  in ("NOT SPECIFIED", "NAN"):
        return 0.0
    embs = _get_biobert().encode([pred, ref])
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


# ---------------------------------------------------------------------------
# 3. FM3S
# ---------------------------------------------------------------------------

def _lin_sim(w1: str, w2: str, pos: str) -> float:
    try:
        from nltk.corpus import wordnet as wn, wordnet_ic
        ic     = wordnet_ic.ic("ic-brown.dat")
        wn_pos = wn.NOUN if pos == "NOUN" else wn.VERB
        s1     = wn.synsets(w1, pos=wn_pos)
        s2     = wn.synsets(w2, pos=wn_pos)
        if not s1 or not s2:
            return 0.0
        best = 0.0
        for a in s1[:3]:
            for b in s2[:3]:
                try:
                    v = a.lin_similarity(b, ic)
                    if v and v > best:
                        best = v
                except Exception:
                    pass
        return best
    except Exception:
        return float(w1.lower() == w2.lower())


def fm3s(pred: str, ref: str) -> float:
    pred = str(pred).strip() if pred else ""
    ref  = str(ref).strip()  if ref  else ""
    if not pred or not ref:
        return 0.0
    if pred.upper() in ("NOT SPECIFIED", "NAN") or \
       ref.upper()  in ("NOT SPECIFIED", "NAN"):
        return 0.0

    nlp  = _get_nlp()
    doc1 = nlp(pred[:1000])
    doc2 = nlp(ref[:1000])

    # Noun similarity
    nouns1 = [t.lemma_.lower() for t in doc1 if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
    nouns2 = [t.lemma_.lower() for t in doc2 if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
    noun_sim = float(np.mean([
        max((_lin_sim(n, m, "NOUN") for m in nouns2), default=0.0)
        for n in nouns1
    ])) if nouns1 and nouns2 else 0.0

    # Verb similarity (tense matching)
    verbs1 = [(t.lemma_.lower(), t.tag_) for t in doc1
              if t.pos_ == "VERB" and t.lemma_.lower() not in STOP_VERBS]
    verbs2 = [(t.lemma_.lower(), t.tag_) for t in doc2
              if t.pos_ == "VERB" and t.lemma_.lower() not in STOP_VERBS]
    if verbs1 and verbs2:
        scores = []
        for v1, tag in verbs1:
            pool = [(v2, t2) for v2, t2 in verbs2 if t2 == tag] or verbs2
            scores.append(max((_lin_sim(v1, v2, "VERB") for v2, _ in pool), default=0.0))
        verb_sim = float(np.mean(scores))
    else:
        verb_sim = 0.0

    # CWO
    w1 = [t.lemma_.lower() for t in doc1 if not t.is_stop and t.is_alpha]
    w2 = [t.lemma_.lower() for t in doc2 if not t.is_stop and t.is_alpha]
    if w1 and w2:
        common         = set(w1) & set(w2)
        simple_cwo     = len(common) / max(len(set(w1)), len(set(w2)))
        bg1            = [(w1[i], w1[i + 1]) for i in range(len(w1) - 1)]
        bg2            = [(w2[i], w2[i + 1]) for i in range(len(w2) - 1)]
        successive_cwo = len(set(bg1) & set(bg2)) / max(len(bg1), len(bg2), 1)
        cwo            = ALPHA * simple_cwo + (1 - ALPHA) * successive_cwo
    else:
        cwo = 0.0

    x     = noun_sim ** LAM
    y     = (verb_sim + cwo) ** (LAM - LAM ** 2)
    score = (x + y) / (1 + x) if (1 + x) > 0 else 0.0
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_row(pred: dict, gold: dict) -> dict:
    scores = {
        "sex":                  number_similarity(pred.get("sex", ""),
                                                  gold.get("sex", "")),
        "minimum_age":          number_similarity(pred.get("minimum_age", ""),
                                                  gold.get("minimum_age", "")),
        "maximum_age":          number_similarity(pred.get("maximum_age", ""),
                                                  gold.get("maximum_age", "")),
        "conditions":           semantic_similarity(str(pred.get("conditions", "")),
                                                    str(gold.get("conditions", ""))),
        "study_type":           semantic_similarity(pred.get("study_type", ""),
                                                    gold.get("study_type", "")),
        "eligibility_criteria": fm3s(pred.get("eligibility_criteria", ""),
                                     gold.get("eligibility_criteria", "")),
    }
    scores["overall"] = sum(WEIGHTS[f] * scores[f] for f in WEIGHTS)
    return scores


# ---------------------------------------------------------------------------
# Evaluate (full FM3S, slow)
# ---------------------------------------------------------------------------

def evaluate(preds_df: pd.DataFrame, gold_df: pd.DataFrame,
             n: int = None) -> pd.DataFrame:
    merged = preds_df.merge(gold_df, on="pmcids", suffixes=("_pred", "_gold"))
    if n:
        merged = merged.sample(n, random_state=42)

    all_scores = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Scoring"):
        pred = {f: row.get(f"{f}_pred", "") for f in WEIGHTS}
        gold = {f: row.get(f"{f}_gold", "") for f in WEIGHTS}
        all_scores.append(score_row(pred, gold))

    scores_df = pd.DataFrame(all_scores)
    print("\n=== Mean Scores ===")
    for col in list(WEIGHTS.keys()) + ["overall"]:
        print(f"  {col:25s}: {scores_df[col].mean():.4f}")
    return scores_df


# ---------------------------------------------------------------------------
# Evaluate fast (batched BioBERT, skips FM3S)
# ---------------------------------------------------------------------------

def evaluate_fast(preds_df: pd.DataFrame, gold_df: pd.DataFrame,
                  n: int = None) -> pd.DataFrame:
    merged = preds_df.merge(gold_df, on="pmcids", suffixes=("_pred", "_gold"))
    if n:
        merged = merged.sample(n, random_state=42)

    fields     = ["conditions", "study_type", "eligibility_criteria"]
    pred_texts = {f: merged[f"{f}_pred"].fillna("").astype(str).tolist() for f in fields}
    gold_texts = {f: merged[f"{f}_gold"].fillna("").astype(str).tolist() for f in fields}

    print("Encoding with BioBERT...")
    biobert    = _get_biobert()
    pred_embs  = {f: biobert.encode(pred_texts[f], show_progress_bar=True) for f in fields}
    gold_embs  = {f: biobert.encode(gold_texts[f], show_progress_bar=True) for f in fields}

    rows = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        s   = {}
        for f in fields:
            s[f] = float(cosine_similarity([pred_embs[f][i]], [gold_embs[f][i]])[0][0])
        s["sex"]         = number_similarity(row.get("sex_pred", ""),
                                              row.get("sex_gold", ""))
        s["minimum_age"] = number_similarity(row.get("minimum_age_pred", ""),
                                              row.get("minimum_age_gold", ""))
        s["maximum_age"] = number_similarity(row.get("maximum_age_pred", ""),
                                              row.get("maximum_age_gold", ""))
        s["overall"]     = sum(WEIGHTS[f] * s[f] for f in WEIGHTS)
        rows.append(s)

    scores_df = pd.DataFrame(rows)
    print("\n=== Mean Scores ===")
    for col in list(WEIGHTS.keys()) + ["overall"]:
        print(f"  {col:25s}: {scores_df[col].mean():.4f}")
    return scores_df
