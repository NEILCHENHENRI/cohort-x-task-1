"""
CohortX Task 1 — Ollama Inference (Qwen 1.5B)
===============================================
Uses a local Ollama server with MiniLM section filtering to keep
prompts within the 1.5B model's context window.

Usage:
  # Start Ollama first:
  ollama serve
  ollama pull qwen2.5:1.5b

  python predict_ollama.py \\
    --data_dir  ~/Downloads/cohort-x-task-1 \\
    --nxml_dir  ~/Downloads/cohort-x-task-1/PMC_NXML_Archives \\
    [--model    qwen2.5:1.5b] \\
    [--test]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import ELIGIBILITY_QUERY
from parser import NXMLParser

# ---------------------------------------------------------------------------
# Section filter (MiniLM)
# ---------------------------------------------------------------------------

THRESHOLD         = 0.20
MAX_CHARS_PER_SEC = 600
MAX_TOTAL_CHARS   = 5000

_ranker    = None
_query_emb = None


def _get_ranker():
    global _ranker, _query_emb
    if _ranker is None:
        from sentence_transformers import SentenceTransformer
        _ranker    = SentenceTransformer("all-MiniLM-L6-v2")
        _query_emb = _ranker.encode(ELIGIBILITY_QUERY, convert_to_tensor=True)
    return _ranker, _query_emb


def get_relevant_sections(parsed: dict) -> str:
    from sentence_transformers import util
    ranker, query_emb = _get_ranker()
    scored = []
    for sec in parsed.get("sections", []):
        title_emb = ranker.encode(sec["title"], convert_to_tensor=True)
        text_emb  = ranker.encode(sec["text"][:200], convert_to_tensor=True)
        score     = max(
            float(util.cos_sim(query_emb, title_emb).item()),
            float(util.cos_sim(query_emb, text_emb).item()),
        )
        if score >= THRESHOLD:
            scored.append((score, sec))

    scored.sort(key=lambda x: x[0], reverse=True)

    body = ""
    for _, sec in scored:
        chunk = (f"\n{sec['title']}\n" if sec["title"] else "") + sec["text"][:MAX_CHARS_PER_SEC] + "\n"
        if len(body) + len(chunk) > MAX_TOTAL_CHARS:
            break
        body += chunk
    return body


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The INPUT half: MiniLM-filtered article context. This is fixed — GEPA never
# touches it. Returns ONLY the article text (no instruction, no "Article:" label).
def build_article_context(parsed: dict) -> str:
    title    = parsed.get("title", "")
    abstract = parsed.get("abstract", "")[:600]
    keywords = ", ".join(parsed.get("keywords", []))
    body     = get_relevant_sections(parsed)
    return f"Title: {title}\n\nAbstract: {abstract}\n\nKeywords: {keywords}\n\nRelevant sections:\n{body}"


# The INSTRUCTION half: the only thing GEPA rewrites. Split in two so paste-back
# only swaps the prose:
#   - INSTRUCTION_PROSE: natural-language guidance GEPA optimizes (and that seeds
#     the DSPy signature docstring). <<< Phase 6 replaces THIS with GEPA's output.
#   - OUTPUT_SCHEMA: the fixed JSON-format scaffold for this Ollama path (the DSPy
#     program uses its own adapter for formatting, so the scaffold is path-specific).
INSTRUCTION_PROSE = (
    "You are a biomedical information extraction system. "
    "Read the following article and extract the requested fields."
)

OUTPUT_SCHEMA = (
    "Return ONLY valid JSON with these exact fields, no other text:\n"
    "{\n"
    '  "conditions": ["primary medical conditions studied"],\n'
    '  "study_type": "INTERVENTIONAL or OBSERVATIONAL",\n'
    '  "sex": "ALL or MALE or FEMALE",\n'
    '  "minimum_age": "number followed by Years e.g. 18 Years",\n'
    '  "maximum_age": "number followed by Years e.g. 65 Years",\n'
    '  "eligibility_criteria": "full inclusion and exclusion criteria text"\n'
    "}\n"
    "If a field cannot be determined, use null."
)

INSTRUCTION = f"{INSTRUCTION_PROSE}\n\n{OUTPUT_SCHEMA}"


def build_prompt(parsed: dict) -> str:
    return f"{INSTRUCTION}\n\nArticle:\n{build_article_context(parsed)}"


def parse_json_response(raw: str) -> dict:
    """Lenient JSON extraction shared by the Ollama path and the DSPy program."""
    raw = raw.strip()
    raw = re.sub(r"[\x00-\x1f\x7f](?<![\n\t])", "", raw)
    raw = re.sub(r"\n\s*", " ", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(match.group()) if match else {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def extract(pid: str, parsed: dict, model: str) -> dict:
    import ollama
    if not parsed:
        return {}
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt(parsed)}],
            options={"temperature": 0.0, "num_predict": 512},
        )
        return parse_json_response(response["message"]["content"])
    except Exception as e:
        print(f"\nError on PMC{pid}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--nxml_dir", required=True)
    p.add_argument("--model",    default="qwen2.5:1.5b")
    p.add_argument("--output",   default="submission_ollama.xlsx")
    p.add_argument("--sheet",    default="Train")
    p.add_argument("--test",     action="store_true",
                   help="Run on 3 random samples only")
    return p.parse_args()


def main():
    import random
    args     = parse_args()
    data_dir = Path(args.data_dir)
    save_path = data_dir / args.output

    df     = pd.read_excel(data_dir / "Task_1.xlsx", sheet_name=args.sheet)
    pmcids = df["pmcids"].astype(str).str.strip().tolist()
    if args.test:
        pmcids = random.sample(pmcids, 3)

    parser     = NXMLParser()
    parsed_map = {}
    for pid in tqdm(pmcids, desc="Parsing"):
        path = Path(args.nxml_dir) / f"PMC{pid}.nxml"
        if path.exists():
            parsed_map[pid] = parser.parse(str(path))

    # Resume support
    if not args.test and save_path.exists():
        existing_df = pd.read_excel(save_path)
        done_pids   = set(existing_df["pmcids"].astype(str).str.strip())
        rows        = existing_df.to_dict("records")
        pmcids      = [p for p in pmcids if p not in done_pids]
        print(f"Resuming: {len(done_pids)} done, {len(pmcids)} remaining.")
    else:
        rows = []

    pbar = tqdm(pmcids, desc=f"Extracting [{args.model}]")
    for pid in pbar:
        pred  = extract(pid, parsed_map.get(pid, {}), args.model)
        conds = pred.get("conditions", [])
        pred["conditions"] = ", ".join(conds) if isinstance(conds, list) else str(conds or "")
        rows.append({"pmcids": pid, **pred})

        if args.test:
            gold_row = df[df["pmcids"].astype(str).str.strip() == pid]
            print(f"\nPMC{pid}")
            for k, v in pred.items():
                print(f"  {k:25s}: {str(v)[:120]}")
            if not gold_row.empty:
                g = gold_row.iloc[0]
                print(f"  --- GOLD ---")
                for field in ["conditions", "study_type", "sex", "minimum_age", "maximum_age"]:
                    print(f"  {field:25s}: {g[field]}")
        elif len(rows) % 50 == 0:
            pd.DataFrame(rows).to_excel(save_path, index=False)
            pbar.set_postfix({"saved": len(rows)})

    if not args.test:
        pd.DataFrame(rows).to_excel(save_path, index=False)
        print(f"Saved {len(rows)} rows → {save_path}")


if __name__ == "__main__":
    main()
