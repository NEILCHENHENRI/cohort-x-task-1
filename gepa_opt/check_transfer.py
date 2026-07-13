"""
CohortX Task 1 — Transfer check (dev-time)
==========================================
Re-score the OFFLINE path (local_llm.predict_ollama) using the GEPA-optimized
instruction, on the SAME holdout used for results/baseline.json. This is the
number that COUNTS.

Why this is needed: GEPA scores the instruction while DSPy's JSONAdapter
formats/parses the output. The shipped predict_ollama.py uses a different wrapper
(OUTPUT_SCHEMA + parse_json_response). So GEPA's val score is a proxy, not the
deliverable — after paste-back you must re-score the real offline path here.

Run AFTER optimize_gepa.py has written results/optimized_instruction.txt:

  python -m gepa_opt.check_transfer [--model qwen2.5:1.5b] [--limit N]

Scoring note: this uses evaluate.score_row — full FM3S for eligibility_criteria,
the SAME metric that produced baseline.json — NOT evaluate_fast (BioBERT cosine),
so the before/after comparison is apples-to-apples.

Rule: if the offline `overall` does not beat baseline.json, GEPA did not help,
no matter what its internal val score said.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from local_llm import predict_ollama as po          # the real offline path
from gepa_opt.data_split import FIELDS, get_splits, gold_dict, load_sheet, nxml_path
from common.evaluate import score_row               # matches baseline.json
from common.parser import NXMLParser

OPT_FILE     = Path("results/optimized_instruction.txt")
HOLDOUT_FILE = Path("results/holdout_ids.json")
BASELINE     = Path("results/baseline.json")
OUT_JSON     = Path("results/transfer_check.json")


def load_holdout_ids() -> list:
    """The canonical holdout (same docs as baseline.json). Prefer the explicit
    export; fall back to results/splits.json (the single source of truth)."""
    if HOLDOUT_FILE.exists():
        return [str(x) for x in json.loads(HOLDOUT_FILE.read_text())]
    return get_splits()["holdout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N docs")
    args = ap.parse_args()

    # 1. swap in the optimized instruction, keep the offline OUTPUT_SCHEMA scaffold.
    #    build_prompt() reads the module-level po.INSTRUCTION.
    optimized = OPT_FILE.read_text().strip()
    po.INSTRUCTION_PROSE = optimized
    po.INSTRUCTION = f"{optimized}\n\n{po.OUTPUT_SCHEMA}"

    # 2. use the EXACT same docs as baseline.json
    holdout_ids = load_holdout_ids()
    if args.limit:
        holdout_ids = holdout_ids[:args.limit]
    gold_df = load_sheet("Train").set_index("pmcids")
    parser = NXMLParser()

    # 3. run the offline extractor doc-by-doc, score with the competition metric
    scores, rows = [], []
    for pid in tqdm(holdout_ids, desc=f"transfer[holdout,{args.model}]"):
        parsed = parser.parse(str(nxml_path(pid)))
        pred = po.extract(pid, parsed, args.model)
        conds = pred.get("conditions", [])
        pred["conditions"] = ", ".join(conds) if isinstance(conds, list) else str(conds or "")
        gold = gold_dict(gold_df.loc[pid])
        s = score_row({f: pred.get(f, "") for f in FIELDS}, gold)
        scores.append(s)
        rows.append({"pmcids": pid, **{f: pred.get(f, "") for f in FIELDS}})

    sdf = pd.DataFrame(scores)
    opt = {c: float(sdf[c].mean()) for c in sdf.columns}

    # 4. score with the competition metric and compare to baseline
    base = json.loads(BASELINE.read_text())["mean_scores"]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"split": "holdout", "n": len(holdout_ids), "model": args.model,
         "source": "offline predict_ollama + optimized_instruction.txt",
         "mean_scores": {k: round(v, 4) for k, v in opt.items()}}, indent=2))
    pd.DataFrame(rows).to_csv(OUT_JSON.with_suffix(".preds.csv"), index=False)

    print(f"\n=== offline transfer check (n={len(holdout_ids)}) ===")
    print("field                  baseline   optimized     delta")
    for f in ["conditions", "study_type", "sex", "minimum_age", "maximum_age",
              "eligibility_criteria", "overall"]:
        b = float(base.get(f, 0.0))
        o = float(opt.get(f, 0.0))
        print(f"{f:22s} {b:8.4f}  {o:9.4f}   {o - b:+.4f}")

    verdict = ("GEPA HELPS on the shipped offline path"
               if opt.get("overall", 0.0) > base.get("overall", 0.0)
               else "NO GAIN on the shipped path (GEPA's internal val score was only a proxy)")
    print(f"\nVerdict: {verdict}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
