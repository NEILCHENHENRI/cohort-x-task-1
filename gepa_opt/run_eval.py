"""
CohortX Task 1 — Shipped-path evaluator (dev-time)
==================================================
Runs the OFFLINE inference path (predict_ollama.extract -> Ollama Qwen) over a
chosen split and scores it with evaluate.py. Used for both:
  - Phase 2 baseline   (current INSTRUCTION)   -> results/baseline.json
  - Phase 6 optimized  (pasted-in INSTRUCTION)  -> results/optimized.json
The only thing that differs between the two runs is predict_ollama.INSTRUCTION,
so this gives an honest, apples-to-apples before/after on the real shipped path.

Usage:
  python -m gepa_opt.run_eval --split holdout --output results/baseline.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from local_llm import predict_ollama
from gepa_opt.data_split import FIELDS, gold_dict, get_splits, load_sheet, nxml_path
from common.evaluate import score_row
from common.parser import NXMLParser


def normalize_conditions(pred: dict) -> dict:
    """Mirror predict_ollama.main: a conditions list -> comma string for scoring."""
    conds = pred.get("conditions", [])
    pred["conditions"] = ", ".join(conds) if isinstance(conds, list) else str(conds or "")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=["holdout", "val", "train"])
    ap.add_argument("--output", default="results/baseline.json")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N rows")
    ap.add_argument("--instruction_file", default=None,
                    help="swap in an optimized INSTRUCTION_PROSE (e.g. GEPA output) for this run")
    args = ap.parse_args()

    # Phase-6 override: evaluate an optimized instruction on the shipped path without
    # editing the source. (The literal paste into predict_ollama.py is the final ship step.)
    if args.instruction_file:
        prose = Path(args.instruction_file).read_text().strip()
        predict_ollama.INSTRUCTION_PROSE = prose
        predict_ollama.INSTRUCTION = f"{prose}\n\n{predict_ollama.OUTPUT_SCHEMA}"
        print(f"Using instruction from {args.instruction_file} "
              f"({len(prose)} chars, sha8 {hashlib.sha1(prose.encode()).hexdigest()[:8]})")

    splits = get_splits()
    pmcids = splits[args.split]
    if args.limit:
        pmcids = pmcids[:args.limit]
    gold_df = load_sheet("Train").set_index("pmcids")
    parser = NXMLParser()

    rows, scores = [], []
    tag = "optimized" if args.instruction_file else "baseline"
    for pid in tqdm(pmcids, desc=f"{tag}[{args.split}]"):
        parsed = parser.parse(str(nxml_path(pid)))
        pred = normalize_conditions(predict_ollama.extract(pid, parsed, args.model))
        gold = gold_dict(gold_df.loc[pid])
        s = score_row({f: pred.get(f, "") for f in FIELDS}, gold)
        scores.append(s)
        rows.append({"pmcids": pid, **{f: pred.get(f, "") for f in FIELDS}})

    sdf = pd.DataFrame(scores)
    means = {c: round(float(sdf[c].mean()), 4) for c in sdf.columns}

    result = {
        "split": args.split,
        "n": len(pmcids),
        "model": args.model,
        "instruction_sha8": hashlib.sha1(predict_ollama.INSTRUCTION.encode()).hexdigest()[:8],
        "mean_scores": means,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    # Also drop raw predictions next to the scores for inspection.
    pd.DataFrame(rows).to_csv(Path(args.output).with_suffix(".preds.csv"), index=False)

    print(f"\n=== {args.split} (n={len(pmcids)}, instr {result['instruction_sha8']}) ===")
    for c in ["conditions", "study_type", "sex", "minimum_age",
              "maximum_age", "eligibility_criteria", "overall"]:
        print(f"  {c:22s}: {means[c]:.4f}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
