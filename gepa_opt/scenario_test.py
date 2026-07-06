"""
CohortX Task 1 — Scenario / qualitative test (dev-time)
=======================================================
Aggregate scores (run_eval.py) tell you *how much* but not *what* the model gets
right or wrong. This prints a side-by-side prediction-vs-gold view per paper, with
the per-field score, so you can eyeball whether the optimized system behaves well
(e.g. does it actually stop hallucinating ages? does it capture exclusion criteria?).

  python -m gepa_opt.scenario_test                       # optimized instruction, 5 holdout papers
  python -m gepa_opt.scenario_test --baseline            # current hand-written prompt
  python -m gepa_opt.scenario_test --n 8 --split val

Reads the optimized instruction from results/optimized_instruction.txt (Phase 5 output)
unless --baseline is given.
"""

import argparse
import textwrap
from pathlib import Path

from local_llm import predict_ollama
from gepa_opt.data_split import FIELDS, get_splits, gold_dict, load_sheet, nxml_path
from common.evaluate import score_row
from common.parser import NXMLParser

OPT_FILE = Path("results/optimized_instruction.txt")


def apply_optimized_instruction() -> str:
    prose = OPT_FILE.read_text().strip()
    predict_ollama.INSTRUCTION_PROSE = prose
    predict_ollama.INSTRUCTION = f"{prose}\n\n{predict_ollama.OUTPUT_SCHEMA}"
    return "OPTIMIZED"


def show(label, value, width=110):
    value = str(value).replace("\n", " ")
    print(f"    {label:9s}: {textwrap.shorten(value, width)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val", "train"])
    ap.add_argument("--baseline", action="store_true", help="use the current prompt, not GEPA's")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args()

    mode = "BASELINE" if args.baseline else apply_optimized_instruction()
    pmcids = get_splits()[args.split][:args.n]
    gold_df = load_sheet("Train").set_index("pmcids")
    parser = NXMLParser()

    print(f"\n### Scenario test — {mode} instruction — {args.split} ({len(pmcids)} papers) ###")
    totals = {f: 0.0 for f in FIELDS}
    for pid in pmcids:
        parsed = parser.parse(str(nxml_path(pid)))
        raw = predict_ollama.extract(pid, parsed, args.model)
        conds = raw.get("conditions", [])
        raw["conditions"] = ", ".join(conds) if isinstance(conds, list) else str(conds or "")
        pred = {f: raw.get(f, "") for f in FIELDS}
        gold = gold_dict(gold_df.loc[pid])
        s = score_row(pred, gold)
        for f in FIELDS:
            totals[f] += s[f]

        print(f"\nPMC{pid}   overall={s['overall']:.3f}")
        for f in FIELDS:
            print(f"  [{f}]  score={s[f]:.2f}")
            show("pred", pred[f])
            show("gold", gold[f])

    print("\n=== mean per-field over %d papers ===" % len(pmcids))
    for f in FIELDS:
        print(f"  {f:22s}: {totals[f] / len(pmcids):.3f}")


if __name__ == "__main__":
    main()
