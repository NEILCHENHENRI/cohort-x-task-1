"""
CohortX Task 1 — Inference with Fine-tuned Models
===================================================
Loads trained model checkpoints and runs extraction on a dataset split.

Usage:
  # Run on test set
  python -m finetuned_models.predict \
    --data_dir   ~/Downloads/cohort-x-task-1 \
    --nxml_dir   ~/Downloads/cohort-x-task-1/PMC_NXML_Archives \
    --models_dir ~/Downloads/cohort-x-task-1/models \
    --gpu

  # Evaluate on training set
  python -m finetuned_models.predict \
    --data_dir   ~/Downloads/cohort-x-task-1 \
    --nxml_dir   ~/Downloads/cohort-x-task-1/PMC_NXML_Archives \
    --models_dir ~/Downloads/cohort-x-task-1/models \
    --sheet      Train \
    --output     train_preds.xlsx \
    --gpu
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from finetuned_models.models import CohortXPipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--nxml_dir",   required=True)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--sheet",      default="Test",
                   help="Excel sheet to run on (default: Test, use Train to evaluate)")
    p.add_argument("--output",     default="submission.xlsx")
    p.add_argument("--gpu",        action="store_true")
    return p.parse_args()


def main():
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    save_path = data_dir / args.output

    df     = pd.read_excel(data_dir / "Task_1.xlsx", sheet_name=args.sheet)
    pmcids = df["pmcids"].astype(str).str.strip().tolist()
    log.info(f"Loaded {len(pmcids)} PMC IDs from sheet '{args.sheet}'")

    # Load pipeline and trained model checkpoints
    pl = CohortXPipeline(
        nxml_dir=args.nxml_dir,
        models_dir=args.models_dir,
        use_gpu=args.gpu,
    )
    pl.load_trained_models()

    # Resume support
    if save_path.exists():
        existing_df = pd.read_excel(save_path)
        done_pids   = set(existing_df["pmcids"].astype(str).str.strip())
        rows        = existing_df.to_dict("records")
        pmcids      = [p for p in pmcids if p not in done_pids]
        log.info(f"Resuming: {len(done_pids)} done, {len(pmcids)} remaining.")
    else:
        rows = []

    # Run extraction
    pbar = tqdm(pmcids, desc="Extracting",
                total=len(df),
                initial=len(df) - len(pmcids))

    for pid in pbar:
        parsed = pl.load_nxml(pid)
        result = pl.extract_all(parsed)

        conds = result.get("conditions", [])
        result["conditions"] = (
            ", ".join(conds) if isinstance(conds, list) else str(conds or "")
        )
        rows.append({"pmcids": pid, **result})

        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_excel(save_path, index=False)
            pbar.set_postfix({"saved": len(rows)})

    pd.DataFrame(rows).to_excel(save_path, index=False)
    log.info(f"Saved {len(rows)} rows → {save_path}")


if __name__ == "__main__":
    main()
