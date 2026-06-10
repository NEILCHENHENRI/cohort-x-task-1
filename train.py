"""
CohortX Task 1 — Training Entry Point

Usage:
  python train.py --data_dir /path/to/data --nxml_dir /path/to/PMC_NXML_Archives [--gpu]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from models import CohortXPipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--nxml_dir",   required=True)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--gpu",        action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    train_df = pd.read_excel(Path(args.data_dir) / "Task_1.xlsx", sheet_name="Train")
    log.info(f"Train: {len(train_df)} rows | columns: {list(train_df.columns)}")
    CohortXPipeline(
        nxml_dir=args.nxml_dir,
        models_dir=args.models_dir,
        use_gpu=args.gpu,
    ).train(train_df)


if __name__ == "__main__":
    main()
