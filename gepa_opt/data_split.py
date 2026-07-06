"""
CohortX Task 1 — Deterministic data split (dev-time)
====================================================
One shared, seeded split of the Train sheet into train / val / holdout, persisted
to results/splits.json so EVERY phase (baseline, GEPA, final eval) scores on the
identical held-out rows. No dspy / no Anthropic imports here — kept dependency-light
so the offline baseline runner can use it too.

  - train   : GEPA reflection set
  - val     : GEPA Pareto-selection set (seen by the optimizer)
  - holdout : honest before/after eval (GEPA never sees it)

Only pmcids whose .nxml file actually exists are eligible.
"""

import json
import random
from pathlib import Path

import pandas as pd

DATA_DIR   = Path.home() / ".cache/kagglehub/competitions/cohort-x-task-1"
DATA_XLSX  = DATA_DIR / "Task_1.xlsx"
NXML_DIR   = DATA_DIR / "PMC_NXML_Archives"
SPLIT_FILE = Path("results/splits.json")
SEED       = 42

FIELDS = ["conditions", "study_type", "sex",
          "minimum_age", "maximum_age", "eligibility_criteria"]


def load_sheet(sheet: str = "Train") -> pd.DataFrame:
    df = pd.read_excel(DATA_XLSX, sheet_name=sheet)
    df["pmcids"] = df["pmcids"].astype(str).str.strip()
    return df


def nxml_path(pid: str) -> Path:
    return NXML_DIR / f"PMC{pid}.nxml"


def gold_dict(row) -> dict:
    """Gold field dict for one row, NaN -> '' (matches evaluate.py expectations)."""
    return {f: ("" if pd.isna(row[f]) else str(row[f])) for f in FIELDS}


def get_splits(n_train: int = 60, n_val: int = 40, n_holdout: int = 60,
               force: bool = False) -> dict:
    """Return {'train': [...], 'val': [...], 'holdout': [...]} of pmcids.

    Persisted on first call; reused verbatim thereafter (pass force=True to
    regenerate, e.g. after changing sizes)."""
    if SPLIT_FILE.exists() and not force:
        return json.loads(SPLIT_FILE.read_text())

    df = load_sheet("Train")
    pmcids = [p for p in df["pmcids"].tolist() if nxml_path(p).exists()]
    random.Random(SEED).shuffle(pmcids)

    a, b, c = n_train, n_train + n_val, n_train + n_val + n_holdout
    splits = {"train": pmcids[:a], "val": pmcids[a:b], "holdout": pmcids[b:c]}

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(splits, indent=2))
    return splits


if __name__ == "__main__":
    s = get_splits()
    df = load_sheet("Train")
    print(f"Eligible (nxml present): {sum(nxml_path(p).exists() for p in df['pmcids'])}/{len(df)}")
    for k, v in s.items():
        print(f"  {k:8s}: {len(v)} rows")
