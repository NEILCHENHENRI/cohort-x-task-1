"""
CohortX Task 1 — Dataset downloader (dev-time only)
===================================================
Pulls the competition files via kagglehub and reports the local layout so the
inference/eval scripts can be pointed at them with --data_dir / --nxml_dir.

Auth: requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY) for an
account that has accepted the competition rules. No secrets live in this file.

Usage:
  python data/download_data.py
"""

from pathlib import Path

import kagglehub

COMPETITION = "cohort-x-task-1"


def main() -> None:
    path = Path(kagglehub.competition_download(COMPETITION))
    print(f"\nDownloaded to: {path}\n")

    xlsx = sorted(path.rglob("*.xlsx"))
    nxml = sorted(path.rglob("*.nxml"))

    print(f"  .xlsx files : {len(xlsx)}")
    for f in xlsx:
        print(f"    {f.relative_to(path)}")
    print(f"  .nxml files : {len(nxml)}")
    if nxml:
        print(f"    e.g. {nxml[0].relative_to(path)}")
        # Report the directory the .nxml files actually live in.
        nxml_dirs = sorted({f.parent for f in nxml})
        print(f"  nxml dir(s) : {[str(d) for d in nxml_dirs]}")

    print("\nTop-level entries:")
    for entry in sorted(path.iterdir()):
        kind = "dir " if entry.is_dir() else "file"
        print(f"    [{kind}] {entry.name}")


if __name__ == "__main__":
    main()
