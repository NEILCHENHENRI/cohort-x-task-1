from pathlib import Path

import pandas as pd


# ============================================================
# File paths
# ============================================================

INPUT_PATH = Path("dual_model_labeled.csv")
OUTPUT_PATH = Path("disagreement_combinations.csv")


# Keep the order consistent with the labeling prompt.
LABELS = [
    "Condition & Study Type",
    "Demographics",
    "Eligibility Criteria",
    "No Useful Evidence",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"haiku_label", "gemini_label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Remove surrounding whitespace.
    df["haiku_label"] = df["haiku_label"].astype("string").str.strip()
    df["gemini_label"] = df["gemini_label"].astype("string").str.strip()

    # Only analyze rows where both models produced valid labels.
    valid_rows = df[
        df["haiku_label"].isin(LABELS)
        & df["gemini_label"].isin(LABELS)
    ].copy()

    disagreements = valid_rows[
        valid_rows["haiku_label"] != valid_rows["gemini_label"]
    ].copy()

    # Generate all 12 possible directional disagreement combinations,
    # including combinations that occurred zero times.
    all_combinations = pd.MultiIndex.from_product(
        [LABELS, LABELS],
        names=["haiku_label", "gemini_label"],
    ).to_frame(index=False)

    all_combinations = all_combinations[
        all_combinations["haiku_label"]
        != all_combinations["gemini_label"]
    ]

    # Count combinations actually observed in the CSV.
    observed_counts = (
        disagreements
        .groupby(
            ["haiku_label", "gemini_label"],
            observed=False,
        )
        .size()
        .reset_index(name="count")
    )

    results = all_combinations.merge(
        observed_counts,
        on=["haiku_label", "gemini_label"],
        how="left",
    )

    results["count"] = results["count"].fillna(0).astype(int)

    total_disagreements = len(disagreements)

    if total_disagreements > 0:
        results["percentage_of_disagreements"] = (
            results["count"] / total_disagreements * 100
        )
    else:
        results["percentage_of_disagreements"] = 0.0

    results["percentage_of_disagreements"] = (
        results["percentage_of_disagreements"].round(2)
    )

    # Add a readable directional name.
    results["combination"] = (
        results["haiku_label"]
        + " → "
        + results["gemini_label"]
    )

    # Put the most useful columns first.
    results = results[
        [
            "combination",
            "haiku_label",
            "gemini_label",
            "count",
            "percentage_of_disagreements",
        ]
    ]

    # Sort from most common to least common.
    results = results.sort_values(
        by=["count", "haiku_label", "gemini_label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print("=" * 100)
    print("MODEL DISAGREEMENT ANALYSIS")
    print("=" * 100)

    print(f"Total rows in CSV:             {len(df):,}")
    print(f"Rows with two valid labels:    {len(valid_rows):,}")
    print(f"Agreement rows:                {len(valid_rows) - total_disagreements:,}")
    print(f"Disagreement rows:             {total_disagreements:,}")

    if len(valid_rows) > 0:
        disagreement_rate = (
            total_disagreements / len(valid_rows) * 100
        )
        print(
            f"Disagreement rate:             "
            f"{disagreement_rate:.2f}%"
        )

    print("\nAll 12 directional disagreement combinations:\n")

    print(
        results.to_string(
            index=False,
            columns=[
                "combination",
                "count",
                "percentage_of_disagreements",
            ],
        )
    )

    print(f"\nSaved summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()