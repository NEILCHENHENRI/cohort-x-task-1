from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path("dual_model_labeled.csv")

# Complete prepared dataset, including provenance columns.
OUTPUT_PATH = Path("bert_finetuning_data.csv")

# Optional separate files for easier training.
TRAIN_PATH = Path("bert_train.csv")
VAL_PATH = Path("bert_val.csv")
TEST_PATH = Path("bert_test.csv")


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

NO_EVIDENCE = "No Useful Evidence"

POSITIVE_LABELS = [
    "Condition & Study Type",
    "Demographics",
    "Eligibility Criteria",
]

ALL_LABELS = POSITIVE_LABELS + [NO_EVIDENCE]

# Binary-target column names used for model training.
TARGET_COLUMNS = {
    "Condition & Study Type": "target_condition_study_type",
    "Demographics": "target_demographics",
    "Eligibility Criteria": "target_eligibility_criteria",
}


# ============================================================
# Cleaning and validation
# ============================================================

def clean_text(value: object) -> str:
    """Convert a CSV cell to a clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_input(df: pd.DataFrame) -> None:
    required_columns = {
        "sample_id",
        "doc_id",
        "block_id",
        "section_path",
        "text",
        "haiku_label",
        "gemini_label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    for model_column in ["haiku_label", "gemini_label"]:
        invalid = df.loc[
            ~df[model_column].isin(ALL_LABELS),
            ["sample_id", model_column],
        ]

        if not invalid.empty:
            preview = invalid.head(10).to_dict("records")
            raise ValueError(
                f"{model_column} contains missing or invalid labels. "
                f"Examples: {preview}"
            )

    if df["sample_id"].duplicated().any():
        duplicate_ids = (
            df.loc[df["sample_id"].duplicated(), "sample_id"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"sample_id must be unique. Duplicates include: {duplicate_ids}"
        )

    if df["doc_id"].isna().any():
        raise ValueError(
            "doc_id contains missing values; document-level splitting "
            "would not be reliable."
        )


# ============================================================
# Label consolidation
# ============================================================

def resolve_labels(
    haiku_label: str,
    gemini_label: str,
) -> tuple[list[str], str]:
    """
    Apply the user's union-based disagreement policy.

    Returns:
        final positive labels
        resolution category
    """
    haiku_label = clean_text(haiku_label)
    gemini_label = clean_text(gemini_label)

    # Both models reject the block.
    if haiku_label == NO_EVIDENCE and gemini_label == NO_EVIDENCE:
        return [], "both_no_useful_evidence"

    # Haiku supplies the only positive label.
    if gemini_label == NO_EVIDENCE:
        return [haiku_label], "kept_haiku_useful_label"

    # Gemini supplies the only positive label.
    if haiku_label == NO_EVIDENCE:
        return [gemini_label], "kept_gemini_useful_label"

    # Both models select the same positive label.
    if haiku_label == gemini_label:
        return [haiku_label], "agreed_useful_label"

    # Both are positive but select different fields.
    final_labels = [
        label
        for label in POSITIVE_LABELS
        if label in {haiku_label, gemini_label}
    ]

    return final_labels, "combined_two_useful_labels"


def add_final_targets(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    resolved = output.apply(
        lambda row: resolve_labels(
            row["haiku_label"],
            row["gemini_label"],
        ),
        axis=1,
    )

    output["final_labels_list"] = resolved.map(lambda result: result[0])
    output["resolution_method"] = resolved.map(lambda result: result[1])

    # A readable form for inspecting the CSV manually.
    output["final_labels"] = output["final_labels_list"].map(
        lambda labels: " | ".join(labels)
        if labels
        else NO_EVIDENCE
    )

    # JSON representation is convenient for reloading.
    output["final_labels_json"] = output["final_labels_list"].map(
        lambda labels: json.dumps(labels, ensure_ascii=False)
    )

    output["number_of_positive_labels"] = output[
        "final_labels_list"
    ].map(len)

    output["is_multilabel"] = (
        output["number_of_positive_labels"] > 1
    ).astype(int)

    output["is_no_useful_evidence"] = (
        output["number_of_positive_labels"] == 0
    ).astype(int)

    # Three independent binary targets for BCEWithLogitsLoss.
    for label, target_column in TARGET_COLUMNS.items():
        output[target_column] = output["final_labels_list"].map(
            lambda labels, current_label=label: int(
                current_label in labels
            )
        )

    # This is the model input. The separator is plain text so the tokenizer
    # does not require a custom vocabulary.
    output["model_input"] = (
        "Section: "
        + output["section_path"].fillna("").astype(str).str.strip()
        + "\nText: "
        + output["text"].fillna("").astype(str).str.strip()
    )

    output = output.drop(columns=["final_labels_list"])

    return output


# ============================================================
# Document-level splitting
# ============================================================

def add_document_level_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create deterministic 80/10/10 splits by doc_id.

    Every block from the same document is assigned to exactly one split.
    """
    if not np.isclose(
        TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION,
        1.0,
    ):
        raise ValueError(
            "TRAIN_FRACTION, VAL_FRACTION, and TEST_FRACTION "
            "must sum to 1."
        )

    output = df.copy()
    output["split"] = ""

    groups = output["doc_id"].astype(str)

    # First split: train versus temporary validation/test pool.
    first_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_SEED,
    )

    train_indices, temporary_indices = next(
        first_splitter.split(
            X=output,
            groups=groups,
        )
    )

    output.loc[output.index[train_indices], "split"] = "train"

    temporary = output.iloc[temporary_indices].copy()
    temporary_groups = temporary["doc_id"].astype(str)

    # Within the remaining 20%, divide validation and test evenly
    # according to their requested relative proportions.
    validation_share_of_temporary = (
        VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    )

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_share_of_temporary,
        random_state=RANDOM_SEED + 1,
    )

    val_relative_indices, test_relative_indices = next(
        second_splitter.split(
            X=temporary,
            groups=temporary_groups,
        )
    )

    val_original_indices = temporary.index[val_relative_indices]
    test_original_indices = temporary.index[test_relative_indices]

    output.loc[val_original_indices, "split"] = "validation"
    output.loc[test_original_indices, "split"] = "test"

    if (output["split"] == "").any():
        raise RuntimeError("Some rows were not assigned to a split.")

    # Verify that no document appears in more than one split.
    split_counts_per_document = (
        output.groupby("doc_id")["split"].nunique()
    )

    leaked_documents = split_counts_per_document[
        split_counts_per_document > 1
    ]

    if not leaked_documents.empty:
        raise RuntimeError(
            "Document leakage detected across splits: "
            f"{leaked_documents.index.tolist()[:10]}"
        )

    return output


# ============================================================
# Output and diagnostics
# ============================================================

def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep model-ready columns first, followed by useful annotation provenance.
    """
    ordered_columns = [
        "sample_id",
        "doc_id",
        "block_id",
        "split",
        "section_path",
        "text",
        "model_input",

        # Model-training targets
        "target_condition_study_type",
        "target_demographics",
        "target_eligibility_criteria",

        # Human-readable target information
        "final_labels",
        "final_labels_json",
        "number_of_positive_labels",
        "is_multilabel",
        "is_no_useful_evidence",

        # Annotation provenance
        "resolution_method",
        "haiku_label",
        "gemini_label",
    ]

    # Include model explanations for later auditing, but they should not be
    # supplied as classifier inputs.
    optional_columns = [
        "haiku_reason",
        "gemini_reason",
        "haiku_overall_relevance",
        "gemini_overall_relevance",
    ]

    for column in optional_columns:
        if column in df.columns:
            ordered_columns.append(column)

    return df[ordered_columns].copy()


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 76)
    print("BERT FINE-TUNING DATA SUMMARY")
    print("=" * 76)

    print(f"Total rows:                 {len(df):,}")
    print(f"Unique documents:           {df['doc_id'].nunique():,}")

    print("\nResolution methods:")
    print(
        df["resolution_method"]
        .value_counts()
        .rename_axis("method")
        .to_string()
    )

    print("\nFinal target counts:")
    for label, column in TARGET_COLUMNS.items():
        print(f"  {label:<28} {int(df[column].sum()):>5,}")

    no_evidence_count = int(df["is_no_useful_evidence"].sum())
    multilabel_count = int(df["is_multilabel"].sum())

    print(f"  {NO_EVIDENCE:<28} {no_evidence_count:>5,}")
    print(f"\nMulti-label rows:           {multilabel_count:,}")
    print(
        "Multi-label percentage:     "
        f"{multilabel_count / len(df):.2%}"
    )

    print("\nSplit sizes:")
    split_summary = (
        df.groupby("split")
        .agg(
            rows=("sample_id", "size"),
            documents=("doc_id", "nunique"),
            condition=("target_condition_study_type", "sum"),
            demographics=("target_demographics", "sum"),
            eligibility=("target_eligibility_criteria", "sum"),
            no_evidence=("is_no_useful_evidence", "sum"),
            multilabel=("is_multilabel", "sum"),
        )
        .reindex(["train", "validation", "test"])
    )

    print(split_summary.to_string())

    print("\nFiles written:")
    print(f"  Complete:   {OUTPUT_PATH}")
    print(f"  Train:      {TRAIN_PATH}")
    print(f"  Validation: {VAL_PATH}")
    print(f"  Test:       {TEST_PATH}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Normalize the two model-label columns before validation.
    for column in ["haiku_label", "gemini_label"]:
        df[column] = df[column].map(clean_text)

    validate_input(df)

    prepared = add_final_targets(df)
    prepared = add_document_level_split(prepared)
    prepared = select_output_columns(prepared)

    # Save the complete dataset.
    prepared.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Save split-specific files.
    prepared.loc[prepared["split"] == "train"].to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    prepared.loc[prepared["split"] == "validation"].to_csv(
        VAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    prepared.loc[prepared["split"] == "test"].to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(prepared)


if __name__ == "__main__":
    main()