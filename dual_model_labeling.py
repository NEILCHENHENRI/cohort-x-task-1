from __future__ import annotations

import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tqdm import tqdm


# ============================================================
# Hardcoded paths
# ============================================================

INPUT_PATH = Path("labeling_samples_text_only_labeled.csv")
OUTPUT_PATH = Path("dual_model_labeled.csv")

# Exact model-returned content is written here.
RAW_OUTPUT_PATH = Path("raw_model_outputs.csv")

# Request settings, latency, usage, costs, errors, etc.
EXPERIMENT_LOG_PATH = Path("experiment_log.csv")


# ============================================================
# OpenRouter configuration
# ============================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "haiku": {
        "model_id": "anthropic/claude-haiku-4.5",
        "input_price_per_million": 1.00,
        "output_price_per_million": 5.00,
    },
    "gemini": {
        "model_id": "google/gemini-2.5-flash",
        "input_price_per_million": 0.30,
        "output_price_per_million": 2.50,
    },
}

# Set to a number such as 10 for an initial experiment.
# Use None for the complete CSV.
ROW_LIMIT: int | None = 5
ROW_LIMIT = None

# Number of API retries after transient failures.
MAX_RETRIES = 6

# Save the main output after this many successful model calls.
SAVE_EVERY = 1

ALLOWED_LABELS = [
    "Condition & Study Type",
    "Demographics",
    "Eligibility Criteria",
    "No Useful Evidence",
]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are labeling evidence for CohortX Task 1, which extracts structured
information from biomedical articles.

Only consider evidence describing the current study. Ignore background
information, literature review, discussion of previous work, cited studies,
future work, and general medical knowledge.

Determine whether the supplied article block provides useful evidence for
extracting one of the following categories.

1. Condition & Study Type

Evidence about the disease, disorder, health condition, population condition,
trial design, observational design, randomized design, prospective or
retrospective design, cohort type, control structure, masking, phase, or other
study-design characteristics of the current study.

2. Demographics

Evidence about participant sex or gender, minimum age, maximum age, age range,
or demographic restrictions in the current study. A mean or median age may
count when it directly describes the current study population.

3. Eligibility Criteria

Evidence about inclusion criteria, exclusion criteria, participant-selection
requirements, diagnostic requirements, clinical thresholds, prior treatments,
contraindications, or other rules determining who could or could not
participate in the current study.

4. No Useful Evidence

Use this when the block does not provide useful evidence for the three
categories above, concerns a previous or cited study, merely describes
procedures, interventions, outcomes, measurements, analyses, or results
without relevant field evidence, or is too vague to support extraction.

Important rules:

- Assign exactly one label.
- Judge the block semantically rather than by keyword matching.
- Only classify evidence about the current study.
- A condition mentioned as general background is No Useful Evidence.
- A heading alone is not sufficient unless the accompanying text provides
  extractable evidence.
- Treatment details, outcomes, measurements, statistical methods, and results
  are not by themselves evidence for these fields.
- When more than one category is present, select the category containing the
  strongest and most directly extractable evidence.
- Do not infer criteria, demographics, conditions, or study designs that are
  not stated or clearly entailed by the text.

Assign an overall_relevance integer from 1 to 5:

1 = clearly irrelevant
2 = weak or indirect relevance
3 = moderately useful but incomplete or ambiguous
4 = directly useful evidence
5 = highly explicit, complete, extraction-ready evidence

The reason must be concise and specific, normally one sentence. Explain why
the block does or does not provide evidence for the selected category. Do not
provide a lengthy analysis.
""".strip()


LABEL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cohortx_evidence_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ALLOWED_LABELS,
                },
                # "overall_relevance": {
                #     "type": "integer",
                #     "minimum": 1,
                #     "maximum": 5,
                # },
                "overall_relevance": {
                    "type": "integer",
                },
                "reason": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": [
                "label",
                "overall_relevance",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


# ============================================================
# CSV log schemas
# ============================================================

RAW_OUTPUT_COLUMNS = [
    "timestamp_utc",
    "sample_id",
    "doc_id",
    "block_id",
    "model_key",
    "model_id",
    "attempt",
    "raw_content",
    "raw_reasoning",
    "raw_reasoning_details",
    "finish_reason",
]

EXPERIMENT_LOG_COLUMNS = [
    "timestamp_utc",
    "sample_id",
    "doc_id",
    "block_id",
    "model_key",
    "model_id",
    "attempt",
    "status",
    "temperature",
    "max_tokens",
    "reasoning_setting",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
    "reasoning_returned",
    "finish_reason",
    "error_type",
    "error_message",
]


# ============================================================
# General helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_user_prompt(row: pd.Series) -> str:
    section_path = clean_cell(row.get("section_path", ""))
    text = clean_cell(row.get("text", ""))

    return f"""
Classify the following article block.

Section path:
{section_path or "[Not provided]"}

Text:
{text}
""".strip()


def append_csv_row(
    path: Path,
    columns: list[str],
    row: dict[str, Any],
) -> None:
    """
    Append one row immediately.

    This means raw responses and experiment records survive even if the main
    script is interrupted later.
    """
    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                column: row.get(column, "")
                for column in columns
            }
        )


def safe_json_dumps(value: Any) -> str:
    if value is None:
        return ""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def get_response_field(obj: Any, field: str) -> Any:
    """
    Read fields returned either as SDK attributes or mapping keys.
    """
    if obj is None:
        return None

    value = getattr(obj, field, None)

    if value is not None:
        return value

    if isinstance(obj, dict):
        return obj.get(field)

    return None


# ============================================================
# Parsing
# ============================================================

def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

        if content.lower().startswith("json"):
            content = content[4:].lstrip()

    result = json.loads(content)

    if not isinstance(result, dict):
        raise ValueError("Response is not a JSON object.")

    label = result.get("label")
    relevance = result.get("overall_relevance")
    reason = result.get("reason")

    if label not in ALLOWED_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    if (
        isinstance(relevance, bool)
        or not isinstance(relevance, int)
        or relevance < 1
        or relevance > 5
    ):
        raise ValueError(
            f"Invalid overall_relevance: {relevance!r}"
        )

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Reason is missing or empty.")

    return {
        "label": label,
        "overall_relevance": relevance,
        "reason": reason.strip(),
    }


# ============================================================
# Token usage and costs
# ============================================================

def extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    input_tokens = int(
        get_response_field(usage, "prompt_tokens") or 0
    )
    output_tokens = int(
        get_response_field(usage, "completion_tokens") or 0
    )

    reasoning_tokens = 0

    # OpenAI-compatible completion token details.
    completion_details = get_response_field(
        usage,
        "completion_tokens_details",
    )

    if completion_details is not None:
        reasoning_tokens = int(
            get_response_field(
                completion_details,
                "reasoning_tokens",
            )
            or 0
        )

    # Fallback in case a provider exposes it directly.
    if reasoning_tokens == 0:
        reasoning_tokens = int(
            get_response_field(usage, "reasoning_tokens") or 0
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def calculate_cost(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    config = MODELS[model_key]

    input_cost = (
        input_tokens
        / 1_000_000
        * config["input_price_per_million"]
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * config["output_price_per_million"]
    )

    return input_cost + output_cost


# ============================================================
# Request configuration
# ============================================================

def make_extra_body(model_key: str) -> dict[str, Any]:
    extra_body: dict[str, Any] = {
        "provider": {
            # Do not route to a provider that silently ignores required
            # parameters such as structured output or reasoning settings.
            "require_parameters": True,
        },
    }

    if model_key == "gemini":
        # Explicitly disable Gemini thinking.
        #
        # Do not use:
        #     {"exclude": True}
        #
        # That would merely hide reasoning from the response while still
        # allowing the model to generate and bill reasoning tokens.
        extra_body["reasoning"] = {
            "effort": "none",
        }

    return extra_body


def reasoning_setting_for_log(model_key: str) -> str:
    if model_key == "gemini":
        return '{"effort":"none"}'

    # Haiku receives no reasoning parameter.
    return "not_set"


# ============================================================
# API call
# ============================================================

def call_model(
    client: OpenAI,
    model_key: str,
    row: pd.Series,
) -> dict[str, Any]:
    model_id = MODELS[model_key]["model_id"]
    user_prompt = make_user_prompt(row)

    sample_id = clean_cell(row.get("sample_id", ""))
    doc_id = clean_cell(row.get("doc_id", ""))
    block_id = clean_cell(row.get("block_id", ""))

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.perf_counter()
        timestamp = utc_now()

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0,
                max_tokens=180,
                response_format=LABEL_SCHEMA,
                extra_body=make_extra_body(model_key),
            )

            latency = time.perf_counter() - start_time

            if not response.choices:
                raise ValueError("API returned no choices.")

            choice = response.choices[0]
            message = choice.message

            # Preserve content exactly as returned.
            raw_content = message.content or ""

            # Preserve reasoning fields if OpenRouter/provider returns them.
            raw_reasoning = get_response_field(
                message,
                "reasoning",
            )
            raw_reasoning_details = get_response_field(
                message,
                "reasoning_details",
            )

            finish_reason = (
                get_response_field(choice, "finish_reason") or ""
            )

            usage = extract_usage(response)

            cost = calculate_cost(
                model_key=model_key,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )

            reasoning_returned = bool(
                raw_reasoning
                or raw_reasoning_details
                or usage["reasoning_tokens"] > 0
            )

            # ------------------------------------------------
            # Raw model output log
            # ------------------------------------------------

            append_csv_row(
                path=RAW_OUTPUT_PATH,
                columns=RAW_OUTPUT_COLUMNS,
                row={
                    "timestamp_utc": timestamp,
                    "sample_id": sample_id,
                    "doc_id": doc_id,
                    "block_id": block_id,
                    "model_key": model_key,
                    "model_id": model_id,
                    "attempt": attempt,
                    "raw_content": raw_content,
                    "raw_reasoning": (
                        raw_reasoning
                        if isinstance(raw_reasoning, str)
                        else safe_json_dumps(raw_reasoning)
                    ),
                    "raw_reasoning_details": safe_json_dumps(
                        raw_reasoning_details
                    ),
                    "finish_reason": finish_reason,
                },
            )

            # Parse only after saving the verbatim output.
            parsed = parse_json_content(raw_content)

            # ------------------------------------------------
            # Experiment log
            # ------------------------------------------------

            append_csv_row(
                path=EXPERIMENT_LOG_PATH,
                columns=EXPERIMENT_LOG_COLUMNS,
                row={
                    "timestamp_utc": timestamp,
                    "sample_id": sample_id,
                    "doc_id": doc_id,
                    "block_id": block_id,
                    "model_key": model_key,
                    "model_id": model_id,
                    "attempt": attempt,
                    "status": "success",
                    "temperature": 0,
                    "max_tokens": 180,
                    "reasoning_setting": (
                        reasoning_setting_for_log(model_key)
                    ),
                    "latency_seconds": round(latency, 4),
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage[
                        "reasoning_tokens"
                    ],
                    "cost_usd": round(cost, 10),
                    "reasoning_returned": reasoning_returned,
                    "finish_reason": finish_reason,
                    "error_type": "",
                    "error_message": "",
                },
            )

            return {
                **parsed,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage[
                    "reasoning_tokens"
                ],
                "cost_usd": cost,
                "latency_seconds": latency,
                "reasoning_returned": reasoning_returned,
                "error": "",
            }

        except (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            latency = time.perf_counter() - start_time
            last_error = exc

            append_csv_row(
                path=EXPERIMENT_LOG_PATH,
                columns=EXPERIMENT_LOG_COLUMNS,
                row={
                    "timestamp_utc": timestamp,
                    "sample_id": sample_id,
                    "doc_id": doc_id,
                    "block_id": block_id,
                    "model_key": model_key,
                    "model_id": model_id,
                    "attempt": attempt,
                    "status": "error",
                    "temperature": 0,
                    "max_tokens": 180,
                    "reasoning_setting": (
                        reasoning_setting_for_log(model_key)
                    ),
                    "latency_seconds": round(latency, 4),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0,
                    "reasoning_returned": False,
                    "finish_reason": "",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

            if attempt == MAX_RETRIES:
                break

            delay = min(60.0, 2 ** (attempt - 1))
            delay += random.uniform(0, 1)
            time.sleep(delay)

    return {
        "label": "",
        "overall_relevance": pd.NA,
        "reason": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "latency_seconds": pd.NA,
        "reasoning_returned": False,
        "error": (
            f"{type(last_error).__name__}: {last_error}"
            if last_error
            else "Unknown error"
        ),
    }


# ============================================================
# Main output helpers
# ============================================================

def initialize_output_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    defaults = {
        "haiku_label": "",
        "haiku_overall_relevance": pd.NA,
        "haiku_reason": "",
        "haiku_input_tokens": 0,
        "haiku_output_tokens": 0,
        "haiku_reasoning_tokens": 0,
        "haiku_cost_usd": 0.0,
        "haiku_latency_seconds": pd.NA,
        "haiku_reasoning_returned": False,
        "haiku_error": "",
        "gemini_label": "",
        "gemini_overall_relevance": pd.NA,
        "gemini_reason": "",
        "gemini_input_tokens": 0,
        "gemini_output_tokens": 0,
        "gemini_reasoning_tokens": 0,
        "gemini_cost_usd": 0.0,
        "gemini_latency_seconds": pd.NA,
        "gemini_reasoning_returned": False,
        "gemini_error": "",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    return df


def result_is_complete(
    row: pd.Series,
    model_key: str,
) -> bool:
    label = clean_cell(
        row.get(f"{model_key}_label", "")
    )
    error = clean_cell(
        row.get(f"{model_key}_error", "")
    )

    return label in ALLOWED_LABELS and not error


def add_comparison_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    haiku_valid = df["haiku_label"].isin(ALLOWED_LABELS)
    gemini_valid = df["gemini_label"].isin(ALLOWED_LABELS)

    df["models_agree"] = (
        haiku_valid
        & gemini_valid
        & (df["haiku_label"] == df["gemini_label"])
    )

    df["label_disagreement"] = (
        haiku_valid
        & gemini_valid
        & (df["haiku_label"] != df["gemini_label"])
    )

    haiku_relevance = pd.to_numeric(
        df["haiku_overall_relevance"],
        errors="coerce",
    )
    gemini_relevance = pd.to_numeric(
        df["gemini_overall_relevance"],
        errors="coerce",
    )

    df["relevance_difference"] = (
        haiku_relevance - gemini_relevance
    ).abs()

    df["needs_review"] = (
        df["label_disagreement"]
        | (df["relevance_difference"] >= 2)
        | df["haiku_error"].fillna("").astype(str).str.len().gt(0)
        | df["gemini_error"].fillna("").astype(str).str.len().gt(0)
    )

    if "label" in df.columns:
        df["haiku_matches_existing"] = (
            haiku_valid
            & (df["haiku_label"] == df["label"])
        )

        df["gemini_matches_existing"] = (
            gemini_valid
            & (df["gemini_label"] == df["label"])
        )

    return df


def save_main_output(df: pd.DataFrame) -> None:
    temporary_path = OUTPUT_PATH.with_suffix(
        OUTPUT_PATH.suffix + ".tmp"
    )

    df.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(OUTPUT_PATH)


def print_summary(df: pd.DataFrame) -> None:
    haiku_valid = df["haiku_label"].isin(ALLOWED_LABELS)
    gemini_valid = df["gemini_label"].isin(ALLOWED_LABELS)
    both_valid = haiku_valid & gemini_valid

    completed = df.loc[both_valid]

    haiku_cost = pd.to_numeric(
        df["haiku_cost_usd"],
        errors="coerce",
    ).fillna(0).sum()

    gemini_cost = pd.to_numeric(
        df["gemini_cost_usd"],
        errors="coerce",
    ).fillna(0).sum()

    print("\nSummary")
    print("=" * 60)
    print(f"Rows selected:             {len(df):,}")
    print(f"Haiku completed:           {int(haiku_valid.sum()):,}")
    print(f"Gemini completed:          {int(gemini_valid.sum()):,}")
    print(f"Both completed:            {len(completed):,}")

    if len(completed) > 0:
        print(
            "Agreement rate:           "
            f"{completed['models_agree'].mean():.2%}"
        )

        print(
            "Rows needing review:      "
            f"{int(completed['needs_review'].sum()):,}"
        )

    gemini_reasoning_rows = int(
        df["gemini_reasoning_returned"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    gemini_reasoning_tokens = int(
        pd.to_numeric(
            df["gemini_reasoning_tokens"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print(f"Haiku cost:                ${haiku_cost:.4f}")
    print(f"Gemini cost:               ${gemini_cost:.4f}")
    print(
        f"Total cost:                "
        f"${haiku_cost + gemini_cost:.4f}"
    )
    print(
        "Gemini reasoning returned: "
        f"{gemini_reasoning_rows:,} rows"
    )
    print(
        "Gemini reasoning tokens:   "
        f"{gemini_reasoning_tokens:,}"
    )

    print(f"\nMain output:       {OUTPUT_PATH}")
    print(f"Raw responses:     {RAW_OUTPUT_PATH}")
    print(f"Experiment log:    {EXPERIMENT_LOG_PATH}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH}"
        )

    # Resume from an existing main output when available.
    if OUTPUT_PATH.exists():
        print(f"Resuming from {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "doc_id",
        "block_id",
        "section_path",
        "text",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = initialize_output_columns(df)

    text_output_columns = [
        "haiku_label",
        "haiku_reason",
        "haiku_error",
        "gemini_label",
        "gemini_reason",
        "gemini_error",
    ]

    for column in text_output_columns:
        df[column] = df[column].astype("object")

    if ROW_LIMIT is None:
        indices = list(df.index)
    else:
        indices = list(df.index[:ROW_LIMIT])

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=90.0,
        max_retries=0,
        default_headers={
            "X-OpenRouter-Title": (
                "CohortX Evidence Labeling"
            ),
        },
    )

    model_keys = ["haiku", "gemini"]
    newly_processed = 0

    total_calls = len(indices) * len(model_keys)

    with tqdm(total=total_calls) as progress:
        for index in indices:
            for model_key in model_keys:
                row = df.loc[index]

                if result_is_complete(row, model_key):
                    progress.update(1)
                    continue

                result = call_model(
                    client=client,
                    model_key=model_key,
                    row=row,
                )

                for field, value in result.items():
                    column = f"{model_key}_{field}"
                    df.at[index, column] = value

                newly_processed += 1
                progress.update(1)

                progress.set_postfix(
                    sample_id=clean_cell(
                        row.get("sample_id", "")
                    ),
                    model=model_key,
                    result=(
                        result["label"]
                        if result["label"]
                        else "ERROR"
                    ),
                )

                if newly_processed % SAVE_EVERY == 0:
                    df = add_comparison_columns(df)
                    save_main_output(df)

    df = add_comparison_columns(df)
    save_main_output(df)
    print_summary(df)


if __name__ == "__main__":
    main()