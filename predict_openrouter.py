from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import csv
import re
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"

# # Current OpenRouter prices for Sonnet 4.6.
# INPUT_PRICE_PER_MILLION = 3.00
# OUTPUT_PRICE_PER_MILLION = 15.00

# CLASSIFIER_DIR = Path("final_retrieval_classifier")

# # These should point to the unified candidate files created by retrieval.py.
# UNIFIED_CANDIDATE_DIR = Path("unified_candidates")
# OUTPUT_DIR = Path("sonnet_extractions")

# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"

INPUT_PRICE_PER_MILLION = 3.00
OUTPUT_PRICE_PER_MILLION = 15.00

# Folder containing your downloaded classifier.
CLASSIFIER_DIR = Path("final_retrieval_classifier")

# Folder containing the unified candidate JSON files.
UNIFIED_CANDIDATE_DIR = Path("unified_candidates")

# Folder where Sonnet extraction outputs will be written.
OUTPUT_DIR = Path("sonnet_run")

# Process 41 papers.
LIMIT = 41
RANDOM_SEED = 42

# True = replace existing per-paper outputs.
# False = skip files already extracted.
OVERWRITE = True

# Deployment thresholds selected from your validation work.
CLASSIFICATION_THRESHOLDS = {
    "condition_study_type": 0.33,
    "demographics": 0.03,
    "eligibility": 0.05,
}

# Used only if config.json contains generic names such as LABEL_0.
#
# IMPORTANT:
# Confirm that this matches the label order used during classifier training.
# DEFAULT_LABEL_ORDER = [
#     "condition_study_type",
#     "demographics",
#     "eligibility",
#     "irrelevant",
# ]

DEFAULT_LABEL_ORDER = [
    "condition_study_type",
    "demographics",
    "eligibility",
]

CLASSIFIER_MAX_LENGTH = 512
CLASSIFIER_BATCH_SIZE = 32

# Prevent abnormally large prompts if one field receives many blocks.
MAX_CONTEXT_CHARS = {
    "condition_study_type": 30_000,
    "demographics": 30_000,
    "eligibility": 50_000,
}

# If a field receives no blocks above threshold, include its highest-probability
# blocks as a safety fallback.
MIN_BLOCKS_PER_FIELD = {
    "condition_study_type": 2,
    "demographics": 2,
    "eligibility": 3,
}

MAX_API_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 180


# =============================================================================
# Output schemas
# =============================================================================

CONDITION_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions": {
            "type": "string",
            "description": (
                "Medical condition, disease, diagnosis, or disorder studied "
                "in the current study. Use 'Not Specified' if unavailable."
            ),
        },
        "study_type": {
            "type": "string",
            "enum": [
                "INTERVENTIONAL",
                "OBSERVATIONAL",
            ],
            "description": "Type of the current study.",
        },
    },
    "required": [
        "conditions",
        "study_type",
    ],
    "additionalProperties": False,
}

DEMOGRAPHICS_SCHEMA = {
    "type": "object",
    "properties": {
        "sex": {
            "type": "string",
            "enum": [
                "MALE",
                "FEMALE",
                "ALL",
            ],
            "description": (
                "Participant sex."
            ),
        },
        "minimum_age": {
            "type": "string",
            "description": (
                "Minimum participant age or lower age limit, preserving units. "
                "Use 'Not Specified' if unavailable."
            ),
        },
        "maximum_age": {
            "type": "string",
            "description": (
                "Maximum participant age or upper age limit, preserving units. "
                "Use 'Not Specified' if unavailable."
            ),
        },
    },
    "required": [
        "sex",
        "minimum_age",
        "maximum_age",
    ],
    "additionalProperties": False,
}

ELIGIBILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "inclusion_criteria": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": (
                "Individual inclusion criteria applied to participants in the "
                "current study. Return an empty array if unavailable."
            ),
        },
        "exclusion_criteria": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": (
                "Individual exclusion criteria applied to participants in the "
                "current study. Return an empty array if unavailable."
            ),
        },
    },
    "required": [
        "inclusion_criteria",
        "exclusion_criteria",
    ],
    "additionalProperties": False,
}

FIELD_SCHEMAS = {
    "condition_study_type": CONDITION_SCHEMA,
    "demographics": DEMOGRAPHICS_SCHEMA,
    "eligibility": ELIGIBILITY_SCHEMA,
}

# Simple extraction generally does not benefit from lengthy reasoning.
REASONING_EFFORT = {
    "condition_study_type": "none",
    "demographics": "none",
    "eligibility": "low",
}

MAX_OUTPUT_TOKENS = {
    "condition_study_type": 500,
    "demographics": 500,
    "eligibility": 1_500,
}


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """
You extract structured information from biomedical articles.

Use only evidence about the current study. Ignore background discussion,
cited studies, and general medical knowledge.

Use the best interpretation supported by the supplied evidence. Reasonable
synthesis across multiple statements is allowed, but do not invent facts.

Follow the JSON schema exactly and return only the required fields.
""".strip()


FIELD_INSTRUCTIONS = {
    "condition_study_type": """
Extract the condition and study type of the current study.

Return study_type as exactly one of:
- INTERVENTIONAL
- OBSERVATIONAL

Use the overall study description to classify the study when appropriate.
""".strip(),

    "demographics": """
Extract sex and age for the current study.

Return sex as exactly one of:
- MALE
- FEMALE
- ALL

Use ALL whenever both male and female participants are represented or eligible,
even if one sex is the large majority.

Preserve reported ages, ranges, and units.
""".strip(),

    "eligibility": """
Extract the inclusion and exclusion criteria used to select participants for
the current study.

Return each criterion as a separate array item. Reasonable synthesis across
multiple current-study statements is allowed. Preserve clinically important thresholds, requirements, and time
windows. Return an empty array when a category is unavailable.
""".strip(),
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RoutedBlock:
    block_id: str
    block_index: int
    section_path: List[str]
    text: str
    retrieval_score: float
    probabilities: Dict[str, float]


# =============================================================================
# Utilities
# =============================================================================

def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "condition": "condition_study_type",
        "condition_and_study_type": "condition_study_type",
        "condition_&_study_type": "condition_study_type",
        "study_type": "condition_study_type",
        "demographic": "demographics",
        "eligibility_criteria": "eligibility",
        "irrelevant": "irrelevant",
        "overall_irrelevant": "irrelevant",
    }

    return aliases.get(normalized, normalized)


def resolve_id2label(model: AutoModelForSequenceClassification) -> Dict[int, str]:
    raw_mapping = {
        int(index): str(label)
        for index, label in model.config.id2label.items()
    }

    generic = all(
        label.upper().startswith("LABEL_")
        for label in raw_mapping.values()
    )

    if generic:
        if model.config.num_labels != len(DEFAULT_LABEL_ORDER):
            raise ValueError(
                "The classifier has generic label names and "
                f"{model.config.num_labels} labels, but DEFAULT_LABEL_ORDER "
                f"contains {len(DEFAULT_LABEL_ORDER)} labels. Update "
                "DEFAULT_LABEL_ORDER to match training."
            )

        resolved = {
            index: DEFAULT_LABEL_ORDER[index]
            for index in range(model.config.num_labels)
        }
    else:
        resolved = {
            index: normalize_label(label)
            for index, label in raw_mapping.items()
        }

    expected = set(DEFAULT_LABEL_ORDER)
    observed = set(resolved.values())

    if observed != expected:
        raise ValueError(
            "Classifier label mapping does not match the expected labels.\n"
            f"Resolved mapping: {resolved}\n"
            f"Expected labels: {sorted(expected)}\n"
            "Change DEFAULT_LABEL_ORDER or the model config before running."
        )

    return resolved


def format_classifier_input(candidate: Dict[str, Any]) -> str:
    section = " > ".join(candidate.get("section_path", []))
    block_type = candidate.get("block_type", "text")
    text = clean_text(candidate.get("text", ""))

    return (
        f"Section path: {section}\n"
        f"Block type: {block_type}\n"
        f"Text: {text}"
    )


def estimate_cost(usage: Dict[str, Any]) -> float:
    prompt_tokens = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )

    return (
        prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
        + completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    )


# =============================================================================
# Classifier loading and inference
# =============================================================================

def load_classifier(
    model_dir: Path,
) -> tuple[
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Dict[int, str],
    torch.device,
]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Classifier directory does not exist: {model_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    id2label = resolve_id2label(model)

    print(f"Classifier device: {device}")
    print(f"Classifier labels: {id2label}")

    return tokenizer, model, id2label, device


@torch.inference_mode()
def classify_candidates(
    candidates: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    device: torch.device,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    all_probabilities: List[torch.Tensor] = []

    for candidate_batch in chunks(candidates, CLASSIFIER_BATCH_SIZE):
        texts = [
            format_classifier_input(candidate)
            for candidate in candidate_batch
        ]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=CLASSIFIER_MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # logits = model(**encoded).logits

        # # This assumes the model was trained as a single-label,
        # # four-class classifier with cross-entropy loss.
        # probabilities = torch.softmax(
        #     logits.float(),
        #     dim=-1,
        # ).cpu()

        # all_probabilities.append(probabilities)

        logits = model(**encoded).logits

        # Multi-label classifier:
        # each field is predicted independently.
        probabilities = torch.sigmoid(
            logits.float()
        ).cpu()

        all_probabilities.append(probabilities)

    probability_matrix = torch.cat(
        all_probabilities,
        dim=0,
    ).numpy()

    classified = []

    for candidate, row in zip(candidates, probability_matrix):
        probabilities = {
            id2label[index]: float(row[index])
            for index in range(len(row))
        }

        # classified.append(
        #     {
        #         **candidate,
        #         "classifier_probabilities": probabilities,
        #         "classifier_prediction": max(
        #             probabilities,
        #             key=probabilities.get,
        #         ),
        #     }
        # )

        classified.append(
            {
                **candidate,
                "classifier_probabilities": probabilities,
            }
        )

    return classified


def route_candidates(
    classified_candidates: List[Dict[str, Any]],
) -> Dict[str, List[RoutedBlock]]:
    routed: Dict[str, List[RoutedBlock]] = {
        field: []
        for field in CLASSIFICATION_THRESHOLDS
    }

    for candidate in classified_candidates:
        probabilities = candidate["classifier_probabilities"]

        for field, threshold in CLASSIFICATION_THRESHOLDS.items():
            if probabilities.get(field, 0.0) >= threshold:
                routed[field].append(
                    RoutedBlock(
                        block_id=str(candidate.get("block_id", "")),
                        block_index=int(candidate.get("block_index", 0)),
                        section_path=list(
                            candidate.get("section_path", [])
                        ),
                        text=clean_text(candidate.get("text", "")),
                        retrieval_score=float(
                            candidate.get(
                                "final_score",
                                candidate.get("similarity", 0.0),
                            )
                        ),
                        probabilities=probabilities,
                    )
                )

    # Safety fallback: ensure each field receives a few blocks even when every
    # probability falls below its deployment threshold.
    for field, minimum_count in MIN_BLOCKS_PER_FIELD.items():
        current_ids = {
            block.block_id
            for block in routed[field]
        }

        ranked = sorted(
            classified_candidates,
            key=lambda candidate: candidate[
                "classifier_probabilities"
            ].get(field, 0.0),
            reverse=True,
        )

        for candidate in ranked:
            if len(routed[field]) >= minimum_count:
                break

            candidate_id = str(candidate.get("block_id", ""))

            if candidate_id in current_ids:
                continue

            routed[field].append(
                RoutedBlock(
                    block_id=candidate_id,
                    block_index=int(candidate.get("block_index", 0)),
                    section_path=list(
                        candidate.get("section_path", [])
                    ),
                    text=clean_text(candidate.get("text", "")),
                    retrieval_score=float(
                        candidate.get(
                            "final_score",
                            candidate.get("similarity", 0.0),
                        )
                    ),
                    probabilities=candidate[
                        "classifier_probabilities"
                    ],
                )
            )
            current_ids.add(candidate_id)

        # Restore original article order.
        routed[field].sort(
            key=lambda block: block.block_index
        )

    return routed


# =============================================================================
# Context construction
# =============================================================================

def build_block_context(
    blocks: List[RoutedBlock],
    max_chars: int,
) -> str:
    parts: List[str] = []
    used_chars = 0

    for block in blocks:
        section = " > ".join(block.section_path) or "Unknown section"

        piece = (
            f"[Block {block.block_index}]\n"
            f"Section: {section}\n"
            f"{block.text}"
        )

        separator_chars = 2 if parts else 0
        remaining = max_chars - used_chars - separator_chars

        if remaining <= 0:
            break

        if len(piece) > remaining:
            piece = piece[:remaining].rstrip()

        parts.append(piece)
        used_chars += len(piece) + separator_chars

        if used_chars >= max_chars:
            break

    return "\n\n".join(parts)


def build_field_contexts(
    document: Dict[str, Any],
    routed: Dict[str, List[RoutedBlock]],
) -> Dict[str, str]:
    title = clean_text(document.get("title", ""))
    abstract = clean_text(document.get("abstract", ""))

    keywords_raw = document.get("keywords", [])
    if isinstance(keywords_raw, list):
        keywords = ", ".join(
            clean_text(keyword)
            for keyword in keywords_raw
            if clean_text(keyword)
        )
    else:
        keywords = clean_text(keywords_raw)

    contexts = {}

    condition_body = build_block_context(
        routed["condition_study_type"],
        MAX_CONTEXT_CHARS["condition_study_type"],
    )

    contexts["condition_study_type"] = (
        f"Title:\n{title or 'Not provided'}\n\n"
        f"Abstract:\n{abstract or 'Not provided'}\n\n"
        f"Keywords:\n{keywords or 'Not provided'}\n\n"
        f"Classifier-routed current-study evidence:\n"
        f"{condition_body or 'No body evidence was routed.'}"
    )

    contexts["demographics"] = build_block_context(
        routed["demographics"],
        MAX_CONTEXT_CHARS["demographics"],
    )

    contexts["eligibility"] = build_block_context(
        routed["eligibility"],
        MAX_CONTEXT_CHARS["eligibility"],
    )

    return contexts


# =============================================================================
# OpenRouter
# =============================================================================

def make_response_format(
    field_group: str,
) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{field_group}_extraction",
            "strict": True,
            "schema": FIELD_SCHEMAS[field_group],
        },
    }


def call_openrouter(
    *,
    api_key: str,
    field_group: str,
    context: str,
    model_name: str = OPENROUTER_MODEL,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/NEILCHENHENRI/cohort-x-task-1",
        "X-OpenRouter-Title": "CohortX Task 1 Extraction",
    }

    user_prompt = (
        f"{FIELD_INSTRUCTIONS[field_group]}\n\n"
        "Evidence follows:\n\n"
        f"{context}"
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS[field_group],
        "reasoning": {
            "effort": REASONING_EFFORT[field_group],
            "exclude": True,
        },
        "response_format": make_response_format(field_group),

        # Do not silently route to a provider that cannot enforce the schema.
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": True,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                raise RuntimeError(
                    f"Rate limited: {response.text[:500]}"
                )

            if response.status_code >= 500:
                raise RuntimeError(
                    f"OpenRouter server error "
                    f"{response.status_code}: {response.text[:500]}"
                )

            response.raise_for_status()
            response_json = response.json()

            choices = response_json.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"OpenRouter returned no choices: {response_json}"
                )

            message = choices[0].get("message", {})
            content = message.get("content")

            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                ]
                content = "".join(text_parts)

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    f"OpenRouter returned empty content: {response_json}"
                )

            parsed = json.loads(content)

            metadata = {
                "request_id": response_json.get("id"),
                "model": response_json.get("model", model_name),
                "finish_reason": choices[0].get("finish_reason"),
                "usage": response_json.get("usage", {}),
            }
            metadata["estimated_cost_usd"] = estimate_cost(
                metadata["usage"]
            )

            return parsed, metadata

        except (
            requests.RequestException,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            if attempt == MAX_API_RETRIES:
                break

            delay = min(
                60,
                (2 ** (attempt - 1)) + random.random(),
            )

            print(
                f"\nOpenRouter attempt {attempt} failed for "
                f"{field_group}: {error}"
            )
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)

    raise RuntimeError(
        f"OpenRouter request failed after {MAX_API_RETRIES} attempts "
        f"for {field_group}: {last_error}"
    )


# =============================================================================
# Per-document extraction
# =============================================================================

def serialize_routed_blocks(
    routed: Dict[str, List[RoutedBlock]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        field: [
            {
                "block_id": block.block_id,
                "block_index": block.block_index,
                "section_path": block.section_path,
                "retrieval_score": block.retrieval_score,
                "classifier_probabilities": block.probabilities,
            }
            for block in blocks
        ]
        for field, blocks in routed.items()
    }


def extract_document(
    *,
    candidate_path: Path,
    api_key: str,
    tokenizer: AutoTokenizer,
    classifier: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    device: torch.device,
    model_name: str,
) -> Dict[str, Any]:
    with open(candidate_path, encoding="utf-8") as file:
        document = json.load(file)

    candidates = document.get("candidates", [])

    classified_candidates = classify_candidates(
        candidates=candidates,
        tokenizer=tokenizer,
        model=classifier,
        id2label=id2label,
        device=device,
    )

    routed = route_candidates(classified_candidates)
    contexts = build_field_contexts(document, routed)

    predictions: Dict[str, Any] = {}
    api_metadata: Dict[str, Any] = {}

    for field_group in (
        "condition_study_type",
        "demographics",
        "eligibility",
    ):
        prediction, metadata = call_openrouter(
            api_key=api_key,
            field_group=field_group,
            context=contexts[field_group],
            model_name=model_name,
        )

        predictions[field_group] = prediction
        api_metadata[field_group] = metadata

    total_cost = sum(
        metadata.get("estimated_cost_usd", 0.0)
        for metadata in api_metadata.values()
    )

    return {
        "doc_id": document.get(
            "doc_id",
            candidate_path.stem,
        ),
        "source_candidate_file": candidate_path.name,
        "model": model_name,
        "predictions": predictions,
        "routing": {
            "thresholds": CLASSIFICATION_THRESHOLDS,
            "candidate_count": len(candidates),
            "field_block_counts": {
                field: len(blocks)
                for field, blocks in routed.items()
            },
            "selected_blocks": serialize_routed_blocks(routed),
        },
        "api_metadata": api_metadata,
        "estimated_cost_usd": total_cost,
    }


# =============================================================================
# New helper
# =============================================================================

SUBMISSION_COLUMNS = [
    "conditions",
    "study_type",
    "sex",
    "minimum_age",
    "maximum_age",
    "eligibility_criteria",
    "pmcids",
]

def normalize_string_list(value: Any) -> List[str]:
    """
    Convert a schema-produced list, or a legacy string, into a clean list.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)

    if not text or text.lower() == "not specified":
        return []

    # Compatibility with older semicolon- or bullet-formatted outputs.
    pieces = re.split(
        r"\s*(?:;|\n|\u2022|\*)\s*",
        text,
    )

    return [
        clean_text(piece).lstrip("- ").strip()
        for piece in pieces
        if clean_text(piece).lstrip("- ").strip()
    ]

def format_conditions_for_submission(value: Any) -> str:
    conditions = normalize_string_list(value)

    if not conditions:
        conditions = ["Not Specified"]

    # repr() deliberately produces the Python-list style used by your file.
    return repr(conditions)

def format_criteria_section(
    heading: str,
    criteria: Any,
) -> str:
    items = normalize_string_list(criteria)

    if not items:
        return f"{heading}: * Not Specified."

    cleaned_items = []

    for item in items:
        item = clean_text(item).rstrip(" ;.")
        if item:
            cleaned_items.append(item)

    if not cleaned_items:
        return f"{heading}: * Not Specified."

    return (
        f"{heading}: "
        + "; ".join(
            f"* {item}"
            for item in cleaned_items
        )
        + "."
    )

def combine_eligibility_for_submission(
    inclusion: Any,
    exclusion: Any,
) -> str:
    return (
        f"{format_criteria_section('Inclusion Criteria', inclusion)} "
        f"{format_criteria_section('Exclusion Criteria', exclusion)}"
    )

def normalize_pmcid(value: Any) -> str:
    text = clean_text(value)

    # Handles PMC10542709, pmc10542709, and plain 10542709.
    match = re.search(
        r"(?:PMC)?(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    return match.group(1) if match else text

def result_to_csv_row(
    result: Dict[str, Any],
) -> Dict[str, str]:
    predictions = result.get("predictions", {})

    condition_prediction = predictions.get(
        "condition_study_type",
        {},
    )
    demographics_prediction = predictions.get(
        "demographics",
        {},
    )
    eligibility_prediction = predictions.get(
        "eligibility",
        {},
    )

    return {
        "conditions": format_conditions_for_submission(
            condition_prediction.get(
                "conditions",
                [],
            )
        ),
        "study_type": clean_text(
            condition_prediction.get(
                "study_type",
                "OBSERVATIONAL",
            )
        ),
        "sex": clean_text(
            demographics_prediction.get(
                "sex",
                "Other",
            )
        ),
        "minimum_age": clean_text(
            demographics_prediction.get(
                "minimum_age",
                "Not Specified",
            )
        )
        or "Not Specified",
        "maximum_age": clean_text(
            demographics_prediction.get(
                "maximum_age",
                "Not Specified",
            )
        )
        or "Not Specified",
        "eligibility_criteria": combine_eligibility_for_submission(
            eligibility_prediction.get(
                "inclusion_criteria",
                [],
            ),
            eligibility_prediction.get(
                "exclusion_criteria",
                [],
            ),
        ),
        "pmcids": normalize_pmcid(
            result.get(
                "doc_id",
                result.get(
                    "source_candidate_file",
                    "",
                ),
            )
        ),
    }

def write_submission_csv(
    path: Path,
    rows: List[Dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUBMISSION_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

def pmcid_sort_key(
    row: Dict[str, str],
) -> tuple[int, str]:
    pmcid = clean_text(
        row.get(
            "pmcids",
            "",
        )
    )

    if pmcid.isdigit():
        return int(pmcid), pmcid

    return 10**30, pmcid


# =============================================================================
# Batch runner
# =============================================================================

def run_batch(
    *,
    candidate_dir: Path,
    output_dir: Path,
    classifier_dir: Path,
    model_name: str,
    limit: int | None,
    overwrite: bool,
) -> None:
    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer, classifier, id2label, device = load_classifier(
        classifier_dir
    )

    # candidate_paths = sorted(
    #     path
    #     for path in candidate_dir.glob("*.json")
    #     if not path.name.startswith("_")
    # )

    # if limit is not None:
    #     candidate_paths = candidate_paths[:limit]

    candidate_paths = sorted(
        path
        for path in candidate_dir.glob("*.json")
        if not path.name.startswith("_")
    )

    if limit is not None:
        all_pmcids = sorted(
            normalize_pmcid(path.stem)
            for path in candidate_paths
        )

        rng = np.random.default_rng(RANDOM_SEED)

        selected = set(
            rng.choice(
                all_pmcids,
                size=limit,
                replace=False,
            ).tolist()
        )

        candidate_paths = [
            path
            for path in candidate_paths
            if normalize_pmcid(path.stem) in selected
        ]

        candidate_paths.sort(
            key=lambda path: normalize_pmcid(path.stem)
        )

    if not candidate_paths:
        raise FileNotFoundError(
            f"No candidate JSON files found in {candidate_dir}"
        )

    submission_rows: List[Dict[str, str]] = []

    completed = 0
    skipped = 0
    failed = 0
    run_cost = 0.0

    failures_path = (
        output_dir
        / "failures.jsonl"
    )

    if overwrite:
        failures_path.unlink(
            missing_ok=True
        )

    for candidate_path in tqdm(
        candidate_paths,
        desc="Extracting with Sonnet",
    ):
        output_path = (
            output_dir
            / candidate_path.name
        )

        # Reuse a previously completed per-paper JSON.
        if output_path.exists() and not overwrite:
            try:
                with open(
                    output_path,
                    encoding="utf-8",
                ) as file:
                    existing_result = json.load(
                        file
                    )

                submission_rows.append(
                    result_to_csv_row(
                        existing_result
                    )
                )

                run_cost += float(
                    existing_result.get(
                        "estimated_cost_usd",
                        0.0,
                    )
                )

                skipped += 1

            except Exception as error:
                failed += 1

                append_jsonl(
                    failures_path,
                    {
                        "candidate_file": candidate_path.name,
                        "stage": "read_existing_output",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )

                tqdm.write(
                    f"FAILED reading existing output "
                    f"{output_path.name}: {error}"
                )

            continue

        try:
            result = extract_document(
                candidate_path=candidate_path,
                api_key=api_key,
                tokenizer=tokenizer,
                classifier=classifier,
                id2label=id2label,
                device=device,
                model_name=model_name,
            )

            atomic_write_json(
                output_path,
                result,
            )

            submission_rows.append(
                result_to_csv_row(
                    result
                )
            )

            document_cost = float(
                result.get(
                    "estimated_cost_usd",
                    0.0,
                )
            )

            run_cost += document_cost
            completed += 1

            tqdm.write(
                f"{result['doc_id']}: "
                f"${document_cost:.4f}; "
                f"blocks="
                f"{result['routing']['field_block_counts']}"
            )

        except Exception as error:
            failed += 1

            append_jsonl(
                failures_path,
                {
                    "candidate_file": candidate_path.name,
                    "stage": "extract_document",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

            tqdm.write(
                f"FAILED {candidate_path.name}: {error}"
            )

    submission_rows.sort(
        key=pmcid_sort_key
    )

    submission_csv_path = (
        output_dir
        / "submission_sonnet.csv"
    )

    write_submission_csv(
        submission_csv_path,
        submission_rows,
    )

    print("\nRun summary")
    print(f"  Newly completed : {completed}")
    print(f"  Reused existing : {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Output rows     : {len(submission_rows)}")
    print(f"  Estimated cost  : ${run_cost:.4f}")
    print(f"  Submission CSV  : {submission_csv_path}")
    print(f"  Per-paper JSONs : {output_dir}")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Route unified retrieval candidates with the final classifier "
#             "and extract CohortX fields using Claude Sonnet through OpenRouter."
#         )
#     )

#     parser.add_argument(
#         "--candidate-dir",
#         type=Path,
#         default=UNIFIED_CANDIDATE_DIR,
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=Path,
#         default=OUTPUT_DIR,
#     )
#     parser.add_argument(
#         "--classifier-dir",
#         type=Path,
#         default=CLASSIFIER_DIR,
#     )
#     parser.add_argument(
#         "--model",
#         default=OPENROUTER_MODEL,
#     )
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=None,
#         help="Process only the first N candidate files.",
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#     )

#     return parser.parse_args()


# if __name__ == "__main__":
#     arguments = parse_args()

#     run_batch(
#         candidate_dir=arguments.candidate_dir,
#         output_dir=arguments.output_dir,
#         classifier_dir=arguments.classifier_dir,
#         model_name=arguments.model,
#         limit=arguments.limit,
#         overwrite=arguments.overwrite,
#     )

if __name__ == "__main__":
    run_batch(
        candidate_dir=UNIFIED_CANDIDATE_DIR,
        output_dir=OUTPUT_DIR,
        classifier_dir=CLASSIFIER_DIR,
        model_name=OPENROUTER_MODEL,
        limit=LIMIT,
        overwrite=OVERWRITE,
    )