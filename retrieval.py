from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =============================================================================
# Retrieval configuration
# =============================================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

FIELD_QUERIES = {
    "condition_study_type": [
        "Medical condition, disease, diagnosis, or disorder of the study population.",
        "Study design including randomized controlled trial, prospective, retrospective, observational cohort, case-control, or cross-sectional study.",
    ],
    "demographics": [
        (
            "Participant demographics and age eligibility including minimum age, maximum age, "
            "lower age limit, upper age limit, adults, children, and years of age. "
            "Participant sex or gender eligibility including male, female, men, women, "
            "pregnancy, and childbearing potential."
        ),
    ],
    "eligibility": [
        (
            "Participant eligibility criteria including inclusion criteria, exclusion criteria, "
            "enrollment criteria, participant selection, eligible participants, and excluded participants."
        ),
    ],
}

CANDIDATE_TOP_K = {
    "condition_study_type": 40,
    "demographics": 40,
    "eligibility": 60,
}

SCORE_TARGET = {
    "condition_study_type": 0.8,
    "demographics": 1.5,
    "eligibility": 2.5,
}

MIN_CHARS = {
    "condition_study_type": 0,
    "demographics": 1800,
    "eligibility": 3000,
}

MAX_CHARS = {
    "condition_study_type": 2000,  # body evidence only; title/abstract/keywords added separately
    "demographics": 4500,
    "eligibility": 8000,
}


POSITIVE_PRIORS = {
    "condition_study_type": {
        "study design": 1.08,
        "methods": 1.03,
        "background": 1.02,
    },
    "demographics": {
        "demographics": 1.10,
        "eligibility": 1.08,
        "inclusion": 1.08,
        "exclusion": 1.08,
        "participants": 1.06,
        "participant": 1.06,
        "patients": 1.04,
        "patient": 1.04,
        "study population": 1.04,
        "population": 1.03,
    },
    "eligibility": {
        "eligibility": 1.10,
        "inclusion": 1.10,
        "exclusion": 1.10,
        "participants": 1.06,
        "participant": 1.06,
        "patients": 1.04,
        "patient": 1.04,
        "study population": 1.04,
        "population": 1.03,
    },
}

NEGATIVE_PRIORS = {
    "references": 0.85,
    "acknowledgements": 0.90,
    "funding": 0.90,
    "competing interests": 0.90,
    "author contributions": 0.90,
    "data availability": 0.92,
    "ethics": 0.95,
    "supplementary": 0.95,
    "appendix": 0.95,
}

SECTION_NORMALIZATION = {
    "acknowledgments": "acknowledgements",
    "acknowledgment": "acknowledgements",
    "acknowledgement": "acknowledgements",
    "conflict of interest": "competing interests",
    "conflicts of interest": "competing interests",
    "authors' contributions": "author contributions",
    "author contribution": "author contributions",
    "availability of data and materials": "data availability",
    "data and materials availability": "data availability",
    "ethical approval": "ethics",
    "ethics approval": "ethics",
    "ethics statement": "ethics",
    "ethical considerations": "ethics",
}


# =============================================================================
# Basic utilities
# =============================================================================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_path(section_path: List[str]) -> str:
    path = " > ".join(section_path).lower()
    path = clean_text(path)

    for src, tgt in SECTION_NORMALIZATION.items():
        path = path.replace(src, tgt)

    return path


def get_blocks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "flat_blocks" in doc:
        return doc["flat_blocks"]

    raw_blocks = doc.get("blocks", [])
    if isinstance(raw_blocks, dict):
        return list(raw_blocks.values())

    return raw_blocks


def get_section_path(block: Dict[str, Any], doc: Dict[str, Any]) -> List[str]:
    if "section_path" in block:
        return block["section_path"]

    section_id = block.get("section_id") or block.get("parent_section_id")
    raw_sections = doc.get("sections", {})

    if isinstance(raw_sections, dict):
        section = raw_sections.get(section_id, {})
    else:
        section = {s["section_id"]: s for s in raw_sections}.get(section_id, {})

    return section.get("path", [])


def block_id(block: Dict[str, Any], idx: int) -> str:
    return block.get("block_id") or block.get("id") or f"block_{idx}"


# =============================================================================
# Embedding text construction
# =============================================================================

def make_block_embedding_text(block: Dict[str, Any], doc: Dict[str, Any]) -> str:
    section_path = get_section_path(block, doc)
    block_type = block.get("block_type") or block.get("type") or "text"
    text = clean_text(block.get("text", ""))

    return (
        f"Section path: {' > '.join(section_path)}\n"
        f"Block type: {block_type}\n"
        f"Text: {text}"
    )


def format_block_for_context(block: Dict[str, Any], doc: Dict[str, Any]) -> str:
    section_path = get_section_path(block, doc)
    text = clean_text(block.get("text", ""))

    return (
        f"Section: {' > '.join(section_path)}\n"
        f"{text}\n"
    )


# =============================================================================
# Metadata prior scoring
# =============================================================================

def metadata_multiplier(
    section_path: List[str],
    field_group: str,
) -> float:
    path = normalize_path(section_path)

    positive_prior = max(
        [1.0]
        + [
            weight
            for term, weight in POSITIVE_PRIORS[field_group].items()
            if term in path
        ]
    )

    negative_prior = min(
        [1.0]
        + [
            weight
            for term, weight in NEGATIVE_PRIORS.items()
            if term in path
        ]
    )

    return positive_prior * negative_prior


# =============================================================================
# Embedding cache
# =============================================================================

# def load_or_compute_block_embeddings(
#     doc: Dict[str, Any],
#     parsed_path: Path,
#     model: SentenceTransformer,
#     embedding_cache_dir: Path
# ) -> tuple[List[Dict[str, Any]], np.ndarray]:
#     blocks = get_blocks(doc)

#     cache_path = embedding_cache_dir / f"{parsed_path.stem}.npz"

#     if cache_path.exists():
#         cache = np.load(cache_path, allow_pickle=True)
#         cached_ids = cache["block_ids"].tolist()
#         current_ids = [block_id(b, i) for i, b in enumerate(blocks)]

#         if cached_ids == current_ids:
#             return blocks, cache["embeddings"]

#     texts = [make_block_embedding_text(b, doc) for b in blocks]

#     embeddings = model.encode(
#         texts,
#         batch_size=64,
#         show_progress_bar=True,
#         normalize_embeddings=True,
#     ).astype(np.float32)

#     np.savez_compressed(
#         cache_path,
#         block_ids=np.array([block_id(b, i) for i, b in enumerate(blocks)]),
#         embeddings=embeddings,
#     )

#     return blocks, embeddings

def load_or_compute_block_embeddings(
    doc: Dict[str, Any],
    parsed_path: Path,
    model: SentenceTransformer,
    embedding_cache_dir: Path,
) -> tuple[List[Dict[str, Any]], np.ndarray, bool]:
    blocks = get_blocks(doc)

    if len(blocks) == 0:
        return [], np.empty((0, model.get_embedding_dimension()), dtype=np.float32), False

    cache_path = embedding_cache_dir / f"{parsed_path.stem}.npz"

    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        cached_ids = cache["block_ids"].tolist()
        current_ids = [block_id(b, i) for i, b in enumerate(blocks)]

        if cached_ids == current_ids:
            return blocks, cache["embeddings"], True

    texts = [make_block_embedding_text(b, doc) for b in blocks]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(
        cache_path,
        block_ids=np.array([block_id(b, i) for i, b in enumerate(blocks)]),
        embeddings=embeddings,
    )

    return blocks, embeddings, False


# =============================================================================
# Retrieval
# =============================================================================

def retrieve_candidates(
    doc: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_embeddings: np.ndarray,
    model: SentenceTransformer,
    field_group: str,
) -> List[Dict[str, Any]]:
    queries = FIELD_QUERIES[field_group]

    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # For multiple queries, use max similarity across query prototypes.
    sims = np.max(query_embeddings @ block_embeddings.T, axis=0)

    candidates = []

    for i, block in enumerate(blocks):
        section_path = get_section_path(block, doc)
        multiplier = metadata_multiplier(section_path, field_group)
        final_score = float(sims[i] * multiplier)

        candidates.append(
            {
                "block_id": block_id(block, i),
                "block_index": i,
                "similarity": float(sims[i]),
                "metadata_multiplier": float(multiplier),
                "final_score": final_score,
                "section_path": section_path,
                "block_type": block.get("block_type") or block.get("type") or "text",
                "text": clean_text(block.get("text", "")),
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return candidates[: CANDIDATE_TOP_K[field_group]]


def select_context_blocks(
    candidates: List[Dict[str, Any]],
    field_group: str,
) -> List[Dict[str, Any]]:
    selected = []
    cumulative_score = 0.0
    char_count = 0
    seen_ids = set()

    for cand in candidates:
        if cand["block_id"] in seen_ids:
            continue

        text_piece = (
            f"Section: {' > '.join(cand['section_path'])}\n"
            f"{cand['text']}\n"
        )
        piece_len = len(text_piece)

        if char_count + piece_len > MAX_CHARS[field_group]:
            if selected:
                continue
            text_piece = text_piece[: MAX_CHARS[field_group]]

        selected.append(cand)
        seen_ids.add(cand["block_id"])

        cumulative_score += cand["final_score"]
        char_count += piece_len

        if (
            cumulative_score >= SCORE_TARGET[field_group]
            and char_count >= MIN_CHARS[field_group]
        ):
            break

    return selected


def build_retrieved_body_context(
    selected: List[Dict[str, Any]],
) -> str:
    parts = []

    for cand in selected:
        parts.append(
            f"Section: {' > '.join(cand['section_path'])}\n"
            f"{cand['text']}"
        )

    return "\n\n".join(parts)


def build_condition_study_context(
    doc: Dict[str, Any],
    retrieved_body: str,
) -> str:
    title = clean_text(doc.get("title", ""))
    abstract = clean_text(doc.get("abstract", ""))
    keywords = doc.get("keywords", [])

    if isinstance(keywords, list):
        keywords = ", ".join(clean_text(k) for k in keywords if clean_text(k))
    else:
        keywords = clean_text(str(keywords))

    return (
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Keywords:\n{keywords}\n\n"
        f"Retrieved body evidence:\n{retrieved_body}"
    ).strip()


# def retrieve_for_doc(
#     doc: Dict[str, Any],
#     parsed_path: Path,
#     model: SentenceTransformer,
#     embedding_cache_dir: Path,
# ) -> Dict[str, Any]:
#     blocks, block_embeddings = load_or_compute_block_embeddings(
#         doc=doc,
#         parsed_path=parsed_path,
#         model=model,
#         embedding_cache_dir=embedding_cache_dir,
#     )

#     output = {
#         "doc_id": doc.get("doc_id", parsed_path.stem),
#         "contexts": {},
#         "selected_blocks": {},
#     }

#     for field_group in FIELD_QUERIES:
#         candidates = retrieve_candidates(
#             doc=doc,
#             blocks=blocks,
#             block_embeddings=block_embeddings,
#             model=model,
#             field_group=field_group,
#         )

#         selected = select_context_blocks(candidates, field_group)
#         retrieved_body = build_retrieved_body_context(selected)

#         if field_group == "condition_study_type":
#             context = build_condition_study_context(doc, retrieved_body)
#         else:
#             context = retrieved_body

#         output["contexts"][field_group] = context
#         output["selected_blocks"][field_group] = selected

#     return output

def retrieve_for_doc(
    doc: Dict[str, Any],
    parsed_path: Path,
    model: SentenceTransformer,
    embedding_cache_dir: Path,
) -> tuple[Dict[str, Any], bool]:

    blocks, block_embeddings, cache_hit = load_or_compute_block_embeddings(
        doc=doc,
        parsed_path=parsed_path,
        model=model,
        embedding_cache_dir=embedding_cache_dir,
    )

    output = {
        "doc_id": doc.get("doc_id", parsed_path.stem),
        "contexts": {},
        "selected_blocks": {},
    }

    for field_group in FIELD_QUERIES:
        candidates = retrieve_candidates(
            doc=doc,
            blocks=blocks,
            block_embeddings=block_embeddings,
            model=model,
            field_group=field_group,
        )

        selected = select_context_blocks(candidates, field_group)
        retrieved_body = build_retrieved_body_context(selected)

        if field_group == "condition_study_type":
            context = build_condition_study_context(doc, retrieved_body)
        else:
            context = retrieved_body

        output["contexts"][field_group] = context
        output["selected_blocks"][field_group] = selected

    return output, cache_hit


# =============================================================================
# Batch export
# =============================================================================

# def export_retrieved_contexts(
#     parsed_dir: Path,
#     output_dir: Path,
#     embedding_cache_dir: Path,
#     overwrite: bool = False,
# ):
#     model = SentenceTransformer(EMBEDDING_MODEL)

#     parsed_files = sorted(parsed_dir.glob("*.json"))
#     print(f"Found {len(parsed_files)} parsed JSON files.")

#     for parsed_path in tqdm(parsed_files, desc="Retrieving evidence"):
#         out_path = output_dir / f"{parsed_path.stem}.json"

#         if out_path.exists() and not overwrite:
#             continue

#         try:
#             with open(parsed_path, encoding="utf-8") as f:
#                 doc = json.load(f)

#             retrieved = retrieve_for_doc(
#                 doc=doc,
#                 parsed_path=parsed_path,
#                 model=model,
#                 embedding_cache_dir=embedding_cache_dir
#             )

#             with open(out_path, "w", encoding="utf-8") as f:
#                 json.dump(retrieved, f, ensure_ascii=False, indent=2)

#         except Exception as e:
#             print(f"Failed {parsed_path.name}: {e}")

#     print(f"Finished. Retrieved contexts written to:\n{output_dir}")

def export_retrieved_contexts(
    parsed_dir: Path,
    output_dir: Path,
    embedding_cache_dir: Path,
    overwrite: bool = False,
):
    model = SentenceTransformer(EMBEDDING_MODEL)

    parsed_files = sorted(parsed_dir.glob("*.json"))

    print(f"Found {len(parsed_files)} parsed JSON files.")

    cache_hits = 0
    cache_misses = 0

    for parsed_path in tqdm(
        parsed_files,
        desc="Retrieving evidence",
    ):
        out_path = output_dir / f"{parsed_path.stem}.json"

        if out_path.exists() and not overwrite:
            continue

        try:
            with open(parsed_path, encoding="utf-8") as f:
                doc = json.load(f)

            retrieved, cache_hit = retrieve_for_doc(
                doc=doc,
                parsed_path=parsed_path,
                model=model,
                embedding_cache_dir=embedding_cache_dir,
            )

            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(retrieved, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed {parsed_path.name}: {e}")

    print("\nEmbedding cache")
    print(f"  Hits    : {cache_hits}")
    print(f"  Misses  : {cache_misses}")

    total = cache_hits + cache_misses
    if total:
        print(f"  Hit rate: {cache_hits / total:.1%}")

    print(f"\nFinished. Retrieved contexts written to:\n{output_dir}")



# =============================================================================
# Unified generous retrieval for evidence labeling
# =============================================================================

UNIFIED_QUERIES = [
    (
        "Current study cohort, study population, participants, patients, "
        "enrollment, recruitment, eligibility criteria, inclusion criteria, "
        "exclusion criteria."
    ),
    (
        "Current study design, medical condition, disease, diagnosis, "
        "participant demographics, age eligibility, sex eligibility."
    ),
]

# UNIFIED_RETRIEVAL_CONFIG = {
#     "char_fraction": 0.30,
# }

UNIFIED_RETRIEVAL_CONFIG = {
    "char_fraction": 0.30,
    "min_blocks": 8,
}


def retrieve_unified_candidates(
    doc: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_embeddings: np.ndarray,
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """
    Retrieve the highest-scoring blocks until they cover at least a fixed
    fraction of the article's usable block characters.

    Selection procedure:
    1. Score every non-empty block by maximum semantic similarity across
       the unified query prototypes.
    2. Rank blocks from highest to lowest score.
    3. Add complete blocks until their combined text reaches at least
       char_fraction of the article's usable block text.
    4. Restore selected blocks to their original document order.
    """
    if len(blocks) == 0:
        return []

    query_embeddings = model.encode(
        UNIFIED_QUERIES,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # One relevance score per block: best similarity across query prototypes.
    similarities = np.max(
        query_embeddings @ block_embeddings.T,
        axis=0,
    )

    candidates = []
    total_chars = 0

    for index, block in enumerate(blocks):
        text = clean_text(block.get("text", ""))

        # Empty blocks should not consume the retrieval budget.
        if not text:
            continue

        char_count = len(text)
        total_chars += char_count

        section_path = get_section_path(
            block,
            doc,
        )

        similarity = float(
            similarities[index]
        )

        candidates.append(
            {
                "doc_id": doc.get(
                    "doc_id",
                    "",
                ),
                "block_id": block_id(
                    block,
                    index,
                ),
                "block_index": index,
                "similarity": similarity,

                # Retained for compatibility with existing diagnostics.
                "metadata_multiplier": 1.0,
                "final_score": similarity,

                "section_path": section_path,
                "block_type": (
                    block.get("block_type")
                    or block.get("type")
                    or "text"
                ),
                "text": text,
                "char_count": char_count,
            }
        )

    if not candidates or total_chars == 0:
        return []

    candidates.sort(
        key=lambda candidate: candidate[
            "final_score"
        ],
        reverse=True,
    )

    target_chars = max(
        1,
        int(
            np.ceil(
                UNIFIED_RETRIEVAL_CONFIG[
                    "char_fraction"
                ]
                * total_chars
            )
        ),
    )

    selected = []
    selected_chars = 0

    # for candidate in candidates:
    #     selected.append(candidate)
    #     selected_chars += candidate[
    #         "char_count"
    #     ]

    #     if selected_chars >= target_chars:
    #         break

    for candidate in candidates:
        selected.append(candidate)
        selected_chars += candidate["char_count"]

        if (
            selected_chars >= target_chars
            and len(selected) >= UNIFIED_RETRIEVAL_CONFIG["min_blocks"]
        ):
            break

    # Preserve article order for downstream inspection and context building.
    selected.sort(
        key=lambda candidate: candidate[
            "block_index"
        ]
    )

    return selected


def retrieve_unified_for_doc(
    doc: Dict[str, Any],
    parsed_path: Path,
    model: SentenceTransformer,
    embedding_cache_dir: Path,
) -> tuple[Dict[str, Any], bool]:
    blocks, block_embeddings, cache_hit = load_or_compute_block_embeddings(
        doc=doc,
        parsed_path=parsed_path,
        model=model,
        embedding_cache_dir=embedding_cache_dir,
    )

    candidates = retrieve_unified_candidates(
        doc=doc,
        blocks=blocks,
        block_embeddings=block_embeddings,
        model=model,
    )

    total_chars = sum(len(clean_text(b.get("text", ""))) for b in blocks)
    selected_chars = sum(c.get("char_count", 0) for c in candidates)

    scores = [
        candidate["final_score"]
        for candidate in candidates
    ]

    top_score = max(scores) if scores else 0.0
    lowest_selected_score = (
        min(scores)
        if scores
        else 0.0
    )

    target_chars = int(
        np.ceil(
            UNIFIED_RETRIEVAL_CONFIG[
                "char_fraction"
            ]
            * total_chars
        )
    )

    output = {
        "doc_id": doc.get(
            "doc_id",
            parsed_path.stem,
        ),
        "title": doc.get(
            "title",
            "",
        ),
        "abstract": doc.get(
            "abstract",
            "",
        ),
        "keywords": doc.get(
            "keywords",
            [],
        ),
        "num_blocks": len(blocks),
        "num_candidates": len(candidates),
        "total_chars": total_chars,
        "target_chars": target_chars,
        "selected_chars": selected_chars,
        "selected_char_fraction": (
            selected_chars / total_chars
            if total_chars
            else 0.0
        ),
        "retrieval_config": (
            UNIFIED_RETRIEVAL_CONFIG
        ),
        "queries": UNIFIED_QUERIES,
        "candidates": candidates,
        "top_score": top_score,
        "lowest_selected_score": (
            lowest_selected_score
        ),
    }

    if len(blocks) == 0:
        output["warning"] = "No retrievable blocks found."

    return output, cache_hit


def export_unified_candidates(
    parsed_dir: Path,
    output_dir: Path,
    embedding_cache_dir: Path,
    overwrite: bool = False,
):
    """
    Export one generous candidate JSON per paper.

    These files are intended for the next evidence-labeling stage.
    Also writes _retrieval_stats.csv with run-level statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)

    parsed_files = sorted(parsed_dir.glob("*.json"))

    print(f"Found {len(parsed_files)} parsed JSON files.")

    cache_hits = 0
    cache_misses = 0
    skipped_outputs = 0
    failures = 0
    run_stats = []

    for parsed_path in tqdm(parsed_files, desc="Unified generous retrieval"):
        out_path = output_dir / f"{parsed_path.stem}.json"

        if out_path.exists() and not overwrite:
            skipped_outputs += 1

            # Still collect stats from existing output if available.
            try:
                with open(out_path, encoding="utf-8") as f:
                    existing = json.load(f)

                run_stats.append({
                    "doc_id": existing.get("doc_id", parsed_path.stem),
                    "num_blocks": existing.get("num_blocks", 0),
                    "num_candidates": existing.get("num_candidates", 0),
                    "total_chars": existing.get("total_chars", 0),
                    "selected_chars": existing.get("selected_chars", 0),
                    "selected_char_fraction": existing.get("selected_char_fraction", 0.0),
                    "top_score": existing.get("top_score", 0.0),
                    "target_chars": existing.get(
                        "target_chars",
                        0,
                    ),
                    "lowest_selected_score": existing.get(
                        "lowest_selected_score",
                        0.0,
                    ),
                    "from_existing_output": True,
                })
            except Exception:
                pass

            continue

        try:
            with open(parsed_path, encoding="utf-8") as f:
                doc = json.load(f)

            retrieved, cache_hit = retrieve_unified_for_doc(
                doc=doc,
                parsed_path=parsed_path,
                model=model,
                embedding_cache_dir=embedding_cache_dir,
            )

            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            run_stats.append({
                "doc_id": retrieved.get("doc_id", parsed_path.stem),
                "num_blocks": retrieved.get("num_blocks", 0),
                "num_candidates": retrieved.get("num_candidates", 0),
                "total_chars": retrieved.get("total_chars", 0),
                "selected_chars": retrieved.get("selected_chars", 0),
                "selected_char_fraction": retrieved.get("selected_char_fraction", 0.0),
                "top_score": retrieved.get("top_score", 0.0),
                "target_chars": retrieved.get(
                    "target_chars",
                    0,
                ),
                "lowest_selected_score": retrieved.get(
                    "lowest_selected_score",
                    0.0,
                ),
                "from_existing_output": False,
            })

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(retrieved, f, ensure_ascii=False, indent=2)

        except Exception as e:
            failures += 1
            print(f"Failed {parsed_path.name}: {e}")

    print("\nEmbedding cache for newly processed files")
    print(f"  Hits    : {cache_hits}")
    print(f"  Misses  : {cache_misses}")

    total_cache_checked = cache_hits + cache_misses
    if total_cache_checked:
        print(f"  Hit rate: {cache_hits / total_cache_checked:.1%}")

    print("\nOutput files")
    print(f"  Skipped existing outputs : {skipped_outputs}")
    print(f"  Newly processed outputs  : {total_cache_checked}")
    print(f"  Failures                 : {failures}")

    if run_stats:
        df = pd.DataFrame(run_stats)

        print("\nUnified retrieval stats")
        print(f"  Papers with stats         : {len(df)}")
        print(f"  Zero-block papers         : {(df['num_blocks'] == 0).sum()}")
        print(f"  Zero-candidate papers     : {(df['num_candidates'] == 0).sum()}")

        print("\nCandidate count")
        print(f"  Mean                      : {df['num_candidates'].mean():.2f}")
        print(f"  Median                    : {df['num_candidates'].median():.2f}")
        print(f"  Min / Max                 : {df['num_candidates'].min()} / {df['num_candidates'].max()}")

        print("\nSelected character count")
        print(f"  Mean                      : {df['selected_chars'].mean():.0f}")
        print(f"  Median                    : {df['selected_chars'].median():.0f}")
        print(f"  Min / Max                 : {df['selected_chars'].min()} / {df['selected_chars'].max()}")

        print("\nSelected character fraction")
        print(f"  Mean                      : {df['selected_char_fraction'].mean():.3f}")
        print(f"  Median                    : {df['selected_char_fraction'].median():.3f}")
        print(
            f"  Min / Max                 : "
            f"{df['selected_char_fraction'].min():.3f} / "
            f"{df['selected_char_fraction'].max():.3f}"
        )

        print("\nScore stats")
        print(f"  Mean top score            : {df['top_score'].mean():.4f}")
        print(f"  Median top score          : {df['top_score'].median():.4f}")

        stats_path = output_dir / "_retrieval_stats.csv"
        df.to_csv(stats_path, index=False)
        print(f"\nSaved retrieval stats to:\n{stats_path}")

    print(f"\nFinished. Unified candidates written to:\n{output_dir}")


def inspect_unified_candidates(
    pmcid: str,
    candidate_dir: Path,
    n: int = 20,
):
    pid = str(pmcid).strip().replace("PMC", "")
    path = candidate_dir / f"{pid}.json"

    if not path.exists():
        path = candidate_dir / f"PMC{pid}.json"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("Document:", data["doc_id"])
    print("Num blocks:", data.get("num_blocks"))
    print("Num candidates:", data.get("num_candidates"))
    print("=" * 80)

    for cand in data["candidates"][:n]:
        print(
            f"\nscore={cand['final_score']:.4f} "
            f"sim={cand['similarity']:.4f} "
            f"mult={cand['metadata_multiplier']:.3f}"
        )
        print("Section:", " > ".join(cand["section_path"]))
        print(cand["text"][:600] + ("..." if len(cand["text"]) > 600 else ""))