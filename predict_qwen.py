"""
CohortX Task 1 — Transformers Inference (Qwen3-4B, 8-bit)
===========================================================
Uses Qwen3-4B through Transformers in 8-bit mode while reusing the
existing MiniLM section filtering and prompt.

Usage:
  python predict_qwen.py \
    --data_dir  ~/Downloads/cohort-x-task-1 \
    --nxml_dir  ~/Downloads/cohort-x-task-1/PMC_NXML_Archives \
    [--model    Qwen/Qwen3-4B] \
    [--test]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import ELIGIBILITY_QUERY
from parser import NXMLParser

# ---------------------------------------------------------------------------
# Section filter (MiniLM)
# ---------------------------------------------------------------------------

THRESHOLD = 0.20
MAX_CHARS_PER_SEC = 600
MAX_TOTAL_CHARS = 5000

_ranker = None
_query_emb = None


def _get_ranker():
    global _ranker, _query_emb
    if _ranker is None:
        from sentence_transformers import SentenceTransformer
        _ranker = SentenceTransformer("all-MiniLM-L6-v2")
        _query_emb = _ranker.encode(ELIGIBILITY_QUERY, convert_to_tensor=True)
    return _ranker, _query_emb


def get_relevant_sections(parsed: dict) -> str:
    from sentence_transformers import util

    ranker, query_emb = _get_ranker()
    scored = []

    for sec in parsed.get("sections", []):
        title_emb = ranker.encode(sec["title"], convert_to_tensor=True)
        text_emb = ranker.encode(sec["text"][:200], convert_to_tensor=True)
        score = max(
            float(util.cos_sim(query_emb, title_emb).item()),
            float(util.cos_sim(query_emb, text_emb).item()),
        )
        if score >= THRESHOLD:
            scored.append((score, sec))

    scored.sort(key=lambda x: x[0], reverse=True)

    body = ""
    for _, sec in scored:
        chunk = (
            f"\n{sec['title']}\n" if sec["title"] else ""
        ) + sec["text"][:MAX_CHARS_PER_SEC] + "\n"

        if len(body) + len(chunk) > MAX_TOTAL_CHARS:
            break
        body += chunk

    return body


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(parsed: dict) -> str:
    title = parsed.get("title", "")
    abstract = parsed.get("abstract", "")
    keywords = ", ".join(parsed.get("keywords", []))
    body = get_relevant_sections(parsed)

    article = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Keywords: {keywords}\n\n"
        f"Relevant sections:\n{body}"
    )

    return f"""You are a biomedical information extraction system. Read the following article and extract the requested fields.

Article:
{article}

Extract the following fields and return ONLY valid JSON, no other text:
{{
  "conditions": ["list of primary medical conditions studied"],
  "study_type": "INTERVENTIONAL or OBSERVATIONAL",
  "sex": "ALL or MALE or FEMALE",
  "minimum_age": "<minimum age with unit, or Not Specified in some very rare cases",
  "maximum_age": "<maximum age with unit, or Not Specified in a few cases"
  "eligibility_criteria": "Inclusion Criteria: [criteria, each prefixed with *]; Exclusion Criteria: [criteria, each prefixed with *]"
}}"""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def extract(pid: str, parsed: dict, model, tokenizer) -> dict:
    if not parsed:
        return {}

    try:
        messages = [{"role": "user", "content": build_prompt(parsed)}]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        raw = re.sub(r"[\x00-\x1f\x7f](?<![\n\t])", "", raw)
        raw = re.sub(r"\n\s*", " ", raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                return json.loads(match.group()) if match else {}
            except Exception:
                return {}

    except Exception as e:
        print(f"\nError on PMC{pid}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--nxml_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--output", default="submission_qwen.xlsx")
    p.add_argument("--sheet", default="Train")
    p.add_argument(
        "--test",
        action="store_true",
        help="Run on 3 random samples only",
    )
    return p.parse_args()


def main():
    import random

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    args = parse_args()
    data_dir = Path(args.data_dir)
    save_path = data_dir / args.output

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    df = pd.read_excel(data_dir / "Task_1.xlsx", sheet_name=args.sheet)
    pmcids = df["pmcids"].astype(str).str.strip().tolist()

    if args.test:
        pmcids = random.sample(pmcids, 3)

    parser = NXMLParser()
    parsed_map = {}

    for pid in tqdm(pmcids, desc="Parsing"):
        path = Path(args.nxml_dir) / f"PMC{pid}.nxml"
        if path.exists():
            parsed_map[pid] = parser.parse(str(path))

    if not args.test and save_path.exists():
        existing_df = pd.read_excel(save_path)
        done_pids = set(existing_df["pmcids"].astype(str).str.strip())
        rows = existing_df.to_dict("records")
        pmcids = [p for p in pmcids if p not in done_pids]
        print(f"Resuming: {len(done_pids)} done, {len(pmcids)} remaining.")
    else:
        rows = []

    pbar = tqdm(pmcids, desc=f"Extracting [{args.model}]")

    for pid in pbar:
        pred = extract(pid, parsed_map.get(pid, {}), model, tokenizer)

        conds = pred.get("conditions", [])
        pred["conditions"] = (
            ", ".join(conds)
            if isinstance(conds, list)
            else str(conds or "")
        )

        rows.append({"pmcids": pid, **pred})

        if args.test:
            gold_row = df[df["pmcids"].astype(str).str.strip() == pid]

            print(f"\nPMC{pid}")
            for k, v in pred.items():
                print(f"  {k:25s}: {str(v)[:120]}")

            if not gold_row.empty:
                g = gold_row.iloc[0]
                print("  --- GOLD ---")
                for field in [
                    "conditions",
                    "study_type",
                    "sex",
                    "minimum_age",
                    "maximum_age",
                ]:
                    print(f"  {field:25s}: {g[field]}")

        elif len(rows) % 50 == 0:
            pd.DataFrame(rows).to_excel(save_path, index=False)
            pbar.set_postfix({"saved": len(rows)})

    if not args.test:
        pd.DataFrame(rows).to_excel(save_path, index=False)
        print(f"Saved {len(rows)} rows → {save_path}")


if __name__ == "__main__":
    main()