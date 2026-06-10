"""
CohortX Task 1 — Model Classes
================================
All training components and the CohortXPipeline.
Constants are in config.py; NXMLParser is in parser.py.
"""

import ast
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    AutoTokenizer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    Trainer, TrainingArguments, pipeline,
    default_data_collator,
)

from config import (
    AGE_MAX_CONTEXT, AGE_QUESTIONS,
    BIOMEDBERT_NAME, BIOBERT_QA_NAME, DISEASE_NER_NAME,
    DISTILBERT_NAME, ELIGIBILITY_QUERY, MINILM_NAME,
    SCIFIVE_CONDITIONS_NAME, SCIFIVE_MAX_INPUT, SCIFIVE_MAX_OUTPUT,
    SCIFIVE_NAME, STAGE1_TOP_K,
)
from parser import NXMLParser

warnings.filterwarnings("ignore")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BiomedBERT Embedder  (shared: stage-1 training + conditions re-ranker)
# ---------------------------------------------------------------------------

class BiomedBERTEmbedder:
    def __init__(self, model_name: str = BIOMEDBERT_NAME):
        log.info(f"Loading BiomedBERT embedder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def embed(self, texts: list, batch_size: int = 16) -> np.ndarray:
        from tqdm import tqdm
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Embedding", unit="batch", leave=False):
            batch  = texts[i: i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                    max_length=512, padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Stage 1 — Fine-tuned MiniLM section ranker
# ---------------------------------------------------------------------------

class Stage1Ranker:
    def __init__(self, model_name: str = MINILM_NAME):
        from sentence_transformers import SentenceTransformer
        self.model     = SentenceTransformer(model_name)
        self._query_emb = None

    def _encode_query(self):
        if self._query_emb is None:
            self._query_emb = self.model.encode(
                ELIGIBILITY_QUERY, convert_to_tensor=True
            )
        return self._query_emb

    def build_training_examples(self, parsed_docs: list, ground_truths: list,
                                 embedder: BiomedBERTEmbedder) -> list:
        from sentence_transformers import InputExample

        # Collect all texts to embed in one pass
        # Structure: [(doc_idx, "gt"/"sec", text), ...]
        records = []   # (doc_idx, gt, section_texts) for valid docs
        all_texts = []
        text_index = {}   # doc_idx -> {"gt": int, "secs": [int, ...]}

        for doc_idx, (parsed, gt) in enumerate(zip(parsed_docs, ground_truths)):
            if not gt or gt.upper() in ("NOT SPECIFIED", "NAN"):
                continue
            if not parsed:
                continue
            sections = [s for s in parsed.get("sections", []) if s["text"].strip()]
            if len(sections) < 2:
                continue
            sec_texts = [(s["title"] + " " + s["text"])[:500] for s in sections]
            text_index[doc_idx] = {
                "gt":   len(all_texts),
                "secs": list(range(len(all_texts) + 1,
                                   len(all_texts) + 1 + len(sec_texts))),
                "sec_texts": sec_texts,
                "gt_text":   gt,
            }
            all_texts.append(gt)
            all_texts.extend(sec_texts)

        if not all_texts:
            return []

        log.info(f"Embedding {len(all_texts)} texts in one batch...")
        all_embs = embedder.embed(all_texts, batch_size=32)

        examples = []
        for doc_idx, info in text_index.items():
            gt_emb   = all_embs[info["gt"]]
            sec_embs = all_embs[info["secs"][0]: info["secs"][-1] + 1]
            scores   = cosine_similarity([gt_emb], sec_embs)[0]
            best     = int(np.argmax(scores))
            sec_texts = info["sec_texts"]
            negs     = [sec_texts[i] for i in range(len(sec_texts)) if i != best][:4]
            if negs:
                examples.append(InputExample(
                    texts=[ELIGIBILITY_QUERY, sec_texts[best]] + negs
                ))

        log.info(f"Stage-1 training examples: {len(examples)}")
        return examples

    def fine_tune(self, examples: list, output_path: str,
                  epochs: int = 3, batch_size: int = 8, use_gpu: bool = False):
        from sentence_transformers import losses
        from torch.utils.data import DataLoader
        if not use_gpu:
            import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss   = losses.MultipleNegativesRankingLoss(self.model)
        self.model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=max(1, len(loader) // 10),
            output_path=output_path,
            show_progress_bar=True,
        )
        self._query_emb = None
        log.info(f"Stage-1 ranker saved → {output_path}")

    def get_candidates(self, sections: list, top_k: int = STAGE1_TOP_K) -> list:
        from sentence_transformers import util as st_util
        query_emb  = self._encode_query()
        candidates = []
        for sec in sections:
            if not sec["text"].strip():
                continue
            snippet = (sec["title"] + " " + sec["text"][:300]).strip()
            score   = float(st_util.cos_sim(
                query_emb,
                self.model.encode(snippet, convert_to_tensor=True)
            ).item())
            candidates.append((score, sec["text"]))
            if not sec["title"].strip():
                for para in sec.get("paragraphs", []):
                    if len(para.split()) < 15:
                        continue
                    p_score = float(st_util.cos_sim(
                        query_emb,
                        self.model.encode(para[:300], convert_to_tensor=True)
                    ).item())
                    candidates.append((p_score, para))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in candidates[:top_k]]

    @classmethod
    def load(cls, path: str) -> "Stage1Ranker":
        from sentence_transformers import SentenceTransformer
        obj            = cls.__new__(cls)
        obj.model      = SentenceTransformer(path)
        obj._query_emb = None
        return obj


# ---------------------------------------------------------------------------
# Stage 2 — SciFive seq2seq generator  (eligibility_criteria)
# ---------------------------------------------------------------------------

class SciFiveGenerator:
    """
    Fine-tuned SciFive-base-Pubmed for targeted summarization.
    Input:  candidate section text (from stage-1)
    Output: eligibility criteria text
    Training: seq2seq on (candidate_text → ground_truth_eligibility) pairs.
    """

    def __init__(self, model_name: str = SCIFIVE_NAME):
        self.model_name = model_name
        self.tokenizer  = None
        self.model      = None
        self.trained    = False

    def _build_pairs(self, parsed_docs: list, ground_truths: list,
                     ranker: Stage1Ranker) -> tuple:
        inputs, targets = [], []
        for parsed, gt in zip(parsed_docs, ground_truths):
            if not parsed or not gt or gt.upper() in ("NOT SPECIFIED", "NAN"):
                continue
            candidates = ranker.get_candidates(parsed.get("sections", []))
            src        = " ".join(candidates).strip()
            if src:
                inputs.append(src)
                targets.append(gt)
        log.info(f"SciFive training pairs: {len(inputs)}")
        return inputs, targets

    def fine_tune(self, parsed_docs: list, ground_truths: list,
                  ranker: Stage1Ranker, output_dir: str,
                  epochs: int = 10, batch_size: int = 4,
                  lr: float = 1e-4, use_gpu: bool = False):
        from datasets import Dataset as HFDataset

        inputs, targets = self._build_pairs(parsed_docs, ground_truths, ranker)
        if len(inputs) < 5:
            log.error("Not enough SciFive training pairs.")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        def tokenize(batch):
            model_inputs = self.tokenizer(
                batch["input"], max_length=SCIFIVE_MAX_INPUT,
                truncation=True, padding="max_length"
            )
            labels = self.tokenizer(
                text_target=batch["target"],
                max_length=SCIFIVE_MAX_OUTPUT,
                truncation=True, padding="max_length"
            )
            # Replace tokenizer pad token id in labels with -100 (ignored in loss)
            label_ids = [
                [-100 if t == self.tokenizer.pad_token_id else t for t in ids]
                for ids in labels["input_ids"]
            ]
            model_inputs["labels"] = label_ids
            return model_inputs

        ds    = HFDataset.from_dict({"input": inputs, "target": targets})
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"].map(tokenize, batched=True,
                                      remove_columns=["input", "target"])
        eval_ds  = split["test"].map(tokenize,  batched=True,
                                     remove_columns=["input", "target"])

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=SCIFIVE_MAX_OUTPUT,
            load_best_model_at_end=True,
            use_cpu=not use_gpu,
            seed=42,
            logging_steps=10,
        )

        Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True
            ),
        ).train()

        self.trained = True
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        log.info(f"SciFive saved → {output_dir}")

    def generate(self, candidate_texts: list) -> str:
        if not self.trained or self.model is None:
            return " ".join(candidate_texts).strip() or "Not Specified"

        src     = " ".join(candidate_texts).strip()
        inputs  = self.tokenizer(
            src, return_tensors="pt",
            max_length=SCIFIVE_MAX_INPUT, truncation=True
        )
        inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=SCIFIVE_MAX_OUTPUT,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
            )
        out = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return out or "Not Specified"

    @classmethod
    def load(cls, path: str) -> "SciFiveGenerator":
        obj           = cls.__new__(cls)
        obj.model_name = path
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model     = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        obj.model.eval()
        obj.trained   = True
        return obj


# ---------------------------------------------------------------------------
# Eligibility Extractor  (stage 1 + stage 2)
# ---------------------------------------------------------------------------

class EligibilityExtractor:
    def __init__(self, ranker: Stage1Ranker, generator: SciFiveGenerator):
        self.ranker    = ranker
        self.generator = generator

    def extract(self, sections: list) -> str:
        candidates = self.ranker.get_candidates(sections)
        return self.generator.generate(candidates)


# ---------------------------------------------------------------------------
# BERT Classifier  (study_type and sex)
# ---------------------------------------------------------------------------

class BERTClassifier:
    def __init__(self, model_name: str = DISTILBERT_NAME):
        self.model_name = model_name
        self.tokenizer  = None
        self.model      = None
        self.label2id   = {}
        self.id2label   = {}
        self.trained    = False

    def _tokenize_fn(self, examples):
        return self.tokenizer(examples["text"], truncation=True,
                              max_length=512, padding="max_length")

    def fit(self, texts: list, labels: list, output_dir: str,
            epochs: int = 5, batch_size: int = 8,
            lr: float = 2e-5, use_gpu: bool = False):
        from datasets import Dataset as HFDataset

        unique        = sorted(set(labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(unique),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        int_labels = [self.label2id[l] for l in labels]
        ds         = HFDataset.from_dict({"text": texts, "label": int_labels})
        split      = ds.train_test_split(test_size=0.1, seed=42)
        train_ds   = split["train"].map(self._tokenize_fn, batched=True)
        eval_ds    = split["test"].map(self._tokenize_fn,  batched=True)

        def compute_metrics(ep):
            preds = np.argmax(ep.predictions, axis=-1)
            return {
                "accuracy": accuracy_score(ep.label_ids, preds),
                "f1":       f1_score(ep.label_ids, preds, average="weighted"),
            }

        Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=lr,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                use_cpu=not use_gpu,
                seed=42,
                logging_steps=10,
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        ).train()

        self.trained = True
        self.save(output_dir)

    def predict(self, text: str) -> str:
        if not self.trained:
            return "Not Specified"
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return self.id2label[logits.argmax(-1).item()]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(f"{path}/label_mapping.json", "w") as f:
            json.dump({"label2id": self.label2id,
                       "id2label": {str(k): v for k, v in self.id2label.items()}}, f)
        log.info(f"BERTClassifier saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BERTClassifier":
        obj           = cls()
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        with open(f"{path}/label_mapping.json") as f:
            m = json.load(f)
        obj.label2id  = m["label2id"]
        obj.id2label  = {int(k): v for k, v in m["id2label"].items()}
        obj.model     = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        obj.model.eval()
        obj.trained   = True
        return obj


# ---------------------------------------------------------------------------
# Conditions Extractor  (NER + BiomedBERT re-ranker)
# ---------------------------------------------------------------------------

class ConditionsExtractor:
    """
    Two-stage conditions extraction:
      Stage 1: NER (pruas/BENT-PubMedBERT-NER-Disease) extracts candidate
               disease entities from title + abstract + keywords.
      Stage 2: SciFive fine-tuned selector reads the abstract and the full
               candidate list and selects which entities are the primary
               study conditions. Constraining to a candidate list makes this
               a selection problem rather than open-ended generation —
               much better suited to small models on limited data.
    """

    def __init__(self, model_name: str = SCIFIVE_CONDITIONS_NAME):
        self.model_name = model_name
        self.ner        = None
        self.tokenizer  = None
        self.model      = None
        self.trained    = False
        self._load_ner()

    def _load_ner(self):
        try:
            self.ner = pipeline("ner", model=DISEASE_NER_NAME,
                                aggregation_strategy="simple",
                                device=PIPELINE_DEVICE)
            log.info(f"Loaded NER: {DISEASE_NER_NAME}")
        except Exception as e:
            log.warning(f"NER not loaded: {e}")

    def _run_ner(self, text: str) -> list:
        if not self.ner or not text.strip():
            return []
        try:
            results  = self.ner(text[:512])
            entities, current = [], []
            for r in results:
                tag = r.get("entity_group", "")
                if tag == "B":
                    if current:
                        entities.append(" ".join(current))
                    current = [r["word"].strip()]
                elif tag == "I" and current:
                    current.append(r["word"].strip())
                elif tag == "DISEASE":
                    entities.append(r["word"].strip())
            if current:
                entities.append(" ".join(current))
            return [e for e in entities if len(e) > 2]
        except Exception:
            return []

    def get_candidates(self, parsed: dict) -> list:
        """Return deduplicated disease entities from title + abstract + keywords."""
        sources = [
            parsed.get("title", ""),
            parsed.get("abstract", ""),
            " ".join(parsed.get("keywords", [])),
        ]
        seen, out = set(), []
        for text in sources:
            for ent in self._run_ner(text):
                if ent.lower() not in seen:
                    seen.add(ent.lower())
                    out.append(ent)
        return out

    def _build_input(self, abstract: str, candidates: list) -> str:
        cand_str = ", ".join(candidates)
        return (
            f"Abstract: {abstract[:400]}\n\n"
            f"Candidates: {cand_str}\n\n"
            f"Select the primary medical conditions being studied:"
        )

    def fine_tune(self, train_records: list, output_dir: str,
                  epochs: int = 10, batch_size: int = 4,
                  lr: float = 1e-4, use_gpu: bool = False):
        """
        Fine-tune SciFive on (abstract + candidates → selected conditions) pairs.
        Target is ground-truth conditions joined by ", ".
        """
        from datasets import Dataset as HFDataset

        inputs, targets = [], []
        for rec in train_records:
            parsed = rec["parsed"]
            gt     = rec["ground_truth"]
            cands  = self.get_candidates(parsed)
            if not cands or not gt:
                continue
            inp = self._build_input(parsed.get("abstract", ""), cands)
            tgt = ", ".join(gt)
            inputs.append(inp)
            targets.append(tgt)

        log.info(f"Conditions selector training pairs: {len(inputs)}")
        if len(inputs) < 5:
            log.error("Not enough conditions training pairs.")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        def tokenize(batch):
            model_inputs = self.tokenizer(
                batch["input"], max_length=SCIFIVE_MAX_INPUT,
                truncation=True, padding="max_length"
            )
            labels = self.tokenizer(
                text_target=batch["target"],
                max_length=64,      # conditions output is short
                truncation=True, padding="max_length"
            )
            label_ids = [
                [-100 if t == self.tokenizer.pad_token_id else t for t in ids]
                for ids in labels["input_ids"]
            ]
            model_inputs["labels"] = label_ids
            return model_inputs

        ds       = HFDataset.from_dict({"input": inputs, "target": targets})
        split    = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"].map(tokenize, batched=True,
                                      remove_columns=["input", "target"])
        eval_ds  = split["test"].map(tokenize,  batched=True,
                                     remove_columns=["input", "target"])

        Seq2SeqTrainer(
            model=self.model,
            args=Seq2SeqTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=lr,
                weight_decay=0.1,
                lr_scheduler_type="cosine",
                warmup_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                predict_with_generate=True,
                generation_max_length=64,
                load_best_model_at_end=True,
                use_cpu=not use_gpu,
                seed=42,
                logging_steps=10,
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        ).train()

        self.trained = True
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        log.info(f"Conditions selector saved → {output_dir}")

    def extract(self, parsed: dict) -> list:
        cands = self.get_candidates(parsed)
        if not cands:
            return ["Not Specified"]

        if self.trained and self.model is not None:
            inp    = self._build_input(parsed.get("abstract", ""), cands)
            inputs = self.tokenizer(inp, return_tensors="pt",
                                    max_length=SCIFIVE_MAX_INPUT, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                )
            output = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            # Parse comma-separated output and match back to candidates
            selected = [s.strip() for s in output.split(",") if s.strip()]
            # Keep only selections that fuzzy-match a known candidate
            cands_lower = {c.lower(): c for c in cands}
            matched = []
            for sel in selected:
                sel_lower = sel.lower()
                # exact match first
                if sel_lower in cands_lower:
                    matched.append(cands_lower[sel_lower])
                else:
                    # partial token overlap fallback
                    sel_toks = set(sel_lower.split())
                    for cand_lower, cand_orig in cands_lower.items():
                        cand_toks = set(cand_lower.split())
                        if sel_toks & cand_toks:
                            matched.append(cand_orig)
                            break
            return matched if matched else selected if selected else ["Not Specified"]

        # Fallback: return all candidates (no selector trained)
        return cands if cands else ["Not Specified"]

    def save(self, path: str):
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "ConditionsExtractor":
        obj            = cls()          # loads NER in __init__
        obj.model_name = path
        obj.tokenizer  = AutoTokenizer.from_pretrained(path)
        obj.model      = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        obj.model.eval()
        obj.trained    = True
        return obj


# ---------------------------------------------------------------------------
# Age Extractor  (fine-tuned BioBERT QA)
# ---------------------------------------------------------------------------

def _normalize_age(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        try:    return f"{int(float(m.group(1)))} Years"
        except: pass
    return text.strip()


class AgeExtractor:
    """
    Fine-tunes BioBERT-SQuAD2 on age span examples constructed from training labels.
    Span supervision is feasible for age because the value (e.g. "18") appears
    literally in the candidate text and can be located with a string search.
    Falls back to regex when QA confidence is below a tuned threshold.
    """

    def __init__(self, model_name: str = BIOBERT_QA_NAME):
        self.model_name    = model_name
        self.tokenizer     = AutoTokenizer.from_pretrained(model_name)
        self.model         = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.min_threshold = 0.35
        self.max_threshold = 0.35

    def _run_qa(self, question: str, context: str) -> dict:
        """Run extractive QA directly without pipeline abstraction."""
        self.model.to(DEVICE)
        inputs = self.tokenizer(
            question, context,
            return_tensors="pt",
            truncation="only_second",
            max_length=AGE_MAX_CONTEXT,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        start = outputs.start_logits.argmax(-1).item()
        end   = outputs.end_logits.argmax(-1).item() + 1
        # Compute a confidence proxy: softmax probability of best span
        import torch.nn.functional as F
        start_prob = F.softmax(outputs.start_logits, dim=-1)[0, start].item()
        end_prob   = F.softmax(outputs.end_logits,   dim=-1)[0, end - 1].item()
        score      = (start_prob + end_prob) / 2.0
        tokens = inputs["input_ids"][0][start:end]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {"answer": answer, "score": score}

    # ------------------------------------------------------------------
    # Span construction from training labels
    # ------------------------------------------------------------------

    def _build_squad_examples(self, records: list) -> list:
        """
        For each training record, locate the gold age value as a character
        span in the candidate text and return SQuAD-format dicts.
        records: list of {candidate_text, gold_min, gold_max}
        """
        from datasets import Dataset as HFDataset
        examples = []

        for rec in records:
            ctx = rec["candidate_text"]
            if not ctx.strip():
                continue
            for bound in ("minimum", "maximum"):
                gold = rec.get(f"gold_{bound[:3]}", "").strip()
                if not gold or gold.upper() in ("NAN", "NOT SPECIFIED", ""):
                    continue
                # Extract the numeric part (e.g. "18" from "18 Years")
                num_match = re.search(r"(\d+)", gold)
                if not num_match:
                    continue
                num_str = num_match.group(1)
                # Find the number in the context (look for it as a standalone token)
                ctx_lower = ctx.lower()
                for m in re.finditer(r"\b" + re.escape(num_str) + r"\b", ctx_lower):
                    start = m.start()
                    end   = m.end()
                    # Verify the surrounding text looks like an age mention
                    window = ctx_lower[max(0, start - 30): end + 30]
                    if any(kw in window for kw in ["year", "yr", "age", "old"]):
                        examples.append({
                            "question":    AGE_QUESTIONS[bound],
                            "context":     ctx,
                            "answer_text": ctx[start:end],
                            "start_pos":   start,
                        })
                        break   # take first valid span per (record, bound)

        log.info(f"Age span examples: {len(examples)}")
        return examples

    def fine_tune(self, records: list, output_dir: str,
                  epochs: int = 3, batch_size: int = 8, use_gpu: bool = False):
        from datasets import Dataset as HFDataset

        raw = self._build_squad_examples(records)
        if len(raw) < 10:
            log.warning("Not enough age span examples — skipping fine-tune.")
            return

        def tokenize_fn(batch):
            tok = self.tokenizer(
                batch["question"], batch["context"],
                truncation="only_second",
                max_length=AGE_MAX_CONTEXT,
                stride=50,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            start_positions, end_positions = [], []
            for i, offsets in enumerate(tok["offset_mapping"]):
                sample_idx = tok["overflow_to_sample_mapping"][i]
                ans_start  = batch["start_pos"][sample_idx]
                ans_end    = ans_start + len(batch["answer_text"][sample_idx])
                # Find token positions
                tok_start = tok_end = 0
                for j, (o_s, o_e) in enumerate(offsets):
                    if o_s <= ans_start < o_e:
                        tok_start = j
                    if o_s < ans_end <= o_e:
                        tok_end = j
                        break
                start_positions.append(tok_start)
                end_positions.append(tok_end)
            tok["start_positions"] = start_positions
            tok["end_positions"]   = end_positions
            tok.pop("offset_mapping")
            tok.pop("overflow_to_sample_mapping")
            return tok

        ds = HFDataset.from_dict({
            "question":    [e["question"]    for e in raw],
            "context":     [e["context"]     for e in raw],
            "answer_text": [e["answer_text"] for e in raw],
            "start_pos":   [e["start_pos"]   for e in raw],
        })
        split    = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"].map(tokenize_fn, batched=True,
                                      remove_columns=ds.column_names)
        eval_ds  = split["test"].map(tokenize_fn,  batched=True,
                                     remove_columns=ds.column_names)

        Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=2e-5,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                use_cpu=not use_gpu,
                seed=42,
                logging_steps=5,
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_data_collator,
        ).train()

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        log.info(f"Age QA model saved → {output_dir}")

    def tune_thresholds(self, records: list):
        """Sweep confidence thresholds on training records."""
        from tqdm import tqdm
        best = {"minimum": (0.35, 0.0), "maximum": (0.35, 0.0)}
        for threshold in tqdm(np.arange(0.1, 0.85, 0.05),
                              desc="Tuning age thresholds", unit="threshold"):
            for bound in ("minimum", "maximum"):
                correct = total = 0
                for rec in records:
                    gold = rec.get(f"gold_{bound[:3]}", "").strip()
                    if not gold or gold.upper() in ("NAN", "NOT SPECIFIED", ""):
                        continue
                    try:
                        result = self._run_qa(AGE_QUESTIONS[bound],
                                              rec["candidate_text"][:AGE_MAX_CONTEXT])
                        pred = _normalize_age(result["answer"]) \
                               if result["score"] > threshold \
                               else "Not Specified"
                    except Exception:
                        pred = "Not Specified"
                    correct += int(pred == gold)
                    total   += 1
                acc = correct / total if total else 0
                if acc > best[bound][1]:
                    best[bound] = (threshold, acc)

        self.min_threshold = best["minimum"][0]
        self.max_threshold = best["maximum"][0]
        log.info(f"Age thresholds → min={self.min_threshold:.2f} "
                 f"(acc={best['minimum'][1]:.3f}), "
                 f"max={self.max_threshold:.2f} "
                 f"(acc={best['maximum'][1]:.3f})")

    def extract(self, candidate_text: str, bound: str) -> str:
        threshold = self.min_threshold if bound == "minimum" else self.max_threshold
        try:
            result = self._run_qa(AGE_QUESTIONS[bound],
                                  candidate_text[:AGE_MAX_CONTEXT])
            if result["score"] > threshold:
                return _normalize_age(result["answer"])
        except Exception:
            pass
        return "Not Specified"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"min_threshold": self.min_threshold,
                       "max_threshold": self.max_threshold}, f)

    def load_thresholds(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.min_threshold = d["min_threshold"]
        self.max_threshold = d["max_threshold"]

    @classmethod
    def load(cls, model_dir: str, threshold_path: str) -> "AgeExtractor":
        obj = cls(model_name=model_dir)
        obj.load_thresholds(threshold_path)
        return obj


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class CohortXPipeline:

    def __init__(self, nxml_dir: str, models_dir: str, use_gpu: bool = False):
        self.nxml_dir   = Path(nxml_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu    = use_gpu
        self.parser     = NXMLParser()

        ranker    = Stage1Ranker()
        generator = SciFiveGenerator()

        self.eligibility_extractor = EligibilityExtractor(ranker, generator)
        self.study_type_clf        = BERTClassifier()
        self.sex_clf               = BERTClassifier()
        self.conditions_extractor  = ConditionsExtractor()
        self.age_extractor         = AgeExtractor()
        self._embedder             = None

    def _get_embedder(self) -> BiomedBERTEmbedder:
        if self._embedder is None:
            self._embedder = BiomedBERTEmbedder()
        return self._embedder

    def load_nxml(self, pmcid: str) -> dict:
        path = self.nxml_dir / f"PMC{pmcid}.nxml"
        if not path.exists():
            log.warning(f"NXML not found: {path}")
            return {}
        return self.parser.parse(str(path))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_df: pd.DataFrame) -> dict:
        log.info(f"Training on {len(train_df)} examples...")
        md = self.models_dir

        # Parse all NXML files once
        log.info("Parsing NXML files...")
        parsed_map = {
            str(row["pmcids"]).strip(): self.load_nxml(str(row["pmcids"]).strip())
            for _, row in train_df.iterrows()
        }
        parsed_list = [parsed_map.get(str(r["pmcids"]).strip(), {})
                       for _, r in train_df.iterrows()]

        def col(name):
            return [str(r.get(name, "")).strip() for _, r in train_df.iterrows()]

        gt_elig  = col("eligibility_criteria")
        gt_study = [s.upper() for s in col("study_type")]
        gt_sex   = [s.upper() for s in col("sex")]
        gt_conds = col("conditions")
        gt_min   = col("minimum_age")
        gt_max   = col("maximum_age")

        embedder = self._get_embedder()

        # 1. Stage-1 MiniLM fine-tuning
        log.info("— Fine-tuning stage-1 ranker...")
        examples = self.eligibility_extractor.ranker.build_training_examples(
            parsed_list, gt_elig, embedder
        )
        if examples:
            self.eligibility_extractor.ranker.fine_tune(
                examples, output_path=str(md / "stage1_ranker"),
                use_gpu=self.use_gpu,
            )

        # 2. SciFive fine-tuning (eligibility stage-2)
        log.info("— Fine-tuning SciFive generator...")
        self.eligibility_extractor.generator.fine_tune(
            parsed_list, gt_elig,
            ranker=self.eligibility_extractor.ranker,
            output_dir=str(md / "scifive_model"),
            use_gpu=self.use_gpu,
        )

        # 3. study_type DistilBERT
        log.info("— Fine-tuning study_type classifier...")
        st_texts, st_labels = zip(*[
            (p.get("title", "") + " " + p.get("abstract", ""), l)
            for p, l in zip(parsed_list, gt_study)
            if p and l not in ("", "NAN", "NOT SPECIFIED")
        ]) if any(p and l not in ("", "NAN", "NOT SPECIFIED")
                  for p, l in zip(parsed_list, gt_study)) else ([], [])
        if len(st_texts) >= 10:
            self.study_type_clf.fit(list(st_texts), list(st_labels),
                                    output_dir=str(md / "study_type_model"),
                                    use_gpu=self.use_gpu)

        # 4. sex DistilBERT
        log.info("— Fine-tuning sex classifier...")
        sx_texts, sx_labels = [], []
        for parsed, label in zip(parsed_list, gt_sex):
            if not parsed or label in ("", "NAN", "NOT SPECIFIED"):
                continue
            cands = self.eligibility_extractor.ranker.get_candidates(
                parsed.get("sections", [])
            )
            sx_texts.append(" ".join(cands) or parsed.get("abstract", ""))
            sx_labels.append(label)
        if len(sx_texts) >= 10:
            self.sex_clf.fit(sx_texts, sx_labels,
                             output_dir=str(md / "sex_model"),
                             use_gpu=self.use_gpu)

        # 5. Conditions selector (SciFive fine-tuning)
        log.info("— Fine-tuning conditions selector...")
        cond_records = []
        for parsed, raw_gt in zip(parsed_list, gt_conds):
            if not parsed or raw_gt in ("", "NAN", "NOT SPECIFIED"):
                continue
            try:
                gt_list = ast.literal_eval(raw_gt) if raw_gt.startswith("[") \
                          else [s.strip() for s in raw_gt.split(",") if s.strip()]
            except Exception:
                gt_list = [raw_gt]
            if gt_list:
                cond_records.append({"parsed": parsed, "ground_truth": gt_list})
        if cond_records:
            self.conditions_extractor.fine_tune(
                cond_records,
                output_dir=str(md / "conditions_selector_model"),
                use_gpu=self.use_gpu,
            )

        # 6. Age: fine-tune BioBERT QA + tune thresholds
        log.info("— Fine-tuning age extractor...")
        age_records = []
        for parsed, gmin, gmax in zip(parsed_list, gt_min, gt_max):
            if not parsed:
                continue
            cands = self.eligibility_extractor.ranker.get_candidates(
                parsed.get("sections", [])
            )
            age_records.append({
                "candidate_text": " ".join(cands) or parsed.get("abstract", ""),
                "gold_min": gmin,
                "gold_max": gmax,
            })
        if age_records:
            self.age_extractor.fine_tune(
                age_records, output_dir=str(md / "age_qa_model"),
                use_gpu=self.use_gpu,
            )
            self.age_extractor.tune_thresholds(age_records)
            self.age_extractor.save(str(md / "age_thresholds.json"))

        log.info("All training complete.")
        result = {"status": "complete"}
        with open(md / "training_eval.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_all(self, parsed: dict) -> dict:
        if not parsed:
            return {k: "Not Specified" for k in
                    ["conditions", "study_type", "sex",
                     "minimum_age", "maximum_age", "eligibility_criteria"]}

        sections       = parsed.get("sections", [])
        abstract       = parsed.get("abstract", "")
        title          = parsed.get("title", "")
        eligibility    = self.eligibility_extractor.extract(sections)
        candidate_text = " ".join(
            self.eligibility_extractor.ranker.get_candidates(sections)
        ) or abstract

        return {
            "conditions":           self.conditions_extractor.extract(parsed),
            "study_type":           self.study_type_clf.predict(title + " " + abstract),
            "sex":                  self.sex_clf.predict(candidate_text),
            "minimum_age":          self.age_extractor.extract(candidate_text, "minimum"),
            "maximum_age":          self.age_extractor.extract(candidate_text, "maximum"),
            "eligibility_criteria": eligibility,
        }

    # ------------------------------------------------------------------
    # Load saved models
    # ------------------------------------------------------------------

    def load_trained_models(self):
        md = self.models_dir
        for attr, (loader, dirname) in [
            ("ranker",     (Stage1Ranker.load,    "stage1_ranker")),
            ("generator",  (SciFiveGenerator.load, "scifive_model")),
        ]:
            p = md / dirname
            if p.exists():
                setattr(self.eligibility_extractor, attr, loader(str(p)))
                log.info(f"Loaded {dirname}.")

        for attr, dirname in [("study_type_clf", "study_type_model"),
                               ("sex_clf",        "sex_model")]:
            p = md / dirname
            if p.exists():
                setattr(self, attr, BERTClassifier.load(str(p)))
                log.info(f"Loaded {dirname}.")

        cond_path = md / "conditions_selector_model"
        if cond_path.exists():
            self.conditions_extractor = ConditionsExtractor.load(str(cond_path))
            log.info("Loaded conditions selector.")

        age_model_path = md / "age_qa_model"
        age_thr_path   = md / "age_thresholds.json"
        if age_model_path.exists() and age_thr_path.exists():
            self.age_extractor = AgeExtractor.load(
                str(age_model_path), str(age_thr_path)
            )
            log.info("Loaded age QA model.")
        elif age_thr_path.exists():
            self.age_extractor.load_thresholds(str(age_thr_path))
            log.info("Loaded age thresholds only.")

