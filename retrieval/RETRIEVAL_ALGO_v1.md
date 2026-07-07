# Retrieval Algorithm

## 1. Parse NXML

Convert each NXML paper into a hierarchy-preserving document consisting of

- section tree
- atomic text blocks (paragraphs, lists, tables, figure captions)
- document metadata (title, abstract, keywords)

---

## 2. Embed blocks

For every block, construct an embedding text:

```text
Section path: Study design > Inclusion Criteria
Block type: paragraph
Text: Patients must be over 18 years of age...
```

Embed using

```text
BAAI/bge-small-en-v1.5
```

Embeddings are cached locally.

---

## 3. Candidate retrieval

For each field group

- Condition + Study Type
- Demographics
- Eligibility

embed one or more field-specific natural-language queries and compute cosine similarity with every block.

For multiple queries, use the maximum similarity.

Retrieve the top

| Field                  | Candidate Pool |
| ---------------------- | -------------: |
| Condition + Study Type |             40 |
| Demographics           |             40 |
| Eligibility            |             60 |

candidate blocks.

---

## 4. Metadata reranking

Adjust each candidate score using section metadata.

Positive priors are field-specific.

Negative priors are universal.

Final score

```text
final_score =
cosine_similarity
×
max(positive_prior)
×
min(negative_prior)
```

---

## 5. Context packing

Sort candidates by reranked score.

Keep adding blocks until

- cumulative relevance score ≥ field threshold
- minimum character budget satisfied
- maximum character budget not exceeded

Thresholds

| Field                  | Score Target | Min Chars | Max Chars |
| ---------------------- | -----------: | --------: | --------: |
| Condition + Study Type |          0.8 |         — |      2000 |
| Demographics           |          1.5 |      1800 |      4500 |
| Eligibility            |          2.5 |      3000 |      8000 |

---

## 6. Construct LLM context

Condition + Study Type always prepends

- Title
- Abstract
- Keywords

Retrieved body evidence supplements these.

Demographics and Eligibility consist only of retrieved evidence.

---

## 7. LLM inference

Three specialized prompts are executed

1. Condition + Study Type
2. Demographics
3. Eligibility

Outputs are merged into the final prediction.