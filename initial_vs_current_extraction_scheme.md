# Initial Approach → Current Approach

```text
Initial Approach                                        Current Approach

Full article                                            Full article
     │                                                       │
     ▼                                                       ▼
Section retrieval                                      Hierarchical parser
     │                                                       │
     ▼                                                       ▼
Embedding similarity                                   Unified semantic retrieval
     │                                                       │
     ▼                                                       ▼
Top-k / threshold heuristic                            Top blocks covering ~30% of article
     │                                                 (minimum 8 blocks)
     ▼                                                       │
Qwen 3 4B (single prompt)                                   ▼
     │                                                 BioMedBERT evidence classifier
     ▼                                                 (finetuned on LLM labeled relevance data)
Structured extraction                                        │
                                                             ├──────────────┬──────────────┐
                                                             ▼              ▼              ▼
                                                        Condition     Demographics    Eligibility
                                                        threshold=.33 threshold=.03 threshold=.05
                                                             │              │              │
                                                             ▼              ▼              ▼
                                                     Field-specific evidence contexts
                                                             │              │              │
                                                             ├──────────────┼──────────────┤
                                                             ▼              ▼              ▼
                                                 Claude Sonnet 4.6 (3 independent calls)
                                                             │              │              │
                                                             └──────────────┴──────────────┘
                                                                            │
                                                                            ▼
                                                              Merge structured outputs
```



``` |


