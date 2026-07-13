"""
CohortX Task 1 — Submission entry point (offline, CPU-only)
===========================================================
Thin wrapper around the shipped Path-B extractor so graders can launch the
submission as a bare script from the repo root:

  ollama serve                       # start the local model server
  ollama pull qwen2.5:1.5b
  python run_submission.py --data_dir <path> --nxml_dir <path> [--test]

This is exactly equivalent to `python -m local_llm.predict_ollama ...`. It runs
fully offline on local Ollama Qwen — no external API is ever called on this path.
"""

from local_llm.predict_ollama import main

if __name__ == "__main__":
    main()
