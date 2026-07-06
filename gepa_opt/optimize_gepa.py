"""
CohortX Task 1 — GEPA optimization driver (DEV-TIME ONLY)
=========================================================
Optimizes the Path-B instruction with DSPy GEPA. Student = local Qwen (Ollama);
reflection = Claude (Anthropic API, dev-time only). Produces one optimized
instruction string, saved for paste-back into predict_ollama.py.

  python -m gepa_opt.optimize_gepa [--auto light] [--train_n 60] [--val_n 40] \
                          [--reflection_model anthropic/claude-sonnet-4-6]

Requires .env with the Anthropic key (ANTHROPIC_API_KEY or CLAUDE_API_KEY).
This file is NEVER imported by the offline inference path.
"""

import argparse
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from gepa_opt.dspy_program import CohortExtractor, build_examples, configure_student, gepa_metric


def setup_anthropic_key() -> None:
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key or key.startswith("sk-ant-xxx"):
        raise SystemExit("No Anthropic key found. Put it in .env as ANTHROPIC_API_KEY=...")
    os.environ["ANTHROPIC_API_KEY"] = key  # what litellm/dspy reads


def reflection_cost(lm: dspy.LM) -> tuple:
    calls = len(getattr(lm, "history", []))
    cost = sum((h.get("cost") or 0.0) for h in getattr(lm, "history", []))
    return calls, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--max_metric_calls", type=int, default=None,
                    help="override --auto with a hard cap (predictable runtime)")
    ap.add_argument("--train_n", type=int, default=None, help="subsample train split")
    ap.add_argument("--val_n", type=int, default=None, help="subsample val split")
    ap.add_argument("--reflection_model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--num_threads", type=int, default=2)
    ap.add_argument("--out", default="results/gepa_optimized.json")
    args = ap.parse_args()

    setup_anthropic_key()
    configure_student()  # student LM = local Qwen + JSONAdapter

    trainset = build_examples("train")
    valset   = build_examples("val")
    if args.train_n:
        trainset = trainset[:args.train_n]
    if args.val_n:
        valset = valset[:args.val_n]
    print(f"trainset={len(trainset)}  valset={len(valset)}")

    reflection_lm = dspy.LM(args.reflection_model, temperature=1.0, max_tokens=32000)

    budget = ({"max_metric_calls": args.max_metric_calls}
              if args.max_metric_calls else {"auto": args.auto})
    optimizer = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
        reflection_minibatch_size=3,
        track_stats=True,
        seed=0,
        **budget,
    )

    optimized = optimizer.compile(CohortExtractor(), trainset=trainset, valset=valset)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    optimized.save(args.out)

    optimized_instruction = optimized.predict.signature.instructions
    Path("results/optimized_instruction.txt").write_text(optimized_instruction)

    calls, cost = reflection_cost(reflection_lm)
    print("\n" + "=" * 70)
    print(f"GEPA done. Saved program -> {args.out}")
    print(f"Reflection calls: {calls}   est. cost: ${cost:.3f}")
    print("=" * 70)
    print("\n--- OPTIMIZED INSTRUCTION (also in results/optimized_instruction.txt) ---\n")
    print(optimized_instruction)


if __name__ == "__main__":
    main()
