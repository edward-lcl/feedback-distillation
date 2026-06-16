"""Deterministically shuffle a JSONL file (fixed seed).

ProcessBench ships all-error rows first then all-correct; a fixed-seed shuffle
gives a balanced correct/error mix in any prefix, so --max_samples N is
representative. Reproducible: same seed -> same order.

    python -m scripts.shuffle_jsonl --input in.jsonl --output out.jsonl --seed 0
"""
import json
import random
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"shuffled {len(rows)} rows (seed {args.seed}) -> {args.output}")


if __name__ == "__main__":
    main()
