"""
Split labeled JSONL into train/dev, grouped by PROBLEM (not by line) so the
same problem never appears in both splits — step-level rows from one problem
are highly correlated, and a line-level split would leak.

Usage:
    python -m scripts.split_jsonl --input data/labeled/gsm8k_labeled.jsonl \
        --train_out data/labeled/train.jsonl --dev_out data/labeled/dev.jsonl \
        --dev_frac 0.1 --seed 42
"""
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--dev_out", required=True)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    problems = sorted({r.get("problem", "") for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(problems)
    n_dev = max(1, int(len(problems) * args.dev_frac)) if len(problems) > 1 else 0
    dev_set = set(problems[:n_dev])

    train = [r for r in rows if r.get("problem", "") not in dev_set]
    dev = [r for r in rows if r.get("problem", "") in dev_set]

    for path, split in ((args.train_out, train), (args.dev_out, dev)):
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
    print(f"{len(problems)} problems → train {len(train)} rows / dev {len(dev)} rows "
          f"({n_dev} dev problems)")


if __name__ == "__main__":
    main()
