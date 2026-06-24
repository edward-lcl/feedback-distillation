"""Create diagnostic ProcessBench-gold train/eval splits.

The train output is flat per-step JSONL compatible with train_slfd.py. The eval
output preserves the ProcessBench sample schema for run_processbench.py.
This is for diagnostics only; do not use train-on-eval-split numbers as a paper
claim without a proper held-out benchmark.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--eval_out", required=True)
    args = parser.parse_args()

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    train_rows = rows[:args.train_samples]
    eval_rows = rows[args.train_samples:args.train_samples + args.eval_samples]

    for path in [args.train_out, args.eval_out]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    n_train_steps = 0
    n_train_errors = 0
    with open(args.train_out, "w") as f:
        for rec in train_rows:
            problem = rec["problem"]
            prefix = ""
            for step in rec["steps"]:
                is_error = bool(step.get("is_error", False))
                f.write(json.dumps({
                    "problem": problem,
                    "solution_prefix": prefix,
                    "step_text": step["text"],
                    "score": -1.0 if is_error else 1.0,
                    "feedback": "Error." if is_error else "Correct.",
                    "is_error": is_error,
                }) + "\n")
                n_train_steps += 1
                n_train_errors += int(is_error)
                prefix += step["text"] + "\n"

    with open(args.eval_out, "w") as f:
        for rec in eval_rows:
            f.write(json.dumps(rec) + "\n")

    n_eval_steps = sum(len(rec["steps"]) for rec in eval_rows)
    n_eval_errors = sum(
        int(bool(step.get("is_error", False)))
        for rec in eval_rows
        for step in rec["steps"]
    )
    print(json.dumps({
        "dataset": args.dataset,
        "train_out": args.train_out,
        "eval_out": args.eval_out,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "train_steps": n_train_steps,
        "train_error_steps": n_train_errors,
        "eval_steps": n_eval_steps,
        "eval_error_steps": n_eval_errors,
    }, indent=2))


if __name__ == "__main__":
    main()
