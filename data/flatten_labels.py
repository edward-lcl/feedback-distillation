"""
Flatten teacher-labeled JSONL into a flat per-step training dataset.

`label_pipeline` writes one record per *solution*:
    {problem, solution, gt_answer, steps: [{text, score, feedback, is_error}]}

`SLFDTrainer` consumes a flat list of per-*step* dicts:
    {problem, solution_prefix, step_text, score, feedback, is_error}

This module bridges the two. `solution_prefix` is reconstructed by accumulating
the text of all preceding steps (the same way the teacher built it at labeling
time), so the student sees exactly the context the teacher scored against.

Usage:
    python -m data.flatten_labels \
        --input data/labeled/train_labeled.jsonl \
        --output data/labeled/train_steps.jsonl
"""
import json
import argparse


def flatten_labeled_records(records: list[dict]) -> list[dict]:
    """Expand per-solution labeled records into per-step training examples."""
    flat = []
    dropped = 0
    for rec in records:
        problem = rec.get("problem", "")
        prefix = ""
        for step in rec.get("steps", []):
            text = step.get("text", "")
            # Never train on unparseable teacher labels — that's label noise.
            if step.get("parse_failed") or step.get("score") is None:
                dropped += 1
                prefix += text + "\n"
                continue
            if not problem or not text:
                prefix += text + "\n"
                continue
            flat.append({
                "problem": problem,
                "solution_prefix": prefix,
                "step_text": text,
                "score": float(step.get("score", 0.0)),
                "feedback": step.get("feedback", ""),
                "is_error": bool(step.get("is_error", False)),
            })
            prefix += text + "\n"
    if dropped:
        print(f"[flatten] dropped {dropped} steps with unparseable teacher labels")
    return flat


def flatten_labeled_file(input_path: str) -> list[dict]:
    """Read a labeled JSONL file and return a flat list of per-step dicts."""
    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return flatten_labeled_records(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Labeled JSONL from label_pipeline.")
    parser.add_argument("--output", required=True, help="Flat per-step JSONL for training.")
    args = parser.parse_args()

    flat = flatten_labeled_file(args.input)
    with open(args.output, "w") as f:
        for ex in flat:
            f.write(json.dumps(ex) + "\n")
    print(f"Flattened {args.input} → {len(flat)} step examples → {args.output}")


if __name__ == "__main__":
    main()
