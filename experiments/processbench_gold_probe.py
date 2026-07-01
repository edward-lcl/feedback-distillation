"""
Matched-distribution frozen representation probe.

This is a diagnostic only: split ProcessBench by problem, train a linear probe on
gold ProcessBench step-error labels from one slice, and evaluate on a held-out
slice. If this is weak, the frozen step-boundary representation is not linearly
separable for the target task. If this is strong while teacher-label probes are
weak, the issue is label/distribution mismatch rather than representation alone.
"""
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch

from experiments.representation_probe import (
    extract_representations,
    fit_probe,
    metric_summary,
    processbench_steps,
)
from models.student import StudentModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/diagnostics/processbench_gold_probe.json")
    parser.add_argument("--dev_mode", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # The source file is already shuffled; use disjoint contiguous problem slices.
    train_inputs, y_train = processbench_steps(args.dataset, args.train_samples)
    eval_inputs_all, y_eval_all = processbench_steps(
        args.dataset,
        args.train_samples + args.eval_samples,
    )
    train_cut = len(train_inputs)
    eval_inputs = eval_inputs_all[train_cut:]
    y_eval = y_eval_all[train_cut:]

    student = StudentModel(args.student_model, dev_mode=args.dev_mode, use_lora=False)
    student.model.eval()

    x_train = extract_representations(student, train_inputs, batch_size=args.batch_size)
    x_eval = extract_representations(student, eval_inputs, batch_size=args.batch_size)
    probe = fit_probe(x_train, y_train)

    train_scores = probe.predict_proba(x_train)[:, 1]
    eval_scores = probe.predict_proba(x_eval)[:, 1]
    result = {
        "student_model": args.student_model,
        "dataset": args.dataset,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_label_counts": dict(Counter(map(int, y_train.tolist()))),
        "eval_label_counts": dict(Counter(map(int, y_eval.tolist()))),
        "train": metric_summary(y_train, train_scores),
        "eval": metric_summary(y_eval, eval_scores),
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
