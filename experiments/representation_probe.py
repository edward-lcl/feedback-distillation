"""
Frozen-representation diagnostic for Phase B.

Train cheap logistic probes on the frozen student's step-boundary hidden states
using privileged vs no-GT teacher labels, then evaluate on ProcessBench gold
step-error labels. This answers whether the current labels/features contain a
usable linear signal before spending more LoRA/BoN compute.
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.flatten_labels import flatten_labeled_file
from models.student import StudentModel


def is_error_step(row: dict) -> bool:
    if "is_error" in row:
        return bool(row["is_error"])
    return float(row.get("score", 0.0)) < 0.0


def sample_training_steps(rows: list[dict], max_steps: int | None, seed: int) -> list[dict]:
    rows = [r for r in rows if r.get("problem") and r.get("step_text")]
    if max_steps is None or len(rows) <= max_steps:
        return rows
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:max_steps]


def processbench_steps(dataset_path: str, max_samples: int | None):
    inputs = []
    labels = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            sample = json.loads(line)
            problem = sample["problem"]
            prefix = ""
            for step in sample["steps"]:
                inputs.append((problem, prefix, step["text"]))
                labels.append(int(bool(step.get("is_error", False))))
                prefix += step["text"] + "\n"
    return inputs, np.asarray(labels, dtype=np.int64)


def train_steps(label_path: str, max_steps: int | None, seed: int):
    rows = sample_training_steps(flatten_labeled_file(label_path), max_steps, seed)
    inputs = [(r["problem"], r.get("solution_prefix", ""), r["step_text"]) for r in rows]
    labels = np.asarray([int(is_error_step(r)) for r in rows], dtype=np.int64)
    return inputs, labels


def extract_representations(
    student: StudentModel,
    inputs: list[tuple[str, str, str]],
    batch_size: int,
) -> np.ndarray:
    reps = student.step_representations(inputs, batch_size=batch_size)
    return reps.numpy()


def metric_summary(y_true: np.ndarray, error_score: np.ndarray) -> dict:
    y_pred = (error_score >= 0.5).astype(np.int64)
    has_both = len(set(y_true.tolist())) > 1
    return {
        "roc_auc": float(roc_auc_score(y_true, error_score)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, error_score)) if has_both else None,
        "f1_at_0p5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_0p5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0p5": float(recall_score(y_true, y_pred, zero_division=0)),
        "pred_error_rate_at_0p5": float(y_pred.mean()) if len(y_pred) else None,
        "n_steps": int(len(y_true)),
        "n_error_steps": int(y_true.sum()),
        "error_rate": float(y_true.mean()) if len(y_true) else None,
    }


def fit_probe(x_train: np.ndarray, y_train: np.ndarray):
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Probe training labels contain only one class")
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
            random_state=0,
        ),
    )
    probe.fit(x_train, y_train)
    return probe


def run_cell(
    name: str,
    label_path: str,
    student: StudentModel,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    max_train_steps: int | None,
    batch_size: int,
    seed: int,
) -> dict:
    start = time.time()
    inputs, y_train = train_steps(label_path, max_train_steps, seed)
    x_train = extract_representations(student, inputs, batch_size=batch_size)
    probe = fit_probe(x_train, y_train)
    train_scores = probe.predict_proba(x_train)[:, 1]
    eval_scores = probe.predict_proba(x_eval)[:, 1]
    return {
        "name": name,
        "label_path": label_path,
        "train": metric_summary(y_train, train_scores),
        "eval": metric_summary(y_eval, eval_scores),
        "train_label_counts": dict(Counter(map(int, y_train.tolist()))),
        "seconds": round(time.time() - start, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--priv_labels", default="data/labeled/math_priv.jsonl")
    parser.add_argument("--nogt_labels", default="data/labeled/math_nogt.jsonl")
    parser.add_argument("--eval_dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/diagnostics/representation_probe.json")
    parser.add_argument("--dev_mode", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    student = StudentModel(args.student_model, dev_mode=args.dev_mode, use_lora=False)
    student.model.eval()

    eval_inputs, y_eval = processbench_steps(args.eval_dataset, args.max_eval_samples)
    x_eval = extract_representations(student, eval_inputs, batch_size=args.batch_size)

    result = {
        "student_model": args.student_model,
        "eval_dataset": args.eval_dataset,
        "max_train_steps": args.max_train_steps,
        "max_eval_samples": args.max_eval_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_label_counts": dict(Counter(map(int, y_eval.tolist()))),
        "cells": [],
    }
    result["cells"].append(run_cell(
        "priv",
        args.priv_labels,
        student,
        x_eval,
        y_eval,
        args.max_train_steps,
        args.batch_size,
        args.seed,
    ))
    result["cells"].append(run_cell(
        "nogt",
        args.nogt_labels,
        student,
        x_eval,
        y_eval,
        args.max_train_steps,
        args.batch_size,
        args.seed,
    ))
    cells = {cell["name"]: cell for cell in result["cells"]}
    result["priv_minus_nogt_eval_roc_auc"] = (
        cells["priv"]["eval"]["roc_auc"] - cells["nogt"]["eval"]["roc_auc"]
        if cells["priv"]["eval"]["roc_auc"] is not None and cells["nogt"]["eval"]["roc_auc"] is not None
        else None
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
