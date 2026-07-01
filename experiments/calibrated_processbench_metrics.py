"""Held-out threshold calibration for saved ProcessBench per-step scores.

The main Phase B claim uses threshold-free ROC/PR-AUC. This script is for
reporting F1-style numbers without choosing a threshold on the same examples
being evaluated.
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    true_first: int | None


def load_spans(dataset_path: str, max_samples: int | None) -> tuple[list[Span], list[int]]:
    spans = []
    y_true = []
    cursor = 0
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            labels = [int(bool(step.get("is_error", False))) for step in rec["steps"]]
            true_first = next((j for j, label in enumerate(labels) if label), None)
            y_true.extend(labels)
            spans.append(Span(cursor, cursor + len(labels), true_first))
            cursor += len(labels)
    return spans, y_true


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    best_idx = 0
    best_f1 = -1.0
    for idx, (precision, recall) in enumerate(zip(precisions, recalls)):
        denom = precision + recall
        f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
        if idx < len(thresholds) and f1 > best_f1:
            best_idx = idx
            best_f1 = f1
    return float(thresholds[best_idx])


def slice_indices(spans: list[Span], start_seq: int, end_seq: int) -> np.ndarray:
    parts = [np.arange(spans[i].start, spans[i].end) for i in range(start_seq, end_seq)]
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)


def first_error_metrics(spans: list[Span], predictions: np.ndarray, seq_offset: int) -> dict:
    total = correct = 0
    err_total = err_correct = 0
    clean_total = clean_correct = 0
    for local_i, span in enumerate(spans):
        pred_first = None
        seq_preds = predictions[span.start:span.end]
        for j, pred in enumerate(seq_preds):
            if pred:
                pred_first = j
                break
        true_first = span.true_first
        correct += int(true_first == pred_first)
        total += 1
        if true_first is None:
            clean_total += 1
            clean_correct += int(pred_first is None)
        else:
            err_total += 1
            err_correct += int(true_first == pred_first)
    return {
        "first_error_acc": correct / max(1, total),
        "first_error_acc_errs": err_correct / err_total if err_total else None,
        "clean_seq_acc": clean_correct / clean_total if clean_total else None,
        "n_sequences": total,
        "n_error_sequences": err_total,
        "n_clean_sequences": clean_total,
        "seq_offset": seq_offset,
    }


def evaluate_split(
    y_true: np.ndarray,
    y_score: np.ndarray,
    spans: list[Span],
    eval_start_seq: int,
    threshold: float,
) -> dict:
    idx = slice_indices(spans, 0, len(spans))
    y = y_true[idx]
    score = y_score[idx]
    pred = (score >= threshold).astype(np.int64)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "clean_specificity": tn / n_neg if n_neg else None,
        "pred_error_rate": float(np.mean(pred)),
        "n_steps": int(len(y)),
        "n_error_steps": n_pos,
        **first_error_metrics(spans, pred, eval_start_seq),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True, help="per_step_scores.json files")
    parser.add_argument("--names", nargs="+", help="Display names matching --scores")
    parser.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--cal_samples", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.scores):
        raise SystemExit("--names must have same length as --scores")

    spans, dataset_y = load_spans(args.dataset, args.max_samples)
    y_true = np.asarray(dataset_y, dtype=np.int64)
    cal_idx = slice_indices(spans, 0, args.cal_samples)
    eval_idx = slice_indices(spans, args.cal_samples, len(spans))
    eval_spans = [
        Span(span.start - eval_idx[0], span.end - eval_idx[0], span.true_first)
        for span in spans[args.cal_samples:]
    ]

    results = []
    for i, path in enumerate(args.scores):
        with open(path) as f:
            data = json.load(f)
        score_y = np.asarray(data["y_true"], dtype=np.int64)
        if not np.array_equal(score_y, y_true):
            raise AssertionError(f"Score labels do not match dataset order: {path}")
        y_score = np.asarray(data["y_score"], dtype=np.float64)
        threshold = best_threshold(y_true[cal_idx], y_score[cal_idx])
        split = evaluate_split(
            y_true=y_true[eval_idx],
            y_score=y_score[eval_idx],
            spans=eval_spans,
            eval_start_seq=args.cal_samples,
            threshold=threshold,
        )
        results.append({
            "name": args.names[i] if args.names else path,
            "scores": path,
            "threshold": threshold,
            "calibration": {
                "start_seq": 0,
                "end_seq": args.cal_samples,
                "n_steps": int(len(cal_idx)),
                "n_error_steps": int(np.sum(y_true[cal_idx])),
            },
            "evaluation": {
                "start_seq": args.cal_samples,
                "end_seq": len(spans),
                **split,
            },
        })

    payload = {
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "cal_samples": args.cal_samples,
        "method": "threshold_max_f1_on_calibration_sequences_then_eval_on_remaining_sequences",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved -> {args.out}")
    for row in results:
        ev = row["evaluation"]
        print(
            f"{row['name']}: threshold={row['threshold']:.4f} "
            f"eval_f1={ev['f1']:.4f} roc_auc={ev['roc_auc']:.4f} "
            f"pr_auc={ev['pr_auc']:.4f} pred_error_rate={ev['pred_error_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
