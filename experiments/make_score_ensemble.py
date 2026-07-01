"""Average compatible ProcessBench per-step score files.

The input files must share the same `y_true` order. This writes another
`per_step_scores.json`-shaped artifact, so existing evaluation and bootstrap
tools can consume score averages without special cases.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def load_scores(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return (
        np.asarray(data["y_true"], dtype=np.int64),
        np.asarray(data["y_score"], dtype=np.float64),
    )


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1s = []
    for precision, recall in zip(precisions, recalls):
        denom = precision + recall
        f1s.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return float(max(f1s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = []
    scores = []
    for path in args.inputs:
        y_true, y_score = load_scores(path)
        labels.append(y_true)
        scores.append(y_score)

    if not all(np.array_equal(labels[0], y) for y in labels):
        raise AssertionError("Input score files do not share the same y_true order")

    y_true = labels[0]
    y_score = np.mean(scores, axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "y_true": y_true.astype(int).tolist(),
                "y_score": y_score.astype(float).tolist(),
                "metadata": {
                    "method": "mean_per_step_score",
                    "inputs": args.inputs,
                    "n_members": len(args.inputs),
                },
            },
            f,
            indent=2,
        )

    metrics = {
        "out": str(out),
        "n_members": len(args.inputs),
        "n_steps": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "best_f1": best_f1(y_true, y_score),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
