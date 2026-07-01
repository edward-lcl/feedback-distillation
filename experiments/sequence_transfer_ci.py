"""Sequence-cluster bootstrap for paired ProcessBench ROC-AUC gaps.

`experiments.transfer_ci` bootstraps individual steps. That is useful for a
quick gate, but ProcessBench steps from the same solution are correlated. This
script reconstructs sequence spans from the dataset JSONL and resamples whole
solutions with replacement.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_spans(dataset_path: str, max_samples: int | None) -> tuple[list[tuple[int, int]], list[int]]:
    spans = []
    y_true = []
    cursor = 0
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            steps = rec["steps"]
            labels = [int(bool(step.get("is_error", False))) for step in steps]
            y_true.extend(labels)
            spans.append((cursor, cursor + len(labels)))
            cursor += len(labels)
    return spans, y_true


def sequence_bootstrap(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    spans: list[tuple[int, int]],
    n_boot: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(0)
    diffs = []
    n_seq = len(spans)

    for _ in range(n_boot):
        sampled = rng.integers(0, n_seq, size=n_seq)
        idx_parts = [np.arange(spans[j][0], spans[j][1]) for j in sampled]
        idx = np.concatenate(idx_parts)
        y = y_true[idx]
        if len(np.unique(y)) < 2:
            continue
        auc_a = roc_auc_score(y, scores_a[idx])
        auc_b = roc_auc_score(y, scores_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.asarray(diffs, dtype=np.float64)
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return diffs, float(np.mean(diffs)), lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True, help="per_step_scores.json for model A")
    parser.add_argument("--model_b", required=True, help="per_step_scores.json for model B")
    parser.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--out")
    args = parser.parse_args()

    data_a = load_scores(args.model_a)
    data_b = load_scores(args.model_b)
    spans, dataset_y = load_spans(args.dataset, args.max_samples)

    y_true = np.asarray(data_a["y_true"], dtype=np.int64)
    scores_a = np.asarray(data_a["y_score"], dtype=np.float64)
    scores_b = np.asarray(data_b["y_score"], dtype=np.float64)
    dataset_y = np.asarray(dataset_y, dtype=np.int64)

    if not np.array_equal(y_true, np.asarray(data_b["y_true"], dtype=np.int64)):
        raise AssertionError("True labels do not match between score files")
    if not np.array_equal(y_true, dataset_y):
        raise AssertionError("Score labels do not match dataset sequence order")

    diffs, mean_diff, lower, upper = sequence_bootstrap(
        y_true=y_true,
        scores_a=scores_a,
        scores_b=scores_b,
        spans=spans,
        n_boot=args.n_boot,
    )
    p_a_gt_b = float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))
    p_b_gt_a = float((np.sum(diffs >= 0) + 1) / (len(diffs) + 1))
    p_two_sided = min(1.0, 2 * min(p_a_gt_b, p_b_gt_a))

    result = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "metric": "roc_auc",
        "gap_model_a_minus_model_b": mean_diff,
        "ci95": [lower, upper],
        "p_one_sided_model_a_gt_model_b": p_a_gt_b,
        "p_two_sided": p_two_sided,
        "n_boot": args.n_boot,
        "n_boot_valid": int(len(diffs)),
        "n_sequences": int(len(spans)),
        "n_steps": int(len(y_true)),
        "method": "paired_sequence_cluster_bootstrap",
    }

    print(f"Sequence-cluster bootstrap CI (N={args.n_boot}) for Model A - Model B ROC-AUC gap:")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"One-sided p(Model A > Model B): {p_a_gt_b:.4f}")
    print(f"Two-sided p: {p_two_sided:.4f}")
    if lower <= 0 <= upper:
        print("Result: NOT SIGNIFICANT. The CI includes 0.")
    else:
        print("Result: SIGNIFICANT. The CI does not include 0.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
