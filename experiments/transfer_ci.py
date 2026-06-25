"""Paired bootstrap CI for ROC-AUC gaps between two ProcessBench evaluators.

Reads the per_step_scores.json sidecars written by experiments.run_processbench.
When y_seq is present, resamples whole sequences for a clustered paired
bootstrap. Older sidecars without y_seq fall back to paired step bootstrap.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def paired_step_bootstrap(y_true, scores_a, scores_b, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(indices, size=n, replace=True)
        auc_a = _auc(y_true[idx], scores_a[idx])
        auc_b = _auc(y_true[idx], scores_b[idx])
        if auc_a is None or auc_b is None:
            continue
        diffs.append(auc_a - auc_b)
    return np.asarray(diffs), "paired_step_bootstrap"


def paired_cluster_bootstrap(y_true, scores_a, scores_b, seq, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    seqs = np.unique(seq)
    by_seq = {s: np.where(seq == s)[0] for s in seqs}
    diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(seqs, size=len(seqs), replace=True)
        idx = np.concatenate([by_seq[s] for s in chosen])
        auc_a = _auc(y_true[idx], scores_a[idx])
        auc_b = _auc(y_true[idx], scores_b[idx])
        if auc_a is None or auc_b is None:
            continue
        diffs.append(auc_a - auc_b)
    return np.asarray(diffs), "clustered_sequence_bootstrap"


def paired_bootstrap(y_true, ys_p, ys_n, seq, n_boot=10000, seed=0):
    """Compatibility API for the Phase-B tests and older runbook wording.

    Treats model A as the privileged scorer and model B as the no-GT scorer.
    """
    y_true = np.asarray(y_true, dtype=int)
    ys_p = np.asarray(ys_p, dtype=float)
    ys_n = np.asarray(ys_n, dtype=float)
    seq = np.asarray(seq, dtype=int)

    auc_priv = _auc(y_true, ys_p)
    auc_nogt = _auc(y_true, ys_n)
    if auc_priv is None or auc_nogt is None:
        raise ValueError("roc_auc undefined (single-class eval) — cannot bootstrap.")

    diffs, _ = paired_cluster_bootstrap(y_true, ys_p, ys_n, seq, n_boot=n_boot, seed=seed)
    if len(diffs) == 0:
        raise ValueError("all bootstrap resamples were single-class")

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = float(np.mean(diffs <= 0))
    return {
        "auc_priv": round(float(auc_priv), 4),
        "auc_nogt": round(float(auc_nogt), 4),
        "gap_priv_minus_nogt": round(float(auc_priv - auc_nogt), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_one_sided_priv_le_nogt": round(p_one_sided, 4),
        "n_seqs": int(len(np.unique(seq))),
        "n_steps": int(len(y_true)),
        "n_boot": int(n_boot),
        "significant_at_95": bool(lo > 0 or hi < 0),
    }


def summarize_gap(model_a_path, model_b_path, n_boot=10000, seed=0, alpha=0.05):
    data_a = load_scores(model_a_path)
    data_b = load_scores(model_b_path)

    y_true = np.asarray(data_a["y_true"], dtype=int)
    y_true_b = np.asarray(data_b["y_true"], dtype=int)
    scores_a = np.asarray(data_a["y_score"], dtype=float)
    scores_b = np.asarray(data_b["y_score"], dtype=float)

    if not np.array_equal(y_true, y_true_b):
        raise ValueError("True labels do not match between the two files")
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays do not have the same length")

    auc_a = _auc(y_true, scores_a)
    auc_b = _auc(y_true, scores_b)
    if auc_a is None or auc_b is None:
        raise ValueError("roc_auc undefined because the eval slice has one class")

    seq_a = data_a.get("y_seq")
    seq_b = data_b.get("y_seq")
    if seq_a is not None and seq_b is not None:
        seq_a = np.asarray(seq_a, dtype=int)
        seq_b = np.asarray(seq_b, dtype=int)
        if np.array_equal(seq_a, seq_b):
            diffs, method = paired_cluster_bootstrap(
                y_true, scores_a, scores_b, seq_a, n_boot=n_boot, seed=seed
            )
        else:
            raise ValueError("y_seq arrays do not match between the two files")
    else:
        diffs, method = paired_step_bootstrap(
            y_true, scores_a, scores_b, n_boot=n_boot, seed=seed, alpha=alpha
        )

    if len(diffs) == 0:
        raise ValueError("all bootstrap resamples were single-class")

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    p_a_gt_b = float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))
    p_b_gt_a = float((np.sum(diffs >= 0) + 1) / (len(diffs) + 1))
    p_two_sided = min(1.0, 2 * min(p_a_gt_b, p_b_gt_a))

    result = {
        "model_a": model_a_path,
        "model_b": model_b_path,
        "metric": "roc_auc",
        "auc_model_a": float(auc_a),
        "auc_model_b": float(auc_b),
        "gap_model_a_minus_model_b": mean_diff,
        "ci95": [lower, upper],
        "p_one_sided_model_a_gt_model_b": p_a_gt_b,
        "p_two_sided": p_two_sided,
        "n_boot": int(n_boot),
        "n_boot_valid": int(len(diffs)),
        "n_steps": int(len(y_true)),
        "method": method,
    }
    if method == "clustered_sequence_bootstrap":
        result["n_sequences"] = int(len(np.unique(seq_a)))
    else:
        result["note"] = (
            "per_step_scores.json has no sequence IDs, so this bootstraps paired "
            "step scores rather than clustered sequences."
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate a paired bootstrap CI for ROC-AUC gap between two models."
    )
    parser.add_argument("--model_a", help="per_step_scores.json for Model A")
    parser.add_argument("--model_b", help="per_step_scores.json for Model B")
    parser.add_argument("--priv", help="Alias for --model_a")
    parser.add_argument("--nogt", help="Alias for --model_b")
    parser.add_argument("--n_iterations", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=None, help="Alias for --n_iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    model_a_path = args.model_a or args.priv
    model_b_path = args.model_b or args.nogt
    if not model_a_path or not model_b_path:
        parser.error("provide --model_a/--model_b or --priv/--nogt")
    n_boot = args.n_iterations or args.n_boot or 10000

    result = summarize_gap(model_a_path, model_b_path, n_boot=n_boot, seed=args.seed)

    print(f"Bootstrap CI (N={n_boot}) for Model A - Model B ROC-AUC gap:")
    print(f"Method: {result['method']}")
    print(f"Mean Difference: {result['gap_model_a_minus_model_b']:.4f}")
    print(f"95% CI: [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}]")
    print(f"One-sided p(Model A > Model B): {result['p_one_sided_model_a_gt_model_b']:.4f}")
    print(f"Two-sided p: {result['p_two_sided']:.4f}")
    if result["ci95"][0] <= 0 <= result["ci95"][1]:
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
