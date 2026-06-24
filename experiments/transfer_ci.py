import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

def load_scores(path):
    with open(path) as f:
        return json.load(f)

def bootstrap_ci(true_labels, scores_a, scores_b, n_iterations=10000, alpha=0.05):
    n = len(true_labels)
    diffs = []
    indices = np.arange(n)
    for _ in range(n_iterations):
        idx = np.random.choice(indices, size=n, replace=True)
        y_true = true_labels[idx]
        y_a = scores_a[idx]
        y_b = scores_b[idx]
        
        # Guard against drawing a sample with only one class
        if len(np.unique(y_true)) < 2:
            continue
            
        auc_a = roc_auc_score(y_true, y_a)
        auc_b = roc_auc_score(y_true, y_b)
        diffs.append(auc_a - auc_b)
        
    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return diffs, float(np.mean(diffs)), float(lower), float(upper)

def main():
    parser = argparse.ArgumentParser(description="Calculate Bootstrap CI for ROC AUC gap between two models.")
    parser.add_argument("--model_a", help="per_step_scores.json for Model A (e.g. priv)")
    parser.add_argument("--model_b", help="per_step_scores.json for Model B (e.g. nogt)")
    parser.add_argument("--priv", help="Alias for --model_a, matching RUNBOOK_PHASE_B.md")
    parser.add_argument("--nogt", help="Alias for --model_b, matching RUNBOOK_PHASE_B.md")
    parser.add_argument("--n_iterations", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=None, help="Alias for --n_iterations")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    model_a_path = args.model_a or args.priv
    model_b_path = args.model_b or args.nogt
    if not model_a_path or not model_b_path:
        parser.error("provide --model_a/--model_b or --priv/--nogt")
    n_iterations = args.n_iterations or args.n_boot or 10000

    data_a = load_scores(model_a_path)
    data_b = load_scores(model_b_path)

    # Assuming format: {"y_true": [...], "y_score": [...]}
    y_true = np.array(data_a["y_true"])
    scores_a = np.array(data_a["y_score"])
    scores_b = np.array(data_b["y_score"])
    
    assert np.array_equal(y_true, np.array(data_b["y_true"])), "True labels do not match between the two files!"

    diffs, mean_diff, lower, upper = bootstrap_ci(y_true, scores_a, scores_b, n_iterations)
    p_a_gt_b = float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))
    p_b_gt_a = float((np.sum(diffs >= 0) + 1) / (len(diffs) + 1))
    p_two_sided = min(1.0, 2 * min(p_a_gt_b, p_b_gt_a))
    result = {
        "model_a": model_a_path,
        "model_b": model_b_path,
        "metric": "roc_auc",
        "gap_model_a_minus_model_b": mean_diff,
        "ci95": [lower, upper],
        "p_one_sided_model_a_gt_model_b": p_a_gt_b,
        "p_two_sided": p_two_sided,
        "n_boot": int(n_iterations),
        "n_boot_valid": int(len(diffs)),
        "n_steps": int(len(y_true)),
        "method": "paired_step_bootstrap",
        "note": "per_step_scores.json has no sequence IDs, so this bootstraps paired step scores rather than clustered sequences.",
    }
    
    print(f"Bootstrap CI (N={n_iterations}) for Model A - Model B ROC AUC gap:")
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
