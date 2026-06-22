import argparse
import json
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
    return np.mean(diffs), lower, upper

def main():
    parser = argparse.ArgumentParser(description="Calculate Bootstrap CI for ROC AUC gap between two models.")
    parser.add_argument("--model_a", required=True, help="per_step_scores.json for Model A (e.g. priv)")
    parser.add_argument("--model_b", required=True, help="per_step_scores.json for Model B (e.g. nogt)")
    parser.add_argument("--n_iterations", type=int, default=10000)
    args = parser.parse_args()

    data_a = load_scores(args.model_a)
    data_b = load_scores(args.model_b)

    # Assuming format: {"y_true": [...], "y_score": [...]}
    y_true = np.array(data_a["y_true"])
    scores_a = np.array(data_a["y_score"])
    scores_b = np.array(data_b["y_score"])
    
    assert np.array_equal(y_true, np.array(data_b["y_true"])), "True labels do not match between the two files!"

    mean_diff, lower, upper = bootstrap_ci(y_true, scores_a, scores_b, args.n_iterations)
    
    print(f"Bootstrap CI (N={args.n_iterations}) for Model A - Model B ROC AUC gap:")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    
    if lower <= 0 <= upper:
        print("Result: NOT SIGNIFICANT. The CI includes 0.")
    else:
        print("Result: SIGNIFICANT. The CI does not include 0.")

if __name__ == "__main__":
    main()
