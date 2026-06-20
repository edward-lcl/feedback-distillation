"""
Paired bootstrap CI on the privilege transfer gap (D1 rigor).

The N=1000 "null" is `roc_auc(priv) - roc_auc(nogt)` ≈ 0.01 on a single eval.
Before anyone calls that "validated", put a confidence interval on it: is the
gap distinguishable from zero, or is it noise?

Both cells are evaluated on the SAME ProcessBench file in the SAME order, so
their per-step arrays are aligned. We resample at the SEQUENCE level (clustered
bootstrap, since steps within a problem are correlated) using ONE shared set of
resampled sequences per iteration (paired), recompute both cells' roc_auc on it,
and take the difference. That yields a CI on the gap and a one-sided bootstrap
p-value for H0: priv does NOT beat nogt.

Inputs are the per_step_scores.json sidecars written by run_processbench.

Usage:
    python -m experiments.transfer_ci \
        --priv results/ablation/priv_critique/per_step_scores.json \
        --nogt results/ablation/nogt_critique/per_step_scores.json \
        --n_boot 10000 --seed 0
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def _load(path: str):
    with open(path) as f:
        d = json.load(f)
    return (np.asarray(d["y_true"], dtype=int),
            np.asarray(d["y_score"], dtype=float),
            np.asarray(d["y_seq"], dtype=int))


def _auc(y_true, y_score):
    """roc_auc, or None when a resample lands on a single class."""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv", required=True, help="per_step_scores.json for the privileged cell")
    ap.add_argument("--nogt", required=True, help="per_step_scores.json for the no-GT cell")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional path to write the result JSON")
    args = ap.parse_args()

    yt_p, ys_p, seq_p = _load(args.priv)
    yt_n, ys_n, seq_n = _load(args.nogt)

    # The two cells must be the SAME eval set in the same order for a paired test.
    if not (len(yt_p) == len(yt_n) and np.array_equal(yt_p, yt_n) and np.array_equal(seq_p, seq_n)):
        raise SystemExit(
            "priv and nogt per-step arrays are not aligned (different eval set/order). "
            "Re-run both cells on the same --dataset/--max_samples before computing the CI.")
    y_true, seq = yt_p, seq_p

    auc_priv = _auc(y_true, ys_p)
    auc_nogt = _auc(y_true, ys_n)
    if auc_priv is None or auc_nogt is None:
        raise SystemExit("roc_auc undefined (single-class eval) — cannot bootstrap.")
    gap = auc_priv - auc_nogt

    # clustered paired bootstrap: resample whole sequences, shared between cells
    seqs = np.unique(seq)
    by_seq = {s: np.where(seq == s)[0] for s in seqs}
    rng = np.random.default_rng(args.seed)

    gaps, aucs_p, aucs_n, skipped = [], [], [], 0
    for _ in range(args.n_boot):
        chosen = rng.choice(seqs, size=len(seqs), replace=True)
        idx = np.concatenate([by_seq[s] for s in chosen])
        ap_ = _auc(y_true[idx], ys_p[idx])
        an_ = _auc(y_true[idx], ys_n[idx])
        if ap_ is None or an_ is None:
            skipped += 1
            continue
        gaps.append(ap_ - an_); aucs_p.append(ap_); aucs_n.append(an_)
    gaps = np.asarray(gaps)

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    # one-sided bootstrap p-value for H0: gap <= 0 (priv does NOT beat nogt)
    p_one_sided = float(np.mean(gaps <= 0))

    result = {
        "auc_priv": round(float(auc_priv), 4),
        "auc_nogt": round(float(auc_nogt), 4),
        "gap_priv_minus_nogt": round(float(gap), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_one_sided_priv_le_nogt": round(p_one_sided, 4),
        "auc_priv_ci95": [round(float(np.percentile(aucs_p, 2.5)), 4),
                          round(float(np.percentile(aucs_p, 97.5)), 4)],
        "auc_nogt_ci95": [round(float(np.percentile(aucs_n, 2.5)), 4),
                          round(float(np.percentile(aucs_n, 97.5)), 4)],
        "n_seqs": int(len(seqs)),
        "n_steps": int(len(y_true)),
        "n_boot": args.n_boot,
        "n_boot_skipped_single_class": skipped,
        "significant_at_95": bool(lo > 0 or hi < 0),
    }
    print(json.dumps(result, indent=2))
    verdict = ("priv > nogt (gap CI excludes 0)" if lo > 0 else
               "nogt > priv (gap CI excludes 0)" if hi < 0 else
               "INDISTINGUISHABLE — gap CI includes 0; the transfer null is not "
               "statistically distinguishable from zero")
    print(f"\n→ {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
