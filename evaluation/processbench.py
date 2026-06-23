"""
ProcessBench evaluation harness.
Measures step-error detection F1 and first-error-step accuracy.

NOTE (2026-06-17): F1/precision/recall/first_error_acc all depend on the fixed
decision threshold `score_logit < 0`. A student whose score head is shifted so
it rarely crosses 0 predicts "no error" almost everywhere — recall→0 collapses
F1, while first_error_acc stays at the *base rate* of error-free sequences
(pred_first=None trivially matches every clean sequence). That is exactly how
two models can share ~identical first_error_acc while their F1 differs 5×.
To compare models apples-to-apples, read the THRESHOLD-FREE numbers (roc_auc,
pr_auc) and the split (error_recall / clean_specificity / pred_error_rate),
not F1 at a fixed cutoff. See the split keys below.
"""
import json
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_processbench(student, dataset_path: str, max_samples: int = None) -> dict:
    """
    dataset_path: JSONL with {problem, steps: [{text, is_error (bool)}]}
    Returns threshold-dependent metrics (f1/precision/recall/first_error_acc)
    AND threshold-free / split diagnostics:
      - roc_auc, pr_auc        : ranking quality of raw scores, NO fixed cutoff
      - error_recall           : frac of true error-steps caught (= recall)
      - clean_specificity      : frac of correct steps correctly left unflagged
      - pred_error_rate        : frac of ALL steps flagged as error (→0 ⇒ silent/degenerate)
      - first_error_acc_errs   : first-error localisation on ERRONEOUS sequences only
      - clean_seq_acc          : frac of error-free sequences correctly left unflagged
    """
    from data.step_segmentation import segment_steps
    from models.device import is_mps
    import gc
    # Same MPS unified-memory accumulation that OOM-kills training also hits a
    # 1000-sample eval (thousands of score_step forwards): the caching allocator
    # grows its pool and SIGKILLs the run despite no_grad. Hand it back periodically.
    on_mps = is_mps()

    y_true, y_pred, y_score, y_seq = [], [], [], []
    first_error_correct = 0
    total_sequences = 0
    # split first-error accuracy so a "predict nothing" model can't bank the base rate
    err_seqs = err_seqs_correct = 0
    clean_seqs = clean_seqs_correct = 0

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            problem = sample["problem"]
            steps = sample["steps"]

            prefix = ""
            pred_labels = []
            for step in steps:
                # Use the SAME representation training optimizes — the score head
                # on the prompt's step-boundary token (score_step) — not the
                # generated response. no_grad: eval needs no graph (and avoids a
                # per-step grad-graph memory leak).
                with torch.no_grad():
                    score_logit = student.score_step(problem, prefix, step["text"])
                logit = float(score_logit.item())
                pred_is_error = logit < 0.0
                pred_labels.append(int(pred_is_error))
                y_true.append(int(step.get("is_error", False)))
                y_pred.append(int(pred_is_error))
                # error-ness score for ranking metrics: more negative logit = more error-like
                y_score.append(-logit)
                y_seq.append(i)   # sequence id → enables a clustered paired bootstrap
                prefix += step["text"] + "\n"

            if on_mps and i % 25 == 0:
                gc.collect()
                torch.mps.empty_cache()

            # First-error-step accuracy (overall + split by whether the seq has an error)
            true_first = next((j for j, s in enumerate(steps) if s.get("is_error")), None)
            pred_first = next((j for j, p in enumerate(pred_labels) if p), None)
            if true_first == pred_first:
                first_error_correct += 1
            if true_first is None:
                clean_seqs += 1
                clean_seqs_correct += int(pred_first is None)
            else:
                err_seqs += 1
                err_seqs_correct += int(true_first == pred_first)
            total_sequences += 1

    # threshold-free ranking metrics (need both classes present)
    has_both = len(set(y_true)) > 1
    roc_auc = roc_auc_score(y_true, y_score) if has_both else None
    pr_auc = average_precision_score(y_true, y_score) if has_both else None

    n_steps = max(1, len(y_true))
    n_pos = sum(y_true)
    n_neg = n_steps - n_pos
    # clean_specificity = TN / (TN+FP) over steps
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    pred_error_rate = sum(y_pred) / n_steps
    error_recall = (sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / n_pos) if n_pos else None

    # --- runtime self-check: surface degeneracy LOUDLY so an agent re-running
    # this notices immediately instead of trusting a near-zero F1 as "capability". ---
    warnings = []
    if pred_error_rate < 0.02:
        warnings.append(
            f"SILENT COLLAPSE: predicts 'error' on only {pred_error_rate:.1%} of steps "
            f"(recall≈{(error_recall or 0):.2f}). F1/first_error_acc are MEANINGLESS here — "
            f"the cell banks the error-free base rate for free. Compare cells on roc_auc/pr_auc, "
            f"and check the score-head threshold/calibration before trusting any gap.")
    if pred_error_rate > 0.98:
        warnings.append(
            f"ALWAYS-FLAG COLLAPSE: predicts 'error' on {pred_error_rate:.1%} of steps — "
            f"high recall is trivial; compare on roc_auc/pr_auc, not recall/F1.")
    if not has_both:
        warnings.append("DEGENERATE EVAL: only one true class present in this slice — roc_auc/pr_auc undefined.")

    return {
        # --- threshold-dependent (fixed cutoff at logit<0): interpret with care ---
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "first_error_acc": first_error_correct / max(1, total_sequences),
        # --- threshold-free: the apples-to-apples comparison across models ---
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        # --- split diagnostics: expose silent/degenerate collapse ---
        "error_recall": error_recall,
        "clean_specificity": (tn / n_neg) if n_neg else None,
        "pred_error_rate": pred_error_rate,
        "first_error_acc_errs": (err_seqs_correct / err_seqs) if err_seqs else None,
        "clean_seq_acc": (clean_seqs_correct / clean_seqs) if clean_seqs else None,
        "n_steps": len(y_true),
        "n_error_steps": n_pos,
        # --- self-check: non-empty ⇒ this cell's F1 is not trustworthy as capability ---
        "warnings": warnings,
        # --- raw per-step arrays for a paired bootstrap CI (D1). run_processbench
        #     splits these into a per_step_scores.json sidecar so the main results
        #     stay readable. y_seq groups steps by problem for a clustered bootstrap. ---
        "_per_step": {"y_true": y_true, "y_score": y_score, "y_seq": y_seq},
    }
