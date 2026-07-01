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
import gc
import json
import torch
from models.device import is_mps
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def _compute_metrics(y_true, y_pred, y_score, sequence_records, y_seq=None) -> dict:
    first_error_correct = 0
    total_sequences = 0
    err_seqs = err_seqs_correct = 0
    clean_seqs = clean_seqs_correct = 0

    for record in sequence_records:
        true_first = record["true_first"]
        pred_first = record["pred_first"]
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
    best_f1 = None
    best_f1_threshold = None
    if has_both:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        f1s = []
        for precision, recall in zip(precisions, recalls):
            denom = precision + recall
            f1s.append(0.0 if denom == 0 else 2 * precision * recall / denom)
        best_idx = max(range(len(f1s)), key=lambda idx: f1s[idx])
        best_f1 = float(f1s[best_idx])
        # precision_recall_curve returns one fewer threshold than precision/recall
        # points. The final point corresponds to predicting no positives.
        if best_idx < len(thresholds):
            best_f1_threshold = float(thresholds[best_idx])

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
        # Optimistic diagnostic: best threshold chosen on this eval slice.
        # Use a held-out calibration split before treating this as a claim.
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
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
        # --- raw scores for bootstrapping ---
        "raw_y_true": y_true,
        "raw_y_score": y_score,
        "_per_step": {
            "y_true": y_true,
            "y_score": y_score,
            "y_seq": y_seq,
        } if y_seq is not None else {
            "y_true": y_true,
            "y_score": y_score,
        },
    }


def evaluate_processbench(
    student,
    dataset_path: str,
    max_samples: int = None,
    batch_size: int = 1,
) -> dict:
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
    step_inputs = []
    step_meta = []
    sequence_records = []

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            problem = sample["problem"]
            steps = sample["steps"]

            prefix = ""
            true_labels = []
            seq_start = len(step_inputs)
            for j, step in enumerate(steps):
                step_inputs.append((problem, prefix, step["text"]))
                is_error = bool(step.get("is_error", False))
                true_labels.append(int(is_error))
                step_meta.append({"seq_idx": len(sequence_records), "step_idx": j, "is_error": is_error})
                prefix += step["text"] + "\n"

            # First-error-step accuracy (overall + split by whether the seq has an error)
            true_first = next((j for j, label in enumerate(true_labels) if label), None)
            sequence_records.append({
                "start": seq_start,
                "end": len(step_inputs),
                "true_first": true_first,
                "pred_first": None,
            })

    y_true = [int(meta["is_error"]) for meta in step_meta]
    y_pred = []
    y_score = []
    y_seq = [int(meta["seq_idx"]) for meta in step_meta]
    on_mps = is_mps()
    with torch.no_grad():
        if batch_size > 1 and hasattr(student, "score_steps"):
            score_logits = student.score_steps(step_inputs, batch_size=batch_size)
            if on_mps:
                gc.collect()
                torch.mps.empty_cache()
        else:
            logits = []
            last_cleared_seq = None
            for idx, (problem, prefix, step_text) in enumerate(step_inputs):
                logits.append(student.score_step(problem, prefix, step_text))
                seq_idx = step_meta[idx]["seq_idx"]
                if on_mps and seq_idx % 25 == 0 and seq_idx != last_cleared_seq:
                    gc.collect()
                    torch.mps.empty_cache()
                    last_cleared_seq = seq_idx
            score_logits = torch.stack(logits) if logits else torch.empty(0)

    for meta, score_logit in zip(step_meta, score_logits):
        logit = float(score_logit.item())
        pred_is_error = logit < 0.0
        y_pred.append(int(pred_is_error))
        # error-ness score for ranking metrics: more negative logit = more error-like
        y_score.append(-logit)
        if pred_is_error:
            record = sequence_records[meta["seq_idx"]]
            if record["pred_first"] is None:
                record["pred_first"] = meta["step_idx"]

    return _compute_metrics(y_true, y_pred, y_score, sequence_records, y_seq=y_seq)
