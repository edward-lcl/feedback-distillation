"""
ProcessBench evaluation harness.
Measures step-error detection F1 and first-error-step accuracy.
"""
import json
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_processbench(student, dataset_path: str, max_samples: int = None) -> dict:
    """
    dataset_path: JSONL with {problem, steps: [{text, is_error (bool)}]}
    Returns: {f1, precision, recall, first_error_acc}
    """
    from data.step_segmentation import segment_steps

    y_true, y_pred = [], []
    first_error_correct = 0
    total_sequences = 0

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
                pred_is_error = float(score_logit.item()) < 0.0
                pred_labels.append(int(pred_is_error))
                y_true.append(int(step.get("is_error", False)))
                y_pred.append(int(pred_is_error))
                prefix += step["text"] + "\n"

            # First-error-step accuracy
            true_first = next((j for j, s in enumerate(steps) if s.get("is_error")), None)
            pred_first = next((j for j, p in enumerate(pred_labels) if p), None)
            if true_first == pred_first:
                first_error_correct += 1
            total_sequences += 1

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "first_error_acc": first_error_correct / max(1, total_sequences),
    }
