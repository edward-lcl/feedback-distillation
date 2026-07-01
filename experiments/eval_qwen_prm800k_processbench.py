"""Evaluate Qwen2.5-Math-7B-PRM800K on ProcessBench JSONL.

This is a public-baseline evaluator for the Phase B result. It follows the
Qwen model-card convention: join response steps with ``<extra_0>`` and read the
positive-class probability at each separator token. For ProcessBench error
detection, we use ``1 - p_positive`` as the error score.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from evaluation.processbench import _compute_metrics


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def step_positive_probs(logits: torch.Tensor, token_mask: torch.Tensor) -> list[float]:
    probabilities = F.softmax(logits, dim=-1)
    masked = probabilities[token_mask]
    if masked.numel() == 0:
        return []
    return masked[:, 1].detach().float().cpu().tolist()


def score_sequence(model, tokenizer, sample: dict, max_length: int | None) -> tuple[list[float], bool]:
    steps = [step["text"] for step in sample["steps"]]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(model.device)
    truncated = max_length is not None and int(encoded["input_ids"].shape[1]) >= max_length

    step_sep_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(step_sep_ids) != 1:
        raise ValueError(f"Expected <extra_0> to be one token, got ids={step_sep_ids}")
    token_mask = input_ids[0] == step_sep_ids[0]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs[0][0]
    scores = step_positive_probs(logits, token_mask)
    return scores, truncated


def evaluate(args: argparse.Namespace) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=args.revision,
    )
    model = AutoModel.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=args.revision,
    ).eval()

    y_true = []
    y_pred = []
    y_score = []
    sequence_records = []
    warnings = []
    n_truncated = 0
    n_bad_score_count = 0

    with open(args.dataset) as f:
        for seq_idx, line in enumerate(f):
            if args.max_samples is not None and seq_idx >= args.max_samples:
                break
            sample = json.loads(line)
            steps = sample["steps"]
            seq_start = len(y_true)
            true_labels = [int(bool(step.get("is_error", False))) for step in steps]
            true_first = next((i for i, label in enumerate(true_labels) if label), None)
            sequence_records.append({
                "start": seq_start,
                "end": seq_start + len(steps),
                "true_first": true_first,
                "pred_first": None,
            })

            try:
                positive_probs, truncated = score_sequence(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    max_length=args.max_length,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise
                raise

            if truncated:
                n_truncated += 1
            if len(positive_probs) != len(steps):
                n_bad_score_count += 1
                warnings.append(
                    f"score_count_mismatch seq={seq_idx}: got {len(positive_probs)} "
                    f"scores for {len(steps)} steps; padding missing scores as 0.5"
                )
                if len(positive_probs) < len(steps):
                    positive_probs = positive_probs + [0.5] * (len(steps) - len(positive_probs))
                else:
                    positive_probs = positive_probs[:len(steps)]

            for step_idx, (is_error, p_positive) in enumerate(zip(true_labels, positive_probs)):
                error_score = 1.0 - float(p_positive)
                pred_is_error = error_score >= args.threshold
                y_true.append(is_error)
                y_score.append(error_score)
                y_pred.append(int(pred_is_error))
                if pred_is_error and sequence_records[-1]["pred_first"] is None:
                    sequence_records[-1]["pred_first"] = step_idx

            if args.progress_every and (seq_idx + 1) % args.progress_every == 0:
                print(f"scored {seq_idx + 1} sequences / {len(y_true)} steps", flush=True)

    metrics = _compute_metrics(y_true, y_pred, y_score, sequence_records)
    raw_y_true = metrics.pop("raw_y_true", [])
    raw_y_score = metrics.pop("raw_y_score", [])
    metrics.update({
        "model": args.model,
        "revision": args.revision,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "threshold": args.threshold,
        "score_convention": "error_score = 1 - p_positive_at_<extra_0>",
        "n_truncated_sequences": n_truncated,
        "n_score_count_mismatches": n_bad_score_count,
        "warnings": (metrics.get("warnings") or []) + warnings[:20],
    })
    return metrics, raw_y_true, raw_y_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-PRM800K")
    parser.add_argument(
        "--revision",
        default="9d6e292f6ccfd474fa44461ce6d5b80d08d8f3c7",
        help="Pinned Hugging Face revision for reproducible public-baseline eval.",
    )
    parser.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--results_dir", default="results/diagnostics/qwen_prm800k_math1000")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    metrics, raw_y_true, raw_y_score = evaluate(args)
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.results_dir, "processbench_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.results_dir, "per_step_scores.json"), "w") as f:
        json.dump({"y_true": raw_y_true, "y_score": raw_y_score}, f, indent=2)

    warnings = metrics.get("warnings") or []
    if warnings:
        print("EVAL WARNINGS:")
        for warning in warnings[:20]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
