"""
Main SLFD evaluation: ProcessBench step-error detection.

Usage:
    python -m experiments.run_processbench \
        --student_model Qwen/Qwen2.5-1.5B-Instruct \
        --checkpoint checkpoints/slfd_student.pt \
        --dataset data/processbench_test.jsonl \
        --max_samples 500
"""
import argparse
import json
from models.student import StudentModel
from evaluation.processbench import evaluate_processbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", default="results/processbench")
    parser.add_argument("--dev_mode", action="store_true",
                        help="Use smaller models for local Apple Silicon development.")
    args = parser.parse_args()

    import os; os.makedirs(args.results_dir, exist_ok=True)

    student = StudentModel(args.student_model, dev_mode=args.dev_mode)
    if args.checkpoint:
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "score_head" in ckpt:
            # Bundled checkpoint from train_slfd: restore base model AND the
            # score head (the component eval reads to predict step errors).
            student.model.load_state_dict(ckpt["model"], strict=False)
            student.score_head.load_state_dict(ckpt["score_head"])
            print(f"Loaded checkpoint (model + score head): {args.checkpoint}")
        else:
            # Legacy: a bare model state_dict.
            student.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded checkpoint (model only — no score head): {args.checkpoint}")

    results = evaluate_processbench(student, args.dataset, args.max_samples)

    # Split the raw per-step arrays into a sidecar so processbench_results.json
    # stays human-readable. The sidecar feeds experiments.transfer_ci (paired
    # bootstrap CI on the priv−nogt roc_auc gap — D1 rigor).
    per_step = results.pop("_per_step", None)
    print(json.dumps(results, indent=2))

    with open(f"{args.results_dir}/processbench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if per_step:
        with open(f"{args.results_dir}/per_step_scores.json", "w") as f:
            json.dump(per_step, f)

    # Loud, immediate self-check — so an agent re-running this sees a broken cell
    # the moment it finishes, rather than reading a near-zero F1 as a real result.
    import sys
    warnings = results.get("warnings") or []
    if warnings:
        print("\n" + "!" * 72, file=sys.stderr)
        print("⚠️  EVAL HEALTH WARNING — this cell's F1 is NOT trustworthy as capability:",
              file=sys.stderr)
        for w in warnings:
            print(f"  • {w}", file=sys.stderr)
        print(f"  → roc_auc={results.get('roc_auc')}  pr_auc={results.get('pr_auc')}  "
              f"pred_error_rate={results.get('pred_error_rate'):.3f}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
    else:
        print(f"✓ eval health OK — roc_auc={results.get('roc_auc')} "
              f"pr_auc={results.get('pr_auc')} pred_error_rate={results.get('pred_error_rate'):.3f}")


if __name__ == "__main__":
    main()
