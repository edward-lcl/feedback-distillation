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
    print(json.dumps(results, indent=2))

    with open(f"{args.results_dir}/processbench_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
