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
        student.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate_processbench(student, args.dataset, args.max_samples)
    print(json.dumps(results, indent=2))

    with open(f"{args.results_dir}/processbench_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
