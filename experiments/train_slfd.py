"""
Train the SLFD student from a teacher-labeled dataset and save a checkpoint.

This closes the label -> flatten -> train -> save loop. The teacher is NOT
loaded here: the labeled JSONL already carries the teacher's per-step score and
feedback, so distillation runs fully locally on the student alone (no 72B load).

Usage:
    # Train from labeled JSONL (output of data.label_pipeline):
    python -m experiments.train_slfd \
        --dataset data/labeled/train_labeled.jsonl \
        --checkpoint checkpoints/slfd_student.pt \
        --epochs 2 --batch_size 4

    # Local Apple Silicon smoke run:
    DEV_MODE=1 python -m experiments.train_slfd \
        --dataset /tmp/slfd_labeled_smoke.jsonl \
        --checkpoint /tmp/slfd_student_smoke.pt --dev_mode
"""
import argparse
import json

import torch
from models.student import StudentModel
from training.slfd_trainer import SLFDTrainer
from data.flatten_labels import flatten_labeled_file, flatten_labeled_records


def load_dataset(path: str) -> list[dict]:
    """Accept either per-solution labeled JSONL or already-flat per-step JSONL."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if rows and "steps" in rows[0]:
        return flatten_labeled_records(rows)          # per-solution -> per-step
    return rows                                        # already flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Labeled JSONL from data.label_pipeline (per-solution or per-step).")
    parser.add_argument("--student_model", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/slfd_student.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dev_mode", action="store_true",
                        help="Use smaller models for local Apple Silicon development.")
    parser.add_argument("--train_dtype", choices=["auto", "fp32", "fp16"], default="auto",
                        help="Weight precision for training. 'auto' uses fp32 on MPS "
                             "(fp16 NaNs there) and the device default elsewhere.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} per-step training examples from {args.dataset}")
    if not dataset:
        raise SystemExit("Empty training set — check the labeled dataset path/format.")

    student = StudentModel(args.student_model, dev_mode=args.dev_mode)

    # Stabilize training precision. fp16 on MPS overflows to NaN; AdamW has no
    # fp32 master copy here, so we train the weights themselves in fp32.
    from models.device import is_mps
    use_fp32 = args.train_dtype == "fp32" or (args.train_dtype == "auto" and is_mps())
    if use_fp32 and student.model.dtype != torch.float32:
        student.model.float()
        print("Training in float32 (numerically stable on MPS).")
    # teacher=None: offline distillation, labels are already in the dataset.
    trainer = SLFDTrainer(student, teacher=None, dataset=dataset, dev_mode=args.dev_mode)

    summary = trainer.train(epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps)
    trainer.save_checkpoint(args.checkpoint)

    print(json.dumps({"steps": summary["steps"],
                      "final_losses": {k: (v[-1] if v else None)
                                       for k, v in summary["loss_history"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
