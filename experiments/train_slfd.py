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
import random

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Labeled JSONL from data.label_pipeline (per-solution or per-step).")
    parser.add_argument("--student_model", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/slfd_student.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for LoRA init, data order, and torch RNGs.")
    parser.add_argument("--dev_mode", action="store_true",
                        help="Use smaller models for local Apple Silicon development.")
    parser.add_argument("--train_dtype", choices=["auto", "fp32", "fp16"], default="auto",
                        help="Weight precision for training. 'auto' uses fp32 on MPS "
                             "(fp16 NaNs there) and the device default elsewhere.")
    parser.add_argument("--model_lr", type=float, default=1e-4)
    parser.add_argument("--score_lr", type=float, default=5e-5)
    parser.add_argument("--align_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=1.0)
    parser.add_argument("--score_loss", choices=["mse", "bce", "rank", "bce_rank"], default="mse")
    parser.add_argument("--error_weight", type=float, default=1.0,
                        help="Per-sample weight for error steps when --score_loss=bce.")
    parser.add_argument("--rank_margin", type=float, default=1.0,
                        help="Margin for pairwise score ranking losses.")
    parser.add_argument("--balanced_batches", action="store_true",
                        help="Oversample error/clean examples so each batch contains both classes.")
    parser.add_argument("--ablation", choices=["score_critique", "score_only"],
                        default="score_critique",
                        help="score_critique = scorer + NL critique (L_score+L_LM); "
                             "score_only = scorer alone (the headline ablation).")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

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
    # Ablation -> loss_flags [LM, hidden, score, logit]. score_only drops the
    # critique (LM) loss so we can isolate whether distilling the NL critique helps.
    loss_flags = ([True, False, True, False] if args.ablation == "score_critique"
                  else [False, False, True, False])
    print(f"Ablation: {args.ablation}  (loss_flags={loss_flags})")
    # teacher=None: offline distillation, labels are already in the dataset.
    trainer = SLFDTrainer(student, teacher=None, dataset=dataset,
                          loss_flags=loss_flags, dev_mode=args.dev_mode,
                          model_lr=args.model_lr, score_lr=args.score_lr,
                          align_lr=args.align_lr, weight_decay=args.weight_decay,
                          lm_weight=args.lm_weight,
                          score_weight=args.score_weight,
                          hidden_weight=args.hidden_weight,
                          score_loss=args.score_loss,
                          error_weight=args.error_weight,
                          rank_margin=args.rank_margin,
                          balanced_batches=args.balanced_batches)

    summary = trainer.train(epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps)
    trainer.save_checkpoint(args.checkpoint)

    print(json.dumps({"steps": summary["steps"],
                      "final_losses": {k: (v[-1] if v else None)
                                       for k, v in summary["loss_history"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
