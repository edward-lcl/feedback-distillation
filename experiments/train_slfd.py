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


# Ablation -> (loss_flags [LM, hidden, score, logit], default score_loss).
# Existing Phase B jobs used --score_loss directly; these names are convenient
# wrappers for the newer distillation-method arms from main.
ABLATIONS = {
    "score_critique": ([True, False, True, False], "mse"),
    "score_only": ([False, False, True, False], "mse"),
    "verdict": ([False, False, True, False], "verdict"),
    "soft": ([False, False, True, False], "soft"),
    "logit_kd": ([False, False, True, True], "mse"),
}


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
    parser.add_argument("--train_dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                        help="Weight precision for training. 'auto' uses bf16 on MPS "
                             "(fp16 NaNs there) and the device default elsewhere.")
    parser.add_argument("--model_lr", type=float, default=1e-4)
    parser.add_argument("--score_lr", type=float, default=5e-5)
    parser.add_argument("--align_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=1.0)
    parser.add_argument("--score_loss", choices=["mse", "bce", "rank", "bce_rank", "verdict", "soft"],
                        default=None,
                        help="Score-head objective. If omitted, the selected ablation chooses it.")
    parser.add_argument("--error_weight", type=float, default=1.0,
                        help="Per-sample weight for error steps when --score_loss=bce.")
    parser.add_argument("--rank_margin", type=float, default=1.0,
                        help="Margin for pairwise score ranking losses.")
    parser.add_argument("--balanced_batches", action="store_true",
                        help="Oversample error/clean examples so each batch contains both classes.")
    parser.add_argument("--ablation", choices=list(ABLATIONS.keys()),
                        default="score_critique",
                        help="score_critique = scorer + NL critique (L_score+L_LM); "
                             "score_only = scorer alone; verdict = hard BCE target; "
                             "soft = BCE on p=(score+1)/2; logit_kd = score loss + "
                             "online token-level KD from a local same-family teacher.")
    parser.add_argument("--kd_teacher", default=None,
                        help="Local HF teacher for online logit-KD (ablation=logit_kd).")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Softmax temperature for the logit-KD loss.")
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
    use_bf16 = args.train_dtype == "bf16" or (args.train_dtype == "auto" and is_mps())
    use_fp32 = args.train_dtype == "fp32"
    if use_bf16 and student.model.dtype != torch.bfloat16:
        student.model.to(torch.bfloat16)
        print("Training in bfloat16 (numerically stable and memory efficient on MPS).")
    elif use_fp32 and student.model.dtype != torch.float32:
        student.model.float()
        print("Training in float32.")

    loss_flags, ablation_score_loss = ABLATIONS[args.ablation]
    score_loss = args.score_loss or ablation_score_loss
    trainer_score_loss = "bce" if score_loss == "verdict" else score_loss
    print(f"Ablation: {args.ablation}  (loss_flags={loss_flags}, score_loss={score_loss})")

    teacher = None
    if loss_flags[3]:
        from models.teacher import TeacherModel
        teacher = TeacherModel(args.kd_teacher, dev_mode=args.dev_mode)
        sv = getattr(student.tokenizer, "vocab_size", None)
        tv = getattr(teacher.tokenizer, "vocab_size", None)
        if sv is not None and tv is not None and sv != tv:
            print(f"WARNING: vocab mismatch (student {sv} vs teacher {tv}); "
                  "logit-KD truncates to min-vocab and may corrupt the signal. "
                  "Use a same-family teacher/student pair.")

    # teacher=None for offline distillation; labels are already in the dataset.
    trainer = SLFDTrainer(student, teacher=teacher, dataset=dataset,
                          loss_flags=loss_flags, dev_mode=args.dev_mode,
                          model_lr=args.model_lr, score_lr=args.score_lr,
                          align_lr=args.align_lr, weight_decay=args.weight_decay,
                          lm_weight=args.lm_weight,
                          score_weight=args.score_weight,
                          hidden_weight=args.hidden_weight,
                          score_loss=trainer_score_loss,
                          error_weight=args.error_weight,
                          rank_margin=args.rank_margin,
                          balanced_batches=args.balanced_batches,
                          kd_temperature=args.kd_temperature)

    summary = trainer.train(epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps)
    trainer.save_checkpoint(args.checkpoint)

    print(json.dumps({"steps": summary["steps"],
                      "final_losses": {k: (v[-1] if v else None)
                                       for k, v in summary["loss_history"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
