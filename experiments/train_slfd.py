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


# Ablation -> (loss_flags [LM, hidden, score, logit], score_loss_mode).
# The score_loss_mode is the B4 distillation-method knob: how much of the
# teacher's scalar we actually distill.
#   score_critique / score_only — original: MSE on the exact scalar (point estimate)
#   verdict — B4c: BCE on the hard binary verdict (correct iff score>=0); drops the
#             free-text critique entirely (the 1.5B can't reproduce a 26B's prose)
#   soft    — B4a-offline: BCE on the soft prob p=(score+1)/2 — keeps the teacher's
#             CONFIDENCE as a distribution instead of collapsing it to a point.
# (True token-level logit-KL from the teacher needs a LIVE teacher with logit
#  access — the served teacher exposes none; see RUNBOOK_PHASE_B.md B4a.)
ABLATIONS = {
    "score_critique": ([True, False, True, False], "mse"),
    "score_only":     ([False, False, True, False], "mse"),
    "verdict":        ([False, False, True, False], "verdict"),
    "soft":           ([False, False, True, False], "soft"),
    # B4a-online: score MSE + soft KL toward a LIVE same-family teacher's critique
    # distribution (needs --kd_teacher). The soft counterpart of score_critique's
    # hard token-CE. The served Gemma-4 exposes no logits → use a LOCAL Gemma-2.
    "logit_kd":       ([False, False, True, True], "mse"),
}


def set_seed(seed: int):
    """Seed every RNG so multi-seed retrains are reproducible (D1 rigor)."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--train_dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                        help="Weight precision for training. 'auto' uses bf16 on MPS "
                             "(fp16 NaNs there) and the device default elsewhere.")
    parser.add_argument("--ablation", choices=list(ABLATIONS.keys()),
                        default="score_critique",
                        help="score_critique = scorer + NL critique (L_score+L_LM); "
                             "score_only = scorer alone; verdict = BCE on the hard "
                             "binary verdict (B4c); soft = BCE on the soft prob "
                             "p=(score+1)/2, a distribution target (B4a-offline).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed all RNGs for a reproducible run. Omit for the "
                             "original stochastic behavior; set per-run for D1 multi-seed.")
    parser.add_argument("--kd_teacher", default=None,
                        help="Local HF teacher for online logit-KD (ablation=logit_kd). "
                             "Use a SAME-FAMILY model so vocabs align (e.g. a Gemma teacher "
                             "for a Gemma student). In --dev_mode, defaults to the dev teacher.")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Softmax temperature for the logit-KD loss (Hinton KD).")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seeded all RNGs with {args.seed}")

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
    loss_flags, score_loss_mode = ABLATIONS[args.ablation]
    print(f"Ablation: {args.ablation}  (loss_flags={loss_flags}, score_loss_mode={score_loss_mode})")

    # Online logit-KD needs a LIVE teacher for its logits (offline labels carry
    # only the scalar + critique text, not a distribution). Everything else is
    # offline distillation: teacher=None, labels already in the dataset.
    teacher = None
    if loss_flags[3]:
        from models.teacher import TeacherModel
        teacher = TeacherModel(args.kd_teacher, dev_mode=args.dev_mode)
        sv = getattr(student.tokenizer, "vocab_size", None)
        tv = getattr(teacher.tokenizer, "vocab_size", None)
        if sv is not None and tv is not None and sv != tv:
            print(f"⚠️  vocab mismatch (student {sv} vs teacher {tv}) — logit-KD truncates to "
                  f"min-vocab, which corrupts the signal. Use a SAME-FAMILY teacher "
                  f"(e.g. Gemma-2-2B for a Gemma student).")
    trainer = SLFDTrainer(student, teacher=teacher, dataset=dataset,
                          loss_flags=loss_flags, score_loss_mode=score_loss_mode,
                          kd_temperature=args.kd_temperature, dev_mode=args.dev_mode)

    summary = trainer.train(epochs=args.epochs, batch_size=args.batch_size, max_steps=args.max_steps)
    trainer.save_checkpoint(args.checkpoint)

    print(json.dumps({"steps": summary["steps"],
                      "final_losses": {k: (v[-1] if v else None)
                                       for k, v in summary["loss_history"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
