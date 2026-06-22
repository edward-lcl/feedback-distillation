"""
SLFD training loop — distills step-level feedback generation from teacher to student.
"""
import random
import torch
from torch.optim import AdamW
from .losses import (compute_lm_loss, compute_hidden_loss, compute_scoring_loss,
                     compute_score_loss, compute_logit_kd_loss)
from .threshold_policy import AdaptiveWeightedKDPolicyEMA
from .loss_config import LossConfig


class SLFDTrainer:
    """
    Trains the student to generate step-level feedback and correctness verdicts.
    Teacher is frozen throughout.
    """

    def __init__(self, student, teacher, dataset: list[dict], loss_flags=None,
                 kd_temperature=2.0, device=None, dev_mode=False):
        import torch.nn as nn
        from models.device import is_dev_mode, best_device
        self.student = student
        self.teacher = teacher
        self.dataset = dataset

        self.kd_temperature = kd_temperature
        self.dev_mode = is_dev_mode(dev_mode)
        if self.dev_mode:
            print("DEV MODE: reduced batch/steps for local Apple Silicon")
        # Match the student's device (MPS on Apple Silicon, CUDA, or CPU) — the
        # student model is already placed there, so inputs must follow.
        self.device = device or best_device()
        self.loss_flags = loss_flags or [True, True, True, False]  # LM, hidden, score, logit
        # Online logit-KD (loss_flags[3]) needs a live teacher for its logits.
        if self.loss_flags[3] and self.teacher is None:
            raise ValueError("logit-KD loss is enabled (loss_flags[3]) but no teacher was "
                             "passed — load a local same-family teacher (--kd_teacher).")
        self.loss_config = LossConfig(self.loss_flags)
        self.threshold_policy = AdaptiveWeightedKDPolicyEMA(
            num_losses=self.loss_config.num_enabled,
            device=str(self.device),
        )

        # Hidden alignment projection.
        # Offline distillation: the labeled dataset already carries the teacher's
        # score + feedback, so the active losses (LM, score) need no live teacher.
        # The teacher is only consulted here to size the hidden-alignment layer,
        # which feeds the optional hidden loss. When teacher is None we skip the
        # probe entirely and train fully locally (no 72B load required).
        if self.teacher is not None:
            self.teacher.model.eval()
            with torch.no_grad():
                dummy = "test"
                s_out = self.student.model(**self.student.tokenizer(dummy, return_tensors="pt").to(self.device), output_hidden_states=True, return_dict=True)
                t_out = self.teacher.model(**self.teacher.tokenizer(dummy, return_tensors="pt").to(self.device), output_hidden_states=True, return_dict=True)
                s_dim = s_out.hidden_states[-1].size(-1)
                t_dim = t_out.hidden_states[-1].size(-1)
            self.align_hidden = (
                nn.Linear(s_dim, t_dim) if s_dim != t_dim else nn.Identity()
            ).to(self.device).to(torch.float32)
        else:
            self.align_hidden = nn.Identity().to(self.device)

        # Train only what's trainable: LoRA adapters (base is frozen by peft) +
        # the score head + the alignment layer. LoRA tolerates a much higher LR
        # than the old full-FT 5e-7.
        trainable_model_params = [p for p in self.student.model.parameters() if p.requires_grad]
        self.optimizer = AdamW([
            {"params": trainable_model_params, "lr": 1e-4, "weight_decay": 0.01},
            {"params": self.align_hidden.parameters(), "lr": 1e-6, "weight_decay": 0.01},
        ], betas=(0.9, 0.999), eps=1e-8)

    def train(self, epochs: int = 2, batch_size: int = 4, max_steps: int = None) -> dict:
        if self.dev_mode:
            batch_size = 1
            if max_steps is None:
                max_steps = 50
            print(f"DEV MODE: batch_size={batch_size}, max_steps={max_steps}")
        random.shuffle(self.dataset)
        loss_history = {"lm_loss": [], "hidden_loss": [], "scoring_loss": [], "logit_loss": []}
        step = 0

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            for batch_start in range(0, len(self.dataset), batch_size):
                if max_steps and step >= max_steps:
                    break
                batch = self.dataset[batch_start: batch_start + batch_size]
                batch_losses = {"lm_loss": [], "hidden_loss": [], "logit_loss": []}

                for sample in batch:
                    problem = sample.get("problem", "")
                    step_text = sample.get("step_text", "")
                    solution_prefix = sample.get("solution_prefix", "")
                    teacher_score = sample.get("score", 0.0)
                    teacher_feedback = sample.get("feedback", "")
                    if not problem or not step_text:
                        continue

                    # L_feedback_LM — natural-language critique loss.
                    if self.loss_flags[0]:
                        input_ids, labels, attn_mask = self.student.prepare_step_inputs_and_labels(
                            problem, solution_prefix, step_text, teacher_feedback, teacher_score
                        )
                        lm_out = self.student.model(
                            input_ids=input_ids.to(self.device),
                            attention_mask=attn_mask.to(self.device),
                            labels=labels.to(self.device),
                            return_dict=True,
                        )
                        batch_losses["lm_loss"].append(lm_out.loss)

                    # L_logit — online KD: soft KL toward the teacher's distribution
                    # over its (privileged) critique. The soft counterpart of L_LM;
                    # needs a live same-family teacher. Scaled down (see LossConfig).
                    if self.loss_flags[3] and self.teacher is not None:
                        kd = compute_logit_kd_loss(
                            self.student, self.teacher, problem, solution_prefix,
                            step_text, teacher_feedback, teacher_score,
                            temperature=self.kd_temperature,
                        )
                        batch_losses["logit_loss"].append(kd * LossConfig.SCALES["logit_loss"])

                aggregated = {k: torch.stack(v).mean() for k, v in batch_losses.items() if v}
                if not aggregated:
                    continue

                total_loss = sum(aggregated.values())
                # Skip non-finite steps so a single NaN/Inf can't corrupt weights
                # (fp16 on MPS overflows easily). The checkpoint stays clean.
                if not torch.isfinite(total_loss):
                    print(f"  step {step}: non-finite loss ({total_loss.item()}), skipping")
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                for k, v in aggregated.items():
                    loss_history[k].append(float(v.item()))

                step += 1
                if step % 10 == 0:
                    parts = " | ".join(f"{k}={v[-1]:.4f}" for k, v in loss_history.items() if v)
                    print(f"  step {step}: {parts}")

        return {"loss_history": loss_history, "steps": step}

    def save_checkpoint(self, path: str):
        """Bundle the trained student weights, score head, and alignment layer.

        The score head is the component eval actually reads — saving only the
        base model (as the old README implied) would discard the learned scorer.
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.student.model.state_dict(),
            "align_hidden": self.align_hidden.state_dict(),
        }, path)
        print(f"Saved checkpoint → {path}")
