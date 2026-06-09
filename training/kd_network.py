"""
AmateurExpertFeedbackNetWork — orchestrates the KD loop between student and teacher.
"""

import random
import torch
import torch.nn as nn
from torch.optim import AdamW

from .loss_config import LossConfig
from .threshold_policy import AdaptiveWeightedKDPolicyEMA
from .losses import (
    compute_lm_loss,
    compute_hidden_loss,
    compute_scoring_loss,
    compute_logit_standardization,
    compute_all_losses_vectorized,
    unpack_feedback_and_scores,
)


class AmateurExpertFeedbackNetWork:
    """
    Ties together the student (amateur) and teacher (expert) models.

    During task execution:
    1. Both models generate feedback on the base model's answer.
    2. The threshold policy decides whether KD is needed.
    3. If KD triggers, the student is trained on expert-labeled data.
    4. Combined feedback is returned for the base model to use in revision.
    """

    def __init__(
        self,
        student,
        teacher,
        expert_datasets: list | None = None,
        loss_flags: list[bool] | None = None,
        device=None,
    ):
        self.student = student
        self.teacher = teacher
        self.expert_datasets = expert_datasets or []
        self.student_tokenizer = student.tokenizer
        self.teacher_tokenizer = teacher.tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student.score_head = self.student.score_head.to(torch.float32)
        self.model_dtype = torch.float32
        self.loss_flags = loss_flags if loss_flags is not None else [True, True, True, True]
        self.loss_config = LossConfig(self.loss_flags)
        self.threshold_policy = AdaptiveWeightedKDPolicyEMA(
            num_losses=self.loss_config.num_enabled,
            device=str(self.device),
        )

        for tok in [self.student_tokenizer, self.teacher_tokenizer]:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        # Build hidden-state alignment projection
        self.teacher.model.eval()
        with torch.no_grad():
            dummy = "Hello World"
            s_out = self.student.model(**self.student_tokenizer(dummy, return_tensors="pt").to(self.device), output_hidden_states=True, return_dict=True)
            t_out = self.teacher.model(**self.teacher_tokenizer(dummy, return_tensors="pt").to(self.device), output_hidden_states=True, return_dict=True)
            s_dim = s_out.hidden_states[-1].size(-1)
            t_dim = t_out.hidden_states[-1].size(-1)

        self.align_hidden = (
            nn.Linear(s_dim, t_dim) if s_dim != t_dim else nn.Identity()
        ).to(self.device).to(self.model_dtype)

        # Disable logit loss when tokenizers differ
        same_vocab = (
            self.student.model.config.vocab_size == self.teacher.model.config.vocab_size
            and self.student_tokenizer.get_vocab() == self.teacher_tokenizer.get_vocab()
        )
        self.projection_layer = nn.Identity().to(self.device)
        if not same_vocab:
            print("[Logit KD] Vocab mismatch — disabling logit_loss.")
            self.loss_config.toggle_loss("logit_loss", False)

        self.optimizer = AdamW(
            [
                {"params": self.student.model.parameters(), "lr": 5e-7, "weight_decay": 0.01},
                {"params": self.student.score_head.parameters(), "lr": 5e-5, "weight_decay": 0.01},
                {"params": self.align_hidden.parameters(), "lr": 1e-6, "weight_decay": 0.01},
                {"params": self.projection_layer.parameters(), "lr": 1e-6, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # ------------------------------------------------------------------
    # KD training loop
    # ------------------------------------------------------------------

    def knowledge_distillation_with_SFT(
        self,
        weights: list[float],
        batch_size: int = 3,
        epochs: int = 1,
        stopped: int = 20,
        is_math: bool = False,
        examples: list | None = None,
    ) -> dict:
        random.shuffle(self.expert_datasets)
        loss_names, _ = self.loss_config.get_active_losses()
        loss_history = {n: [] for n in loss_names}
        global_step = 0

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            num_samples = 0
            stop_epoch = False

            for batch_start in range(0, len(self.expert_datasets), batch_size):
                if stopped is not None and num_samples >= stopped:
                    break
                batch = self.expert_datasets[batch_start: batch_start + batch_size]
                batch_losses: dict[str, list[torch.Tensor]] = {n: [] for n in loss_names}
                last_qna = ""

                for sample in batch:
                    prompt = sample.get("prompt", "")
                    answer = sample.get("answer", "")
                    teacher_score = sample.get("score")
                    teacher_feedback = sample.get("feedback", "")
                    if not prompt or not answer or teacher_score is None:
                        continue

                    _, student_score, student_logit = self.student.generate_feedback(prompt, answer, is_math, examples)
                    with torch.no_grad():
                        live_teacher_feedback, _ = self.teacher.generate_feedback(prompt, answer, is_math, examples)

                    if "lm_loss" in loss_names:
                        batch_losses["lm_loss"].append(compute_lm_loss(live_teacher_feedback, self.student, prompt, answer, self.device, self.model_dtype, is_math, examples, teacher_score))
                    if "hidden_loss" in loss_names:
                        batch_losses["hidden_loss"].append(compute_hidden_loss(live_teacher_feedback, self.student_tokenizer, self.teacher_tokenizer, self.student, self.teacher, self.align_hidden, self.device, self.model_dtype))
                    if "scoring_loss" in loss_names:
                        batch_losses["scoring_loss"].append(compute_scoring_loss(student_logit, teacher_score, self.model_dtype, self.device))
                    if "logit_loss" in loss_names:
                        batch_losses["logit_loss"].append(compute_logit_standardization(self.student, self.teacher, self.student_tokenizer, self.teacher_tokenizer, self.projection_layer, teacher_feedback))

                    num_samples += 1
                    last_qna = prompt + answer

                aggregated = {n: torch.stack(v).mean() for n, v in batch_losses.items() if v}
                if not aggregated:
                    continue

                detached = [aggregated[n].detach() for n in loss_names if n in aggregated]
                if global_step == 0:
                    assigned_weights = weights
                    skip_kd = False
                else:
                    assigned_weights, skip_kd, freeze = self.threshold_policy.update_weights(detached, last_qna, in_KD=True)
                    if freeze:
                        self.student.freeze_student_model()

                total_loss = sum(
                    assigned_weights[i] * self.loss_config.SCALES.get(n, 1.0) * aggregated[n]
                    for i, n in enumerate(loss_names)
                    if n in aggregated
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                for n, v in aggregated.items():
                    loss_history[n].append(float(v.item()))

                global_step += 1
                if skip_kd:
                    stop_epoch = True
                    break

            if stop_epoch:
                break

        return {"student": self.student, "loss_history": loss_history}

    # ------------------------------------------------------------------
    # Inference-time combined feedback (with optional KD trigger)
    # ------------------------------------------------------------------

    def generate_combined_feedback(
        self,
        prompt: str,
        answer: str,
        is_math: bool = False,
        examples: list | None = None,
        epochs: int = 1,
        iterations: int = 20,
        threshold: float = 0.65,
        kd: bool = True,
        update_baseline: bool = True,
    ) -> dict:
        result = compute_all_losses_vectorized(
            self.loss_config, prompt, answer,
            self.student, self.teacher,
            self.student_tokenizer, self.teacher_tokenizer,
            self.align_hidden, self.projection_layer,
            self.device, is_math, examples, self.model_dtype,
        )
        student_fb, teacher_fb, student_score, teacher_score = unpack_feedback_and_scores(result)
        combined_score = (threshold * student_score + teacher_score) / (1 + threshold)

        if self.student.student_frozen:
            return {"expert_feedback": teacher_fb, "amateur_feedback": student_fb, "combined_score": combined_score, "loss_vector": result["loss_vector"]}

        is_skipping, info = self.threshold_policy.should_skip_kd(result["loss_vector"], kd, update_baseline)

        if is_skipping or not kd:
            return {"expert_feedback": teacher_fb, "amateur_feedback": student_fb, "combined_score": combined_score, "loss_vector": result["loss_vector"]}

        print("[Distill Trigger] Initiating KD...")
        self.knowledge_distillation_with_SFT(info["adaptive_weights"], epochs=epochs, stopped=iterations, is_math=is_math, examples=examples)

        updated = compute_all_losses_vectorized(
            self.loss_config, prompt, answer,
            self.student, self.teacher,
            self.student_tokenizer, self.teacher_tokenizer,
            self.align_hidden, self.projection_layer,
            self.device, is_math, examples, self.model_dtype,
        )
        student_fb, teacher_fb, student_score, teacher_score = unpack_feedback_and_scores(updated)
        combined_score = (threshold * student_score + teacher_score) / (1 + threshold)
        return {"expert_feedback": teacher_fb, "amateur_feedback": student_fb, "combined_score": combined_score, "loss_vector": updated["loss_vector"]}
