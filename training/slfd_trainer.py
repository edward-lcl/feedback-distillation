"""
SLFD training loop — distills step-level feedback generation from teacher to student.
"""
import random
import torch
from torch.optim import AdamW
from .losses import (
    compute_lm_loss,
    compute_hidden_loss,
    compute_scoring_loss,
    compute_logit_kd_loss,
)
from .threshold_policy import AdaptiveWeightedKDPolicyEMA
from .loss_config import LossConfig


class SLFDTrainer:
    """
    Trains the student to generate step-level feedback and scores.
    Teacher is frozen throughout. Student's score_head has gradients.
    """

    def __init__(
        self,
        student,
        teacher,
        dataset: list[dict],
        loss_flags=None,
        device=None,
        dev_mode=False,
        model_lr: float = 1e-4,
        score_lr: float = 5e-5,
        align_lr: float = 1e-6,
        weight_decay: float = 0.01,
        lm_weight: float = 1.0,
        score_weight: float = 1.0,
        hidden_weight: float = 1.0,
        score_loss: str = "mse",
        error_weight: float = 1.0,
        rank_margin: float = 1.0,
        balanced_batches: bool = False,
        kd_temperature: float = 2.0,
    ):
        import torch.nn as nn
        from models.device import is_dev_mode, best_device
        self.student = student
        self.teacher = teacher
        self.dataset = dataset
        self.dev_mode = is_dev_mode(dev_mode)
        if self.dev_mode:
            print("DEV MODE: reduced batch/steps for local Apple Silicon")
        # Match the student's device (MPS on Apple Silicon, CUDA, or CPU) — the
        # student model is already placed there, so inputs must follow.
        self.device = device or best_device()
        self.loss_flags = loss_flags or [True, True, True, False]  # LM, hidden, score, logit
        self.loss_config = LossConfig(self.loss_flags)
        self.loss_weights = {
            "lm_loss": lm_weight,
            "hidden_loss": hidden_weight,
            "scoring_loss": score_weight,
        }
        self.score_loss = "bce" if score_loss == "verdict" else score_loss
        self.error_weight = error_weight
        self.rank_margin = rank_margin
        self.balanced_batches = balanced_batches
        self.kd_temperature = kd_temperature
        if self.score_loss not in ("mse", "bce", "rank", "bce_rank", "soft"):
            raise ValueError(f"unknown score_loss {self.score_loss!r}")
        if self.loss_flags[3] and self.teacher is None:
            raise ValueError("logit-KD loss is enabled but no teacher was passed")
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
        print(f"Optimizer: model_lr={model_lr:g} score_lr={score_lr:g} "
              f"align_lr={align_lr:g} weight_decay={weight_decay:g}")
        print("Loss weights: " + " ".join(
            f"{name}={weight:g}" for name, weight in self.loss_weights.items()
        ))
        print(f"Score loss: {score_loss} error_weight={error_weight:g} "
              f"rank_margin={rank_margin:g} balanced_batches={balanced_batches}")
        self.optimizer = AdamW([
            {"params": trainable_model_params, "lr": model_lr, "weight_decay": weight_decay},
            {"params": self.student.score_head.parameters(), "lr": score_lr, "weight_decay": weight_decay},
            {"params": self.align_hidden.parameters(), "lr": align_lr, "weight_decay": weight_decay},
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
        # On Apple Silicon, MPS allocations live in unified GPU memory and the
        # caching allocator can hold freed blocks long enough to OOM long sweeps.
        import gc, os
        on_mps = str(self.device).startswith("mps")
        empty_every = max(1, int(os.environ.get("MPS_EMPTY_CACHE_EVERY", "5")))
        if on_mps:
            cap = int(os.environ.get("TRAIN_MAX_TOKENS", "1024"))
            tok = self.student.tokenizer

            def _ntok(s):
                txt = (f"Problem: {s.get('problem','')}\nSteps so far:\n"
                       f"{s.get('solution_prefix','')}\nCurrent step: {s.get('step_text','')}\n\n"
                       f"Score: {float(s.get('score',0.0)):.2f}\nFeedback: {s.get('feedback','')}")
                return len(tok(txt)["input_ids"])

            before = len(self.dataset)
            self.dataset = [s for s in self.dataset if _ntok(s) <= cap]
            dropped = before - len(self.dataset)
            print(f"MPS length filter: dropped {dropped}/{before} samples over {cap} tokens "
                  f"({100*dropped/max(before,1):.2f}%); {len(self.dataset)} remain.")

        def is_error_sample(sample: dict) -> bool:
            score = sample.get("score", 0.0)
            return bool(sample.get("is_error", score < 0.0))

        def iter_balanced_batches():
            errors = [s for s in self.dataset if is_error_sample(s)]
            clean = [s for s in self.dataset if not is_error_sample(s)]
            if not errors or not clean:
                raise ValueError("balanced_batches requires at least one error and one clean sample")
            n_error = max(1, batch_size // 2)
            n_clean = max(1, batch_size - n_error)
            error_i = clean_i = 0
            random.shuffle(errors)
            random.shuffle(clean)
            while True:
                batch = []
                for _ in range(n_error):
                    if error_i >= len(errors):
                        random.shuffle(errors)
                        error_i = 0
                    batch.append(errors[error_i])
                    error_i += 1
                for _ in range(n_clean):
                    if clean_i >= len(clean):
                        random.shuffle(clean)
                        clean_i = 0
                    batch.append(clean[clean_i])
                    clean_i += 1
                random.shuffle(batch)
                yield batch

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            if self.balanced_batches:
                balanced_source = iter_balanced_batches()
                n_batches = max_steps if max_steps else max(1, len(self.dataset) // batch_size)
                batch_iter = (next(balanced_source) for _ in range(n_batches))
            else:
                batch_iter = (
                    self.dataset[i: i + batch_size]
                    for i in range(0, len(self.dataset), batch_size)
                )

            for batch in batch_iter:
                if max_steps and step >= max_steps:
                    break
                batch_losses = {"lm_loss": [], "hidden_loss": [], "scoring_loss": [], "logit_loss": []}
                score_logits = []
                error_targets = []

                for sample in batch:
                    problem = sample.get("problem", "")
                    step_text = sample.get("step_text", "")
                    solution_prefix = sample.get("solution_prefix", "")
                    teacher_score = sample.get("score", 0.0)
                    teacher_feedback = sample.get("feedback", "")
                    is_error = bool(sample.get("is_error", teacher_score < 0.0))
                    if not problem or not step_text:
                        continue

                    # Student forward — score_logit has gradient (boundary-token
                    # read, no generation).
                    student_score_logit = self.student.score_step(problem, solution_prefix, step_text)
                    score_logits.append(student_score_logit.to(torch.float32).view(-1)[0])
                    error_targets.append(is_error)

                    # L_feedback_LM — natural-language critique loss. Gated by the
                    # LM flag so the score-only ablation trains the scorer alone
                    # (score+critique = LM on; score-only = LM off).
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

                    # L_score
                    if self.score_loss in ("bce", "bce_rank"):
                        # Treat score_head output as a correctness logit:
                        # positive => correct, negative => error. Upweight
                        # error steps to counter the mostly-correct label mix.
                        target = torch.tensor(
                            [0.0 if is_error else 1.0],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        weight = torch.tensor(
                            [self.error_weight if is_error else 1.0],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        batch_losses["scoring_loss"].append(
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                student_score_logit.to(torch.float32),
                                target,
                                weight=weight,
                            )
                        )
                    elif self.score_loss == "mse":
                        target = torch.tensor([teacher_score], dtype=torch.float32, device=self.device)
                        batch_losses["scoring_loss"].append(
                            torch.nn.functional.mse_loss(student_score_logit.to(torch.float32), target)
                        )
                    elif self.score_loss == "soft":
                        p = (float(teacher_score) + 1.0) / 2.0
                        target = torch.tensor([p], dtype=torch.float32, device=self.device)
                        batch_losses["scoring_loss"].append(
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                student_score_logit.to(torch.float32),
                                target,
                            )
                        )

                    # L_logit — online KD toward a local same-family teacher's
                    # critique-token distribution. Used only when loss_flags[3].
                    if self.loss_flags[3] and self.teacher is not None:
                        kd = compute_logit_kd_loss(
                            self.student,
                            self.teacher,
                            problem,
                            solution_prefix,
                            step_text,
                            teacher_feedback,
                            teacher_score,
                            temperature=self.kd_temperature,
                        )
                        batch_losses["logit_loss"].append(kd * LossConfig.SCALES["logit_loss"])

                aggregated = {k: torch.stack(v).mean() for k, v in batch_losses.items() if v}
                if self.score_loss in ("rank", "bce_rank") and score_logits:
                    logits = torch.stack(score_logits)
                    errors = torch.tensor(error_targets, dtype=torch.bool, device=self.device)
                    error_logits = logits[errors]
                    clean_logits = logits[~errors]
                    if error_logits.numel() and clean_logits.numel():
                        # score_head is a correctness logit, so clean steps
                        # should score above error steps by rank_margin.
                        rank_terms = torch.nn.functional.softplus(
                            error_logits[:, None] - clean_logits[None, :] + self.rank_margin
                        )
                        rank_loss = rank_terms.mean()
                        if "scoring_loss" in aggregated:
                            aggregated["scoring_loss"] = aggregated["scoring_loss"] + rank_loss
                        else:
                            aggregated["scoring_loss"] = rank_loss
                if not aggregated:
                    continue

                total_loss = sum(
                    self.loss_weights.get(name, 1.0) * value
                    for name, value in aggregated.items()
                )
                # Skip non-finite steps so a single NaN/Inf can't corrupt weights
                # (fp16 on MPS overflows easily). The checkpoint stays clean.
                if not torch.isfinite(total_loss):
                    print(f"  step {step}: non-finite loss ({total_loss.item()}), skipping")
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.student.score_head.parameters(), max_norm=1.0)
                self.optimizer.step()

                for k, v in aggregated.items():
                    loss_history[k].append(float(v.item()))

                step += 1
                if on_mps and step % empty_every == 0:
                    gc.collect()
                    torch.mps.empty_cache()
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
            "score_head": self.student.score_head.state_dict(),
            "align_hidden": self.align_hidden.state_dict(),
        }, path)
        print(f"Saved checkpoint → {path}")
