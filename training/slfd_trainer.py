"""
SLFD training loop — distills step-level feedback generation from teacher to student.
"""
import random
import torch
from torch.optim import AdamW
from .losses import compute_lm_loss, compute_hidden_loss, compute_scoring_loss
from .threshold_policy import AdaptiveWeightedKDPolicyEMA
from .loss_config import LossConfig


class SLFDTrainer:
    """
    Trains the student to generate step-level feedback and scores.
    Teacher is frozen throughout. Student's score_head has gradients.
    """

    def __init__(self, student, teacher, dataset: list[dict], loss_flags=None, device=None):
        import torch.nn as nn
        self.student = student
        self.teacher = teacher
        self.dataset = dataset
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_flags = loss_flags or [True, True, True, False]  # LM, hidden, score, logit
        self.loss_config = LossConfig(self.loss_flags)
        self.threshold_policy = AdaptiveWeightedKDPolicyEMA(
            num_losses=self.loss_config.num_enabled,
            device=str(self.device),
        )

        # Hidden alignment projection
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

        self.optimizer = AdamW([
            {"params": self.student.model.parameters(), "lr": 5e-7, "weight_decay": 0.01},
            {"params": self.student.score_head.parameters(), "lr": 5e-5, "weight_decay": 0.01},
            {"params": self.align_hidden.parameters(), "lr": 1e-6, "weight_decay": 0.01},
        ], betas=(0.9, 0.999), eps=1e-8)

    def train(self, epochs: int = 2, batch_size: int = 4, max_steps: int = None) -> dict:
        random.shuffle(self.dataset)
        loss_history = {"lm_loss": [], "hidden_loss": [], "scoring_loss": []}
        step = 0

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            for batch_start in range(0, len(self.dataset), batch_size):
                if max_steps and step >= max_steps:
                    break
                batch = self.dataset[batch_start: batch_start + batch_size]
                batch_losses = {"lm_loss": [], "hidden_loss": [], "scoring_loss": []}

                for sample in batch:
                    problem = sample.get("problem", "")
                    step_text = sample.get("step_text", "")
                    solution_prefix = sample.get("solution_prefix", "")
                    teacher_score = sample.get("score", 0.0)
                    teacher_feedback = sample.get("feedback", "")
                    if not problem or not step_text:
                        continue

                    # Student forward — score_logit has gradient
                    _, _, student_score_logit = self.student.evaluate_step(problem, solution_prefix, step_text)

                    # L_feedback_LM
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
                    target = torch.tensor([teacher_score], dtype=torch.float32, device=self.device)
                    batch_losses["scoring_loss"].append(
                        torch.nn.functional.mse_loss(student_score_logit.to(torch.float32), target)
                    )

                aggregated = {k: torch.stack(v).mean() for k, v in batch_losses.items() if v}
                if not aggregated:
                    continue

                total_loss = sum(aggregated.values())
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                for k, v in aggregated.items():
                    loss_history[k].append(float(v.item()))

                step += 1
                if step % 10 == 0:
                    parts = " | ".join(f"{k}={v[-1]:.4f}" for k, v in loss_history.items() if v)
                    print(f"  step {step}: {parts}")

        return {"loss_history": loss_history, "steps": step}
