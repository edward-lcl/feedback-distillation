"""
StudentModel — lightweight model (Qwen2.5-1.5B) with LoRA + step scoring head.
Trained to generate step-level feedback and predict step correctness scores.
Operates GT-free at test time.
"""
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from .device import (
    best_device,
    best_dtype,
    is_mps,
    is_dev_mode,
    load_model_for_device,
    DEV_MODELS,
    PROD_MODELS,
    MPS_SAFE_MAX_LENGTH,
)


def _parse_score(response: str) -> float:
    m = re.search(r'Score:\s*([-\d.]+)', response)
    try:
        return float(m.group(1)) if m else -1.0
    except (ValueError, AttributeError):
        return -1.0


class StudentModel:
    """
    Student model for SLFD. Learns to:
    1. Predict a step-level correctness score (score head on step-boundary token)
    2. Generate natural-language feedback explaining why a step is wrong
    3. Detect which step is the first error (is_error classification)
    """

    STEP_EVAL_PROMPT = (
        "Evaluate this reasoning step. Score -1.0 (wrong) to 1.0 (correct).\n"
        "If wrong, explain precisely what the error is (1-2 sentences).\n\n"
        "Problem: {problem}\n"
        "Steps so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n\n"
        "Score: "
    )

    def __init__(self, model_name: str = None, use_bf16: bool = True, dev_mode: bool = False):
        self.device = best_device()
        self.dev_mode = is_dev_mode(dev_mode)
        if model_name is None:
            model_name = (DEV_MODELS if self.dev_mode else PROD_MODELS)["student"]
        if self.dev_mode:
            print("# DEV MODE — reduced models for local Apple Silicon development")
        self.model_dtype = best_dtype() if use_bf16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = load_model_for_device(model_name, dev_mode=self.dev_mode)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_dim = self.model.config.hidden_size
        self.score_head = nn.Linear(hidden_dim, 1).to(self.device).to(torch.float32)
        self.student_frozen = False
        print(f"StudentModel loaded: {model_name} on {self.device}")

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 512
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def evaluate_step(
        self,
        problem: str,
        solution_prefix: str,
        step_text: str,
    ) -> tuple[str, float, torch.Tensor]:
        """
        Returns (feedback_text, scalar_score, score_logit_tensor).
        score_logit has grad — used in L_score during training.
        """
        prompt = self.STEP_EVAL_PROMPT.format(
            problem=problem,
            solution_prefix=solution_prefix,
            step_text=step_text,
        )
        response = self._generate(prompt)
        score_val = _parse_score(response)

        parts = response.split("Feedback:", 1)
        feedback = parts[1].strip().split("\n")[0].strip() if len(parts) > 1 else ""

        # Score head forward — OUTSIDE no_grad so gradients flow.
        # Cast hidden states (float16 on MPS) to float32 before the float32 score head.
        sh_max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 256
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=sh_max_length).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1][:, -1, :].to(torch.float32)
        score_logit = self.score_head(last_hidden).squeeze(-1)  # grad flows here

        return feedback, score_val, score_logit

    def prepare_step_inputs_and_labels(
        self,
        problem: str,
        solution_prefix: str,
        step_text: str,
        teacher_feedback: str,
        teacher_score: float,   # actual score, not hardcoded
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (input_ids, labels, attention_mask) for L_feedback_LM loss."""
        full_text = (
            f"Problem: {problem}\n"
            f"Steps so far:\n{solution_prefix}\n"
            f"Current step: {step_text}\n\n"
            f"Score: {teacher_score:.2f}\nFeedback: {teacher_feedback}"
        )
        ml = MPS_SAFE_MAX_LENGTH if is_mps() else 512
        enc = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=ml)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Only supervise on feedback token span
        prefix_text = (
            f"Problem: {problem}\n"
            f"Steps so far:\n{solution_prefix}\n"
            f"Current step: {step_text}\n\n"
            f"Score: {teacher_score:.2f}\nFeedback: "
        )
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt")["input_ids"]
        prefix_len = prefix_ids.shape[1]

        labels = input_ids.clone()
        labels[:, :prefix_len] = -100
        return input_ids, labels, attention_mask

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.student_frozen = True
        print("[StudentModel] Frozen.")
