"""
StudentModel — lightweight model (Qwen2.5-1.5B) with LoRA + step scoring head.
Trained to generate step-level feedback and predict step correctness scores.
Operates GT-free at test time.
"""
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
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





class StudentModel:
    """
    Student model for SLFD. Learns to:
    1. Predict a step-level correctness score (score head on step-boundary token)
    2. Generate natural-language feedback explaining why a step is wrong
    3. Detect which step is the first error (is_error classification)
    """

    STEP_EVAL_PROMPT = (
        "Evaluate this reasoning step. If wrong, explain precisely what the error is.\n"
        "End your explanation with 'Verdict: Correct' or 'Verdict: Incorrect'.\n\n"
        "Problem: {problem}\n"
        "Steps so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n\n"
        "Feedback:"
    )

    def __init__(self, model_name: str = None, use_bf16: bool = True, dev_mode: bool = False,
                 use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32):
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

        # Parameter-efficient fine-tuning: freeze the base model, train only LoRA
        # adapters (+ the score head). This is what the README/architecture claim
        # and what makes training cheap; the old path full-fine-tuned the base.
        self.use_lora = use_lora
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules="all-linear", task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        self.student_frozen = False
        print(f"StudentModel loaded: {model_name} on {self.device} (LoRA={'on' if use_lora else 'off'})")

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Keep the TAIL on overflow — the step + "Score:" cue are at the end.
        if inputs["input_ids"].shape[1] > max_length:
            inputs = {k: v[:, -max_length:] for k, v in inputs.items()}
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
        Returns (feedback_text, scalar_score, None).
        scalar_score is derived from P(Correct) / (P(Correct) + P(Incorrect)).
        """
        prompt = self.STEP_EVAL_PROMPT.format(
            problem=problem,
            solution_prefix=solution_prefix,
            step_text=step_text,
        )
        response = self._generate(prompt)

        feedback = response.strip()

        # To get the continuous probability, we need the logits right BEFORE the model
        # generated 'Correct' or 'Incorrect'.
        prefix_response = response
        if prefix_response.endswith("Correct"):
            prefix_response = prefix_response[:-7]
        elif prefix_response.endswith("Incorrect"):
            prefix_response = prefix_response[:-9]
            
        inputs = self.tokenizer(prompt + prefix_response, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            # The logits of the very last token in the prefix (predicting the next word)
            logits = out.logits[0, -1, :]

        # " Correct" vs " Incorrect" token extraction
        correct_id = self.tokenizer.encode(" Correct", add_special_tokens=False)[-1]
        incorrect_id = self.tokenizer.encode(" Incorrect", add_special_tokens=False)[-1]

        pc = logits[correct_id]
        pi = logits[incorrect_id]
        score_val = float(torch.softmax(torch.tensor([pi, pc], dtype=torch.float32), dim=0)[1].item())

        return feedback, score_val, None

    def prepare_step_inputs_and_labels(
        self,
        problem: str,
        solution_prefix: str,
        step_text: str,
        teacher_feedback: str,
        teacher_score: float,   # actual score
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (input_ids, labels, attention_mask) for Generative PRM training."""
        verdict = "Correct" if teacher_score > 0 else "Incorrect"
        full_text = (
            f"Evaluate this reasoning step. If wrong, explain precisely what the error is.\n"
            f"End your explanation with 'Verdict: Correct' or 'Verdict: Incorrect'.\n\n"
            f"Problem: {problem}\n"
            f"Steps so far:\n{solution_prefix}\n"
            f"Current step: {step_text}\n\n"
            f"Feedback: {teacher_feedback}\nVerdict: {verdict}"
        )
        ml = MPS_SAFE_MAX_LENGTH if is_mps() else 512
        enc = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=ml)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Only supervise on feedback + verdict span
        prefix_text = (
            f"Evaluate this reasoning step. If wrong, explain precisely what the error is.\n"
            f"End your explanation with 'Verdict: Correct' or 'Verdict: Incorrect'.\n\n"
            f"Problem: {problem}\n"
            f"Steps so far:\n{solution_prefix}\n"
            f"Current step: {step_text}\n\n"
            f"Feedback:"
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
