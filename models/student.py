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
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            self.model.print_trainable_parameters()

        self.score_head = nn.Linear(hidden_dim, 1).to(self.device).to(torch.float32)
        self.student_frozen = False
        print(f"StudentModel loaded: {model_name} on {self.device} (LoRA={'on' if use_lora else 'off'})")

    def _backbone_model(self):
        """Return the decoder backbone so score probes avoid LM logits/all layers."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        # PeftModelForCausalLM -> LoraModel -> wrapped AutoModelForCausalLM -> backbone.
        base_model = getattr(model, "base_model", None)
        wrapped = getattr(base_model, "model", None) if base_model is not None else None
        if wrapped is not None and hasattr(wrapped, "model"):
            return wrapped.model
        # AutoModelForCausalLM -> backbone.
        if hasattr(model, "model"):
            return model.model
        return None

    def _last_hidden_state(self, inputs: dict) -> torch.Tensor:
        backbone = self._backbone_model()
        if backbone is not None:
            out = backbone(**inputs, return_dict=True)
            return out.last_hidden_state
        out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

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

    def score_step(self, problem: str, solution_prefix: str, step_text: str) -> torch.Tensor:
        """Training-time step score: a single forward over the prompt, read the
        hidden state at the step-boundary token (the last token, after "Score: "),
        apply the score head. Returns score_logit with grad. No generation — the
        old path generated ~150 tokens per training step, which was slow and made
        L_score depend on sampled text rather than the boundary representation.
        """
        prompt = self.STEP_EVAL_PROMPT.format(
            problem=problem, solution_prefix=solution_prefix, step_text=step_text,
        )
        ml = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] > ml:
            inputs = {k: v[:, -ml:] for k, v in inputs.items()}
        hidden = self._last_hidden_state(inputs)
        last_hidden = hidden[:, -1, :].to(self.device, dtype=torch.float32)
        return self.score_head(last_hidden).squeeze(-1)

    def _iter_step_last_hidden(
        self,
        step_inputs: list[tuple[str, str, str]],
        batch_size: int = 16,
    ):
        """Yield batched step-boundary hidden states for score/probe heads."""
        ml = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        for start in range(0, len(step_inputs), batch_size):
            chunk = step_inputs[start:start + batch_size]
            encoded = []
            for problem, solution_prefix, step_text in chunk:
                prompt = self.STEP_EVAL_PROMPT.format(
                    problem=problem,
                    solution_prefix=solution_prefix,
                    step_text=step_text,
                )
                ids = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )["input_ids"][0]
                if ids.numel() > ml:
                    ids = ids[-ml:]
                encoded.append(ids)

            max_len = max(int(ids.numel()) for ids in encoded)
            input_ids = torch.full(
                (len(encoded), max_len),
                pad_id,
                dtype=encoded[0].dtype,
            )
            attention_mask = torch.zeros_like(input_ids)
            last_indices = []
            for row, ids in enumerate(encoded):
                input_ids[row, :ids.numel()] = ids
                attention_mask[row, :ids.numel()] = 1
                last_indices.append(int(ids.numel()) - 1)

            inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            inputs["position_ids"] = position_ids.to(self.device)
            hidden = self._last_hidden_state(inputs)
            row_idx = torch.arange(len(encoded), device=hidden.device)
            col_idx = torch.tensor(last_indices, device=hidden.device)
            last_hidden = hidden[row_idx, col_idx, :].to(self.device, dtype=torch.float32)
            yield last_hidden

    def score_steps(
        self,
        step_inputs: list[tuple[str, str, str]],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Vectorized score_step for exploratory evaluation/diagnostics.

        Each item is (problem, solution_prefix, step_text). This preserves the
        prompt and tail-truncation convention, but padded batched bf16 inference
        can slightly perturb rankings versus the exact serial score_step path.
        Use batch_size=1 for headline metrics.
        """
        if not step_inputs:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        logits = []
        for last_hidden in self._iter_step_last_hidden(step_inputs, batch_size):
            logits.append(self.score_head(last_hidden).squeeze(-1))

        return torch.cat(logits, dim=0)

    def step_representations(
        self,
        step_inputs: list[tuple[str, str, str]],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Return CPU step-boundary hidden states for cheap downstream probes."""
        if not step_inputs:
            hidden_dim = self.model.config.hidden_size
            return torch.empty((0, hidden_dim), dtype=torch.float32)
        with torch.no_grad():
            reps = [
                last_hidden.detach().cpu()
                for last_hidden in self._iter_step_last_hidden(step_inputs, batch_size)
            ]
        return torch.cat(reps, dim=0)

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
        sh_max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        inputs = self.tokenizer(response, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] > sh_max_length:
            inputs = {k: v[:, -sh_max_length:] for k, v in inputs.items()}
        hidden = self._last_hidden_state(inputs)
        last_hidden = hidden[:, -1, :].to(self.device, dtype=torch.float32)
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
        ml = MPS_SAFE_MAX_LENGTH if is_mps() else 512

        # Only supervise on feedback token span. Tokenize the prefix and
        # feedback separately so long reasoning prefixes cannot right-truncate
        # away every supervised token and produce an all-ignore NaN LM loss.
        prefix_text = (
            f"Problem: {problem}\n"
            f"Steps so far:\n{solution_prefix}\n"
            f"Current step: {step_text}\n\n"
            f"Score: {teacher_score:.2f}\nFeedback: "
        )
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        feedback_ids = self.tokenizer(
            teacher_feedback or " ",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

        if feedback_ids.numel() == 0:
            feedback_ids = torch.tensor(
                [self.tokenizer.eos_token_id],
                dtype=prefix_ids.dtype,
            )

        if feedback_ids.numel() >= ml:
            feedback_ids = feedback_ids[: ml - 1]

        keep_prefix = max(1, ml - int(feedback_ids.numel()))
        if prefix_ids.numel() > keep_prefix:
            prefix_ids = prefix_ids[-keep_prefix:]

        input_ids = torch.cat([prefix_ids, feedback_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:, : prefix_ids.numel()] = -100
        return input_ids, labels, attention_mask

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.student_frozen = True
        print("[StudentModel] Frozen.")
