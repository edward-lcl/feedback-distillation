"""
TeacherModel — frozen Qwen2.5-7B used to generate step-level labels.
Has GT access during labeling (privileged signal). Never updated.
"""
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .device import (
    best_device,
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


class TeacherModel:
    """
    Frozen expert model. Generates per-step (score, feedback, is_error) labels.
    Has access to ground-truth answers during labeling — this is the privileged
    signal that the student never sees at test time.
    """

    STEP_EVAL_PROMPT = (
        "You are evaluating a single reasoning step in a math solution.\n"
        "The correct final answer is provided — use it to judge correctness.\n\n"
        "Problem: {problem}\n"
        "Full solution so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n"
        "Correct answer: {gt_answer}\n\n"
        "Is this step correct? Score -1.0 (wrong) to 1.0 (correct).\n"
        "Explain precisely what is wrong if incorrect (1-2 sentences).\n\n"
        "Score: "
    )

    def __init__(self, model_name: str = None, dev_mode: bool = False):
        self.device = best_device()
        self.dev_mode = is_dev_mode(dev_mode)
        if model_name is None:
            model_name = (DEV_MODELS if self.dev_mode else PROD_MODELS)["teacher"]
        if self.dev_mode:
            print("# DEV MODE — reduced models for local Apple Silicon development")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = load_model_for_device(model_name, dev_mode=self.dev_mode)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"TeacherModel loaded: {model_name} on {self.device} (frozen)")

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        # Use chat template if available — improves format adherence on smaller models
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens — never echo the prompt
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    def label_step(
        self,
        problem: str,
        solution_prefix: str,
        step_text: str,
        gt_answer: str,
    ) -> dict:
        """
        Returns: {score: float, feedback: str, is_error: bool}
        is_error = True means this step is likely wrong.
        """
        prompt = self.STEP_EVAL_PROMPT.format(
            problem=problem,
            solution_prefix=solution_prefix,
            step_text=step_text,
            gt_answer=gt_answer,
        )
        response = self._generate(prompt)
        score = _parse_score(response)

        # Extract feedback text after "Score: X.XX\n"
        # Extract feedback — only from generated text (prompt echo already stripped)
        parts = response.split("Feedback:", 1)
        if len(parts) > 1:
            feedback = parts[1].strip().split("\n")[0].strip()
        else:
            # Fallback: first substantive line after the score line
            lines = [l.strip() for l in response.strip().split("\n")]
            score_idx = next((i for i, l in enumerate(lines) if l.startswith("Score:")), -1)
            candidates = lines[score_idx + 1:] if score_idx >= 0 else lines
            feedback = next((l for l in candidates if l and len(l) > 10), "")

        return {
            "score": score,
            "feedback": feedback,
            "is_error": score < 0.0,
        }

    def label_solution(
        self,
        problem: str,
        steps: list[str],
        gt_answer: str,
    ) -> list[dict]:
        """Label every step in a solution. Returns list of {score, feedback, is_error}."""
        labels = []
        prefix = ""
        for step in steps:
            label = self.label_step(problem, prefix, step, gt_answer)
            labels.append(label)
            prefix += step + "\n"
        return labels
