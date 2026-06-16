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


def _parse_score(response: str) -> float | None:
    """Parse a step score from the teacher's reply.

    Accepts "Score: X" anywhere, or a bare leading number (the prompt ends in
    "Score: ", so models often reply with just the number). Returns a float
    clamped to [-1, 1], or None if unparseable. None means "drop this label" —
    the old behavior of defaulting to -1.0 silently injected false error labels
    whenever the teacher deviated from the format.
    """
    m = re.search(r'Score:\s*(-?\d+(?:\.\d+)?)', response)
    if not m:
        m = re.match(r'\s*(-?\d+(?:\.\d+)?)', response)
    if not m:
        return None
    try:
        return max(-1.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


class TeacherModel:
    """
    Frozen expert model. Generates per-step (score, feedback, is_error) labels.
    Has access to ground-truth answers during labeling — this is the privileged
    signal that the student never sees at test time.
    """

    # Chat-templated models answer fresh — they do NOT continue a trailing
    # "Score: " cue. So the format contract must be explicit, and the local
    # path additionally prefills the assistant turn with "Score: ".
    STEP_EVAL_PROMPT = (
        "You are evaluating a single reasoning step in a math solution.\n"
        "The correct final answer is provided — use it to judge correctness.\n\n"
        "Problem: {problem}\n"
        "Solution so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n"
        "Correct answer: {gt_answer}\n\n"
        "Score the current step from -1.0 (definitely wrong) to 1.0 (definitely correct).\n"
        "Reply in EXACTLY this format and nothing else:\n"
        "Score: <number>\n"
        "Feedback: <1-2 sentences; if the step is correct, write 'Correct.'>"
    )

    # Richer-privilege variant: the teacher sees the full worked reference
    # solution, not just the final number. A bare answer is weak privilege on
    # easy math (the teacher can self-verify); a worked solution is strong
    # privilege. Used to test whether the ~0 privileged gap on GSM8K was an
    # artifact of thin privilege rather than privilege being useless.
    STEP_EVAL_PROMPT_SOLUTION = (
        "You are evaluating a single reasoning step in a math solution.\n"
        "A correct reference solution is provided — use it to judge correctness.\n\n"
        "Problem: {problem}\n"
        "Solution so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n"
        "Reference solution:\n{gt_solution}\n\n"
        "Score the current step from -1.0 (definitely wrong) to 1.0 (definitely correct).\n"
        "Reply in EXACTLY this format and nothing else:\n"
        "Score: <number>\n"
        "Feedback: <1-2 sentences; if the step is correct, write 'Correct.'>"
    )

    # GT-free variant — used to measure the privileged gap (how much the GT
    # answer actually helps the teacher). Identical except no answer line.
    STEP_EVAL_PROMPT_NO_GT = (
        "You are evaluating a single reasoning step in a math solution.\n\n"
        "Problem: {problem}\n"
        "Solution so far:\n{solution_prefix}\n"
        "Current step: {step_text}\n\n"
        "Score the current step from -1.0 (definitely wrong) to 1.0 (definitely correct).\n"
        "Reply in EXACTLY this format and nothing else:\n"
        "Score: <number>\n"
        "Feedback: <1-2 sentences; if the step is correct, write 'Correct.'>"
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

    def _generate(self, prompt: str, max_new_tokens: int = 150, prefill: str = "") -> str:
        max_length = MPS_SAFE_MAX_LENGTH if is_mps() else 1024
        # Use chat template if available — improves format adherence on smaller models
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        # Assistant prefill: start the reply for the model ("Score: ") so the
        # continuation is the number — small models otherwise drift to prose.
        formatted += prefill
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        # On overflow keep the TAIL — the current step and "Score:" cue live at
        # the end; right-truncation made the model continue the solution text.
        if inputs["input_ids"].shape[1] > max_length:
            inputs = {k: v[:, -max_length:] for k, v in inputs.items()}
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
        gt_answer: str = None,
    ) -> dict:
        """
        Returns: {score: float|None, feedback: str, is_error: bool|None, parse_failed: bool}
        is_error = True means this step is likely wrong. gt_answer=None uses the
        GT-free prompt (for measuring the privileged gap). score=None means the
        reply was unparseable — drop the label downstream, never train on it.
        """
        if gt_answer is not None:
            prompt = self.STEP_EVAL_PROMPT.format(
                problem=problem,
                solution_prefix=solution_prefix,
                step_text=step_text,
                gt_answer=gt_answer,
            )
        else:
            prompt = self.STEP_EVAL_PROMPT_NO_GT.format(
                problem=problem,
                solution_prefix=solution_prefix,
                step_text=step_text,
            )
        # Prefill "Score: " — the reply comes back as just the number onward
        # (e.g. "0.8\nFeedback: ..."); _parse_score accepts the bare number.
        response = self._generate(prompt, prefill="Score: ")
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
            "is_error": (score < 0.0) if score is not None else None,
            "parse_failed": score is None,
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
