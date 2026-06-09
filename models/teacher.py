"""
TeacherModel — frozen Qwen2.5-7B used to generate step-level labels.
Has GT access during labeling (privileged signal). Never updated.
"""
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .device import best_device


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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.device = best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"TeacherModel loaded: {model_name} (frozen)")

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

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
        parts = response.split("Feedback:", 1)
        feedback = parts[1].strip().split("\n")[0].strip() if len(parts) > 1 else ""
        if not feedback:
            # Try to extract anything after the score line
            lines = response.strip().split("\n")
            feedback = next((l.strip() for l in lines if l.strip() and "Score:" not in l and len(l) > 10), "")

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
