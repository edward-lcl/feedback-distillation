import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .device import best_device, best_dtype


def _first_paragraph(text: str) -> str:
    for part in text.strip().split("\n\n"):
        line = part.strip().split("\n")[0].strip()
        if line:
            return line
    return text.strip()


class AmateurFeedbackModel:
    """
    Lightweight student model. Generates feedback via NLP (same prompt format as
    expert) and learns a scoring head that is aligned to the teacher during KD.
    """

    SCORE_SCALE = (
        "  •  1.0 — Excellent\n"
        "  •  0.7–0.9 — Good\n"
        "  •  0.4–0.6 — Fair\n"
        "  •  0.1–0.3 — Weak\n"
        "  •  0.0 — Not helpful\n"
        "  • -0.1 to -0.9 — Misleading or incorrect\n"
        "  • -1.0 — Harmful\n"
    )

    def __init__(self, tokenizer, model, model_name: str, use_bf16: bool = True):
        self.device = best_device()
        self.tokenizer = tokenizer
        self.model = model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_dim = self.model.config.hidden_size
        self.model_dtype = best_dtype() if use_bf16 else torch.float32
        self.score_head = nn.Linear(hidden_dim, 1, dtype=self.model_dtype).to(self.device)
        self.student_frozen = False
        print(f"AmateurFeedbackModel: {model_name} on {self.device} ({self.model_dtype})")

    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_feedback(
        self,
        question: str,
        answer: str,
        is_math: bool = False,
        examples: list | None = None,
    ) -> tuple[str, float, torch.Tensor]:
        """Returns (feedback_text, scalar_score, score_logit_tensor)."""
        if is_math and examples:
            ex = examples[0]
            one_shot = (
                f"Question: {ex['question']}\nAnswer: {ex['answer']}\n"
                f"Score: {ex['score']}\nFeedback: {ex.get('amateur_feedback', ex.get('expert_feedback', ''))}\n"
            )
        else:
            one_shot = (
                "Question: Explain photosynthesis.\n"
                "Answer: Plants convert sunlight to energy using chlorophyll...\n"
                "Score: 0.6\nFeedback: Covers the basics but lacks detail on the light reactions.\n"
            )

        prompt = (
            "You are an amateur evaluator learning to assess answers.\n"
            f"Scoring scale:\n{self.SCORE_SCALE}\n"
            "Format:\nScore: <number>\nFeedback: <one-paragraph>\n\n"
            f"{one_shot}"
            f"Question: {question}\nAnswer: {answer}\nScore:"
        )

        response = self.generate_text(prompt, max_new_tokens=200).strip()

        score_parts = response.split("Score:")
        try:
            score_val = float(score_parts[3].strip().split()[0]) if len(score_parts) >= 4 else -1000.0
        except (IndexError, ValueError):
            score_val = -1000.0

        feedback_parts = [f for f in response.split("Feedback:") if "<" not in f and ">" not in f]
        feedback = _first_paragraph(feedback_parts[1].strip()) if len(feedback_parts) >= 2 else "Failed to generate meaningful feedback"

        # Score logit via scoring head on last hidden state
        inputs = self.tokenizer(response, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1][:, -1, :].to(self.model_dtype)
            score_logit = self.score_head(last_hidden).squeeze(-1)

        return feedback, score_val, score_logit

    # ------------------------------------------------------------------

    def prepare_inputs_and_labels(
        self,
        prompt: str,
        answer: str,
        teacher_feedback: str,
        is_math: bool = False,
        examples: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (input_ids, labels, attention_mask) for language-model loss."""
        if is_math and examples:
            ex = examples[0]
            prefix = f"Question: {ex['question']}\nAnswer: {ex['answer']}\nFeedback: {ex.get('expert_feedback', '')}\n\n"
        else:
            prefix = ""

        full_text = (
            f"{prefix}Question: {prompt}\nAnswer: {answer}\n"
            f"Score: 0.9\nFeedback: {teacher_feedback}"
        )
        enc = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Only supervise on the feedback token span
        feedback_prefix = f"{prefix}Question: {prompt}\nAnswer: {answer}\nScore: 0.9\nFeedback: "
        prefix_ids = self.tokenizer(feedback_prefix, return_tensors="pt")["input_ids"]
        prefix_len = prefix_ids.shape[1]

        labels = input_ids.clone()
        labels[:, :prefix_len] = -100
        return input_ids, labels, attention_mask

    def freeze_student_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.student_frozen = True
        print("[Info] Student model frozen.")
