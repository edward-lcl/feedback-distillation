import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .device import best_device


def _first_paragraph(text: str) -> str:
    for part in text.strip().split("\n\n"):
        line = part.strip().split("\n")[0].strip()
        if line:
            return line
    return text.strip()


def _parse_score(response: str) -> float:
    m = re.search(r'Score:\s*([-\d.]+)', response)
    try:
        return float(m.group(1)) if m else -1.0
    except (ValueError, AttributeError):
        return -1.0


class ExpertFeedbackModel:
    """
    Serves as both the base model (generates initial answers) and the expert
    feedback model (scores + critiques those answers). The same LLM is reused
    for both roles, eliminating a separate inference pass.
    """

    SCORE_SCALE = (
        "  •  1.0 — Excellent\n"
        "  •  0.7–0.9 — Good\n"
        "  •  0.4–0.6 — Fair\n"
        "  •  0.1–0.3 — Weak\n"
        "  •  0.0 — Not helpful\n"
        "  • -0.1 to -0.4 — Somewhat misleading\n"
        "  • -0.5 to -0.9 — Mostly incorrect\n"
        "  • -1.0 — Harmful or dangerously misleading\n"
    )

    def __init__(self, tokenizer, model, model_name: str):
        self.device = best_device()
        self.model = model
        self.tokenizer = tokenizer
        self.network = None  # set externally after AmateurExpertFeedbackNetWork is built
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"ExpertFeedbackModel: {model_name} on {self.device}")

    # ------------------------------------------------------------------
    # Low-level generation
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        is_math: bool = False,
        task_description: str | None = None,
        examples: list | None = None,
        max_new_tokens: int = 500,
        max_words: int = 300,
    ) -> str:
        question = question.strip()

        one_shot_text = ""
        if is_math and examples:
            ex = examples[0]
            one_shot_text = f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"

        if is_math:
            task_text = f"Task Description: {task_description}\n" if task_description else ""
            prompt = (
                "You are a specialized math language model tasked with generating accurate, "
                "well-reasoned, step-by-step solutions to specific math problems.\n"
                "Identify the concept, solve step by step, and present the solution concisely.\n"
                "Write your answer in a single paragraph.\n"
                f"Limit your response to approximately {max_words} words.\n\n"
                "Format: Answer: <one-paragraph response>\n\n"
                f"{one_shot_text}{task_text}Question: {question}\nAnswer:"
            )
        else:
            task_text = f"Task Description: {task_description}\n" if task_description else ""
            prompt = (
                "You are an expert language model. Provide a precise, well-reasoned, "
                "single-paragraph response.\n"
                f"Limit to ~{max_words} words. No lists or bullet points.\n\n"
                f"{task_text}Question: {question}\n\nAnswer:"
            )

        response = self.generate_text(prompt, max_new_tokens)
        blocks = response.split("Answer:")
        if is_math:
            answer = blocks[2].strip() if len(blocks) >= 3 else ""
        else:
            answer = blocks[1].strip() if len(blocks) >= 2 else ""
        return _first_paragraph(answer)

    # ------------------------------------------------------------------
    # Feedback generation
    # ------------------------------------------------------------------

    def generate_feedback(
        self,
        question: str,
        answer: str,
        is_math: bool = False,
        examples: list | None = None,
    ) -> tuple[str, float]:
        if is_math and examples:
            ex = examples[0]
            one_shot = (
                f"Question: {ex['question']}\nAnswer: {ex['answer']}\n"
                f"Score: {ex['score']}\nFeedback: {ex['expert_feedback']}\n"
            )
        else:
            one_shot = (
                "Question: How does the author use symbolism?\n"
                "Answer: The broken mirror represents the protagonist's fractured inner state...\n"
                "Score: 0.85\nFeedback: Perceptive interpretation; needs textual evidence.\n"
            )

        prompt = (
            "You are an expert feedback evaluator.\n"
            f"Scoring scale:\n{self.SCORE_SCALE}\n"
            "Format:\nScore: <number>\nFeedback: <one-paragraph>\n\n"
            f"{one_shot}"
            f"Question: {question}\nAnswer: {answer}\nScore:"
        )

        response = self.generate_text(prompt, max_new_tokens=200).strip()

        score = _parse_score(response)

        feedback_parts = [f for f in response.split("Feedback:") if "<" not in f and ">" not in f]
        feedback = _first_paragraph(feedback_parts[1].strip()) if len(feedback_parts) >= 2 else ""
        return feedback, score

    # ------------------------------------------------------------------
    # Feedback merging
    # ------------------------------------------------------------------

    def generate_unified_feedback(
        self, expert_feedback: str, amateur_feedback: str, max_new_tokens: int = 200
    ) -> str:
        if amateur_feedback == "Failed to generate meaningful feedback":
            return expert_feedback.strip()

        prompt = (
            "Summarize two feedback comments (max 70 words). Prioritize expert; include unique amateur points.\n\n"
            "Expert Feedback: Clear and well-structured, but conclusion needs more data.\n"
            "Amateur Feedback: Flows well; shorten some sentences.\n"
            "Final Merged Feedback: Clear and logical. Strengthen conclusion with supporting data; minor sentence shortening improves readability.\n\n"
            f"Expert Feedback: {expert_feedback.strip()}\n"
            f"Amateur Feedback: {amateur_feedback.strip()}\n"
            "Final Merged Feedback:"
        )
        response = self.generate_text(prompt, max_new_tokens).strip()
        parts = [p.strip() for p in response.split("Final Merged Feedback:") if "<" not in p and ">" not in p and p.strip()]
        return _first_paragraph(parts[2] if len(parts) >= 3 else response)

    # ------------------------------------------------------------------
    # Answer revision
    # ------------------------------------------------------------------

    def apply_feedback(self, question: str, answer: str, feedback: str, max_new_tokens: int = 500) -> str:
        prompt = (
            "Revise the following answer using the feedback. Single paragraph, ~200 words.\n\n"
            "Revised Answer: <one-paragraph response>\n\n"
            f"Question: {question}\nOriginal Answer: {answer}\nFeedback: {feedback}\nRevised Answer:"
        )
        response = self.generate_text(prompt, max_new_tokens).strip()
        parts = [a.strip() for a in response.split("Revised Answer:") if "<" not in a and ">" not in a and a.strip()]
        return _first_paragraph(parts[1] if len(parts) >= 2 else response)

    def generate_self_critique(self, question: str, answer: str, max_new_tokens: int = 150) -> str:
        prompt = (
            "Critically evaluate your own answer (~70 words). Identify strengths and weaknesses.\n\n"
            "Self-Critique: <one-paragraph response>\n\n"
            f"Question: {question}\nAnswer: {answer}\nSelf-Critique:"
        )
        response = self.generate_text(prompt, max_new_tokens).strip()
        parts = [c.strip() for c in response.split("Self-Critique:") if "<" not in c and ">" not in c and c.strip()]
        return _first_paragraph(parts[2] if len(parts) >= 3 else "")

    def apply_self_critique(self, question: str, answer: str, critique: str, max_new_tokens: int = 500) -> str:
        prompt = (
            "Revise the answer using the self-critique. ~200 words, single paragraph.\n\n"
            "Final Answer: <one-paragraph response>\n\n"
            f"Question: {question}\nAnswer: {answer}\nSelf-Critique: {critique}\nFinal Answer:"
        )
        response = self.generate_text(prompt, max_new_tokens).strip()
        parts = [a.strip() for a in response.split("Final Answer:") if "<" not in a and ">" not in a and a.strip()]
        return _first_paragraph(parts[1] if len(parts) >= 2 else "")

    # ------------------------------------------------------------------
    # Full improvement loop
    # ------------------------------------------------------------------

    def improve_answer_with_feedback_and_critique(
        self,
        question: str,
        answer: str,
        epochs: int = 1,
        iterations: int = 20,
        target_score: float = 1.0,
        mode: str = "train",
        is_math: bool = False,
        examples: list | None = None,
    ) -> tuple[str, str, str, str]:
        assert self.network is not None, "ExpertFeedbackModel.network must be set before calling improve_answer."
        kd = mode == "train"
        feedback_dict = self.network.generate_combined_feedback(
            question, answer, is_math, examples, epochs, iterations, 0.65, kd
        )
        expert_fb = feedback_dict.get("expert_feedback", "")
        amateur_fb = feedback_dict.get("amateur_feedback", "")
        initial_score = float(feedback_dict.get("combined_score", -1.0))

        if initial_score >= target_score:
            return answer.strip(), expert_fb, amateur_fb, ""

        condensed = self.generate_unified_feedback(expert_fb, amateur_fb)
        unified = condensed.strip().split("\n\n")[0].strip()

        improved = self.apply_feedback(question, answer, unified)
        critique = self.generate_self_critique(question, improved)
        refined = self.apply_self_critique(question, improved, critique)

        return refined.strip(), expert_fb, amateur_fb, unified
