import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _first_paragraph(text: str) -> str:
    for part in text.strip().split("\n\n"):
        if part.strip():
            return part.strip()
    return text.strip()


class ParsingModel:
    """Uses the teacher model to extract a clean final answer from generated text."""

    def __init__(self, tokenizer, model, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"ParsingModel: {model_name} on {self.device}")

    def _build_prompt(self, answer: str, is_math: bool) -> str:
        if is_math:
            return (
                "Extract only the final numerical value from the answer (integer, decimal, fraction, or %). "
                "No words or units.\n\n"
                "Answer: The total cost is $120, so \\boxed{120}.\n"
                "Extracted final numerical answer: 120\n\n"
                f"Answer: {answer}\n"
                "Extracted final numerical answer:"
            )
        return (
            "Copy the shortest span that directly answers the question. One line: "
            "'Extracted final answer: <span>'\n\n"
            "Answer: The capital of France is Paris.\n"
            "Extracted final answer: Paris\n\n"
            f"Answer: {answer}\n"
            "Extracted final answer:"
        )

    def generate_answer(self, answer: str, is_math: bool = False, max_new_tokens: int = 50) -> str:
        prompt = self._build_prompt(answer, is_math)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _first_paragraph(text[len(prompt):].strip())
