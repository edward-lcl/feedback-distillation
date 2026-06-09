"""
Chain-of-Thought (CoT) baseline (Wei et al., 2022).
Base model generates step-by-step feedback, then revises its own answer.
"""

import re
from tqdm import tqdm
from evaluation import extract_gsm8k_answer, evaluate_toxicity, evaluate_rouge, evaluate_bleu, evaluate_bertscore


def _generate(pipe, prompt: str, max_new_tokens: int = 400) -> str:
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def run_cot_baseline(base_pipe, dataset, task: str = "gsm8k", max_samples: int = 200):
    correct, total = 0, 0
    bert_f1s, rouge_ls, bleus, toxicities = [], [], [], []

    for i, sample in tqdm(enumerate(dataset), total=max_samples, desc="CoT"):
        if i >= max_samples:
            break

        if task == "gsm8k":
            question = sample["question"]
            gt_text = sample["answer"]
            gt = extract_gsm8k_answer(gt_text)

            cot_prompt = (
                "Solve the following math problem step by step. "
                "Think through each step carefully before giving your final answer.\n"
                f"Question: {question}\n"
                "Let me think step by step:\n"
            )
            base_answer = _generate(base_pipe, cot_prompt)

            feedback_prompt = (
                f"You are evaluating a math solution.\nQuestion: {question}\n"
                f"Solution: {base_answer}\n"
                "Provide detailed step-by-step feedback on how to improve this solution:\n"
            )
            feedback = _generate(base_pipe, feedback_prompt)
            final = _generate(base_pipe, f"Revise using feedback.\nQuestion: {question}\nOriginal: {base_answer}\nFeedback: {feedback}\nRevised Answer:")

            pred_raw = re.sub(r"[^\d.\-]", "", final.replace(",", "").strip())
            try:
                pred = float(pred_raw)
                if gt is not None and abs(pred - gt) < 1e-6:
                    correct += 1
            except ValueError:
                pass
            total += 1
            toxicities.append(evaluate_toxicity(final))

        else:
            question = sample.get("instruction", "")
            reference = sample.get("output", "")
            base_answer = _generate(base_pipe, f"Answer step by step.\n{question}\nAnswer:")
            feedback = _generate(base_pipe, f"Give detailed feedback.\nQ: {question}\nA: {base_answer}\nFeedback:")
            final = _generate(base_pipe, f"Revise.\nQ: {question}\nA: {base_answer}\nFeedback: {feedback}\nRevised:")
            bert_f1s.append(evaluate_bertscore([final], [reference])["f1"])
            rouge_ls.append(evaluate_rouge(final, reference)["rougeL_f"])
            bleus.append(evaluate_bleu(final, reference))
            toxicities.append(evaluate_toxicity(final))

    if task == "gsm8k":
        acc = correct / total if total else 0.0
        return {"accuracy": acc, "avg_toxicity": sum(toxicities) / max(1, len(toxicities))}
    else:
        return {
            "avg_bert_f1": sum(bert_f1s) / max(1, len(bert_f1s)),
            "avg_rougeL": sum(rouge_ls) / max(1, len(rouge_ls)),
            "avg_bleu": sum(bleus) / max(1, len(bleus)),
            "avg_toxicity": sum(toxicities) / max(1, len(toxicities)),
        }
