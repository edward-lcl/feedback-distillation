"""
CLEAR baseline (Rufail et al., 2025).

Three separate models: base, expert, amateur.
Base generates → expert + amateur give feedback → base revises.
No knowledge distillation. Amateur model is fixed.
"""

import re
import torch
from tqdm import tqdm
from datasets import load_dataset

from evaluation import compute_similarity, extract_gsm8k_answer, evaluate_toxicity, evaluate_rouge, evaluate_bleu, evaluate_bertscore


def _generate(model_pipeline, prompt: str, max_new_tokens: int = 400) -> str:
    out = model_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def _extract_numeric(text: str):
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", "").strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def run_clear_baseline(base_pipe, expert_pipe, amateur_pipe, dataset, task: str = "gsm8k", max_samples: int = 200):
    """
    Args:
        base_pipe / expert_pipe / amateur_pipe: HF pipeline("text-generation", model=...)
        dataset: iterable of samples with at minimum "question" and "answer" keys
        task: "gsm8k" | "alpaca"
    """
    correct, total = 0, 0
    bert_f1s, rouge_ls, bleus, toxicities, similarities = [], [], [], [], []

    for i, sample in tqdm(enumerate(dataset), total=max_samples, desc="CLEAR"):
        if i >= max_samples:
            break

        if task == "gsm8k":
            question = sample["question"]
            gt_text = sample["answer"]
            gt = extract_gsm8k_answer(gt_text)

            base_answer = _generate(base_pipe, f"Solve step by step.\nQuestion: {question}\nAnswer:")
            expert_feedback = _generate(expert_pipe, f"Give expert feedback on this math answer.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback:")
            amateur_feedback = _generate(amateur_pipe, f"Give brief feedback on this math answer.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback:")

            merged = f"Expert: {expert_feedback}\nAmateur: {amateur_feedback}"
            revised = _generate(base_pipe, f"Revise using this feedback.\nQuestion: {question}\nOriginal: {base_answer}\nFeedback: {merged}\nRevised Answer:")
            self_critique = _generate(base_pipe, f"Self-critique this answer.\nAnswer: {revised}\nSelf-Critique:")
            final = _generate(base_pipe, f"Apply self-critique to finalize.\nAnswer: {revised}\nCritique: {self_critique}\nFinal Answer:")

            pred_raw = re.sub(r"[^\d.\-]", "", final.replace(",", "").strip())
            try:
                pred = float(pred_raw)
                if gt is not None and abs(pred - gt) < 1e-6:
                    correct += 1
            except ValueError:
                pass
            total += 1
            tox = evaluate_toxicity(final)
            toxicities.append(tox)
            sim = compute_similarity(expert_feedback, amateur_feedback)
            similarities.append(sim)

        else:  # alpaca
            instruction = sample["instruction"]
            reference = sample.get("output", "")
            ctx = sample.get("input", "")
            question = f"{instruction}\n{ctx}".strip() if ctx else instruction

            base_answer = _generate(base_pipe, f"Answer concisely.\n{question}\nAnswer:")
            expert_feedback = _generate(expert_pipe, f"Expert feedback.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback:")
            amateur_feedback = _generate(amateur_pipe, f"Brief feedback.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback:")
            merged = f"Expert: {expert_feedback}\nAmateur: {amateur_feedback}"
            final = _generate(base_pipe, f"Revise.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback: {merged}\nRevised:")

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
