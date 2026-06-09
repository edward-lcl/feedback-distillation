"""
Chain-of-Draft (CoD) baseline.
Uses a structured outline-style feedback prompt before revision.
"""

import re
from tqdm import tqdm
from evaluation import extract_gsm8k_answer, evaluate_toxicity, evaluate_rouge, evaluate_bleu, evaluate_bertscore


def _generate(pipe, prompt: str, max_new_tokens: int = 400) -> str:
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def run_cod_baseline(base_pipe, dataset, task: str = "gsm8k", max_samples: int = 200):
    correct, total = 0, 0
    bert_f1s, rouge_ls, bleus, toxicities = [], [], [], []

    for i, sample in tqdm(enumerate(dataset), total=max_samples, desc="CoD"):
        if i >= max_samples:
            break

        if task == "gsm8k":
            question = sample["question"]
            gt_text = sample["answer"]
            gt = extract_gsm8k_answer(gt_text)

            base_answer = _generate(base_pipe, f"Solve this math problem.\nQuestion: {question}\nAnswer:")

            cod_prompt = (
                f"Evaluate this math answer. First write an outline of your evaluation criteria, "
                f"then provide a one-paragraph feedback summary.\n"
                f"Question: {question}\nAnswer: {base_answer}\n"
                "Outline:\n1. Correctness\n2. Clarity\n3. Completeness\n\nFeedback Summary:"
            )
            feedback = _generate(base_pipe, cod_prompt)
            final = _generate(base_pipe, f"Revise.\nQuestion: {question}\nAnswer: {base_answer}\nFeedback: {feedback}\nRevised:")

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
            base_answer = _generate(base_pipe, f"Answer.\n{question}\nAnswer:")
            cod_prompt = (
                f"Outline evaluation criteria, then give a one-paragraph feedback.\n"
                f"Q: {question}\nA: {base_answer}\nFeedback:"
            )
            feedback = _generate(base_pipe, cod_prompt)
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
