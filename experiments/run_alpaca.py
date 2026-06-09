"""
Alpaca experiment: BERTScore / ROUGE-L / BLEU / toxicity of refined answers.

Usage:
  python -m experiments.run_alpaca \
      --results_dir results/alpaca \
      --max_samples 200
"""

import os
import csv
import json
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import ExpertFeedbackModel, AmateurFeedbackModel, ParsingModel
from training import AmateurExpertFeedbackNetWork
from evaluation import compute_similarity, evaluate_bertscore, evaluate_rouge, evaluate_bleu, evaluate_toxicity


def run_alpaca(
    expert: ExpertFeedbackModel,
    amateur: AmateurFeedbackModel,
    expert_datasets: list[dict],
    results_dir: str,
    max_samples: int = 200,
    epochs: int = 1,
    iterations: int = 20,
):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "alpaca_results.csv")

    network = AmateurExpertFeedbackNetWork(amateur, expert, expert_datasets, loss_flags=[True, True, True, True])
    expert.network = network

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    split = ds.train_test_split(test_size=0.1, seed=42)
    test_ds = split["test"]

    all_preds, all_refs = [], []
    bert_f1s, rouge_ls, bleus, toxicities = [], [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "instruction", "initial_answer", "refined_answer", "reference",
                         "bert_f1", "rougeL", "bleu", "toxicity_refined", "cosine_sim"])

        for i, sample in tqdm(enumerate(test_ds), total=min(max_samples, len(test_ds)), desc="Alpaca"):
            if i >= max_samples:
                break

            instruction = sample["instruction"]
            reference = sample["output"]
            ctx = sample.get("input", "")
            question = f"{instruction}\n{ctx}".strip() if ctx else instruction

            initial = expert.generate_answer(question, is_math=False)
            refined, exp_fb, am_fb, _ = expert.improve_answer_with_feedback_and_critique(
                question, initial, epochs=epochs, iterations=iterations, is_math=False
            )

            bert = evaluate_bertscore([refined], [reference])["f1"]
            rouge = evaluate_rouge(refined, reference)["rougeL_f"]
            bleu = evaluate_bleu(refined, reference)
            tox = evaluate_toxicity(refined)
            sim = compute_similarity(exp_fb, am_fb)

            bert_f1s.append(bert)
            rouge_ls.append(rouge)
            bleus.append(bleu)
            toxicities.append(tox)
            all_preds.append(refined)
            all_refs.append(reference)

            writer.writerow([i, instruction, initial, refined, reference,
                             f"{bert:.4f}", f"{rouge:.4f}", f"{bleu:.4f}", f"{tox:.6f}", f"{sim:.4f}"])

    summary = {
        "avg_bert_f1": sum(bert_f1s) / max(1, len(bert_f1s)),
        "avg_rougeL": sum(rouge_ls) / max(1, len(rouge_ls)),
        "avg_bleu": sum(bleus) / max(1, len(bleus)),
        "avg_toxicity": sum(toxicities) / max(1, len(toxicities)),
        "n": len(bert_f1s),
    }
    print(f"\nAlpaca results: {json.dumps(summary, indent=2)}")
    with open(os.path.join(results_dir, "alpaca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--student", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--kd_dataset", default="data/300_sample.jsonl")
    parser.add_argument("--results_dir", default="results/alpaca")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, json as _json

    def _load_models(t, s):
        tt = AutoTokenizer.from_pretrained(t, trust_remote_code=True)
        tm = AutoModelForCausalLM.from_pretrained(t, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        st = AutoTokenizer.from_pretrained(s, trust_remote_code=True)
        sm = AutoModelForCausalLM.from_pretrained(s, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        return ExpertFeedbackModel(tt, tm, t), AmateurFeedbackModel(st, sm, s)

    def _load_kd(path):
        data = []
        with open(path) as f:
            for line in f:
                item = _json.loads(line)
                data.append({"prompt": item["prompt"], "answer": item["original_answer"], "feedback": item["feedback"], "score": item["score"]})
        return data

    expert, amateur = _load_models(args.teacher, args.student)
    kd_data = _load_kd(args.kd_dataset)
    run_alpaca(expert, amateur, kd_data, args.results_dir, args.max_samples, args.epochs, args.iterations)
