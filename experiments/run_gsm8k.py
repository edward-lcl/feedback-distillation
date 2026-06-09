"""
GSM8K experiment: measure accuracy of our feedback-distillation method vs baselines.

Usage (from repo root):
  python -m experiments.run_gsm8k \
      --results_dir results/gsm8k \
      --max_samples 200 \
      --epochs 2 \
      --iterations 10
"""

import os
import csv
import re
import argparse
import json
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import ExpertFeedbackModel, AmateurFeedbackModel, ParsingModel, load_model_auto
from training import AmateurExpertFeedbackNetWork
from evaluation import compute_similarity, extract_gsm8k_answer, is_correct_gsm8k, evaluate_toxicity

MATH_FEW_SHOT = [
    {
        "question": "A train travels at 80 km/h for 3 hours. How far does it travel?",
        "answer": "Distance = Speed × Time = 80 × 3 = 240 km.",
        "expert_feedback": "Correct formula and clear step-by-step work.",
        "amateur_feedback": "Clear steps; add real-world context.",
        "score": 0.90,
    }
]


def load_models(teacher_name: str, student_name: str):
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    teacher_model = load_model_auto(teacher_name)

    student_tok = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)
    student_model = load_model_auto(student_name)

    expert = ExpertFeedbackModel(teacher_tok, teacher_model, teacher_name)
    amateur = AmateurFeedbackModel(student_tok, student_model, student_name)
    parser = ParsingModel(teacher_tok, teacher_model, teacher_name)
    return expert, amateur, parser


def load_kd_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append({"prompt": item["prompt"], "answer": item["original_answer"], "feedback": item["feedback"], "score": item["score"]})
    return data


def run_gsm8k(
    expert: ExpertFeedbackModel,
    amateur: AmateurFeedbackModel,
    parser: ParsingModel,
    expert_datasets: list[dict],
    results_dir: str,
    max_samples: int = 200,
    epochs: int = 2,
    iterations: int = 10,
):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "gsm8k_results.csv")

    network = AmateurExpertFeedbackNetWork(amateur, expert, expert_datasets, loss_flags=[True, True, True, True])
    expert.network = network

    ds = load_dataset("openai/gsm8k", "main", split="test")

    correct, total = 0, 0
    similarities, toxicities = [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "question", "initial_answer", "refined_answer", "pred", "gt", "correct",
                         "expert_feedback", "amateur_feedback", "cosine_sim", "toxicity_refined"])

        for i, sample in tqdm(enumerate(ds), total=min(max_samples, len(ds)), desc="GSM8K"):
            if i >= max_samples:
                break

            question = sample["question"]
            gt_text = sample["answer"]
            gt = extract_gsm8k_answer(gt_text)

            initial = expert.generate_answer(question, is_math=True, task_description="Solve the following math problem step-by-step", examples=MATH_FEW_SHOT)
            refined, exp_fb, am_fb, _ = expert.improve_answer_with_feedback_and_critique(
                question, initial, epochs=epochs, iterations=iterations, is_math=True, examples=MATH_FEW_SHOT
            )

            pred_raw = parser.generate_answer(refined, is_math=True)
            pred_raw = pred_raw.replace(",", "").strip()
            pred_raw = re.sub(r"[^\d.\-]", "", pred_raw)

            try:
                pred = float(pred_raw)
                correct_flag = abs(pred - gt) < 1e-6 if gt is not None else False
            except (ValueError, TypeError):
                pred = None
                correct_flag = False

            if correct_flag:
                correct += 1
            total += 1

            sim = compute_similarity(exp_fb, am_fb)
            tox = evaluate_toxicity(refined)
            similarities.append(sim)
            toxicities.append(tox)

            writer.writerow([i, question, initial, refined, pred, gt, int(correct_flag), exp_fb, am_fb, f"{sim:.4f}", f"{tox:.6f}"])

    acc = correct / total if total else 0.0
    print(f"\nGSM8K Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Avg cosine sim: {sum(similarities)/len(similarities):.4f}")
    print(f"Avg toxicity:   {sum(toxicities)/len(toxicities):.6f}")

    summary = {"accuracy": acc, "correct": correct, "total": total,
               "avg_cosine_sim": sum(similarities)/max(1,len(similarities)),
               "avg_toxicity": sum(toxicities)/max(1,len(toxicities))}
    with open(os.path.join(results_dir, "gsm8k_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--student", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--kd_dataset", default="data/300_sample.jsonl")
    parser.add_argument("--results_dir", default="results/gsm8k")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    expert, amateur, parse_model = load_models(args.teacher, args.student)
    kd_data = load_kd_dataset(args.kd_dataset)
    run_gsm8k(expert, amateur, parse_model, kd_data, args.results_dir, args.max_samples, args.epochs, args.iterations)
