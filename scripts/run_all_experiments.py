"""
Master experiment runner.

Runs our method + all baselines on GSM8K and Alpaca, then saves a
consolidated results table.

Usage (on RunPod/Colab):
  python scripts/run_all_experiments.py \
      --kd_dataset data/300_sample.jsonl \
      --results_dir results/ \
      --max_samples 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from models import ExpertFeedbackModel, AmateurFeedbackModel, ParsingModel
from experiments.run_gsm8k import run_gsm8k, load_models, load_kd_dataset
from experiments.run_alpaca import run_alpaca
from baselines import run_clear_baseline, run_cot_baseline, run_cod_baseline
from datasets import load_dataset


def load_pipeline(model_name: str):
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Load models ----
    print("Loading teacher + student models...")
    expert, amateur, parse_model = load_models(args.teacher, args.student)
    kd_data = load_kd_dataset(args.kd_dataset)

    # ---- GSM8K ----
    print("\n=== GSM8K ===")
    gsm8k_ours = run_gsm8k(expert, amateur, parse_model, kd_data,
                            os.path.join(args.results_dir, "gsm8k"),
                            args.max_samples, args.epochs, args.iterations)

    # Baselines share the same base model pipeline
    base_pipe = load_pipeline(args.teacher)

    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_cot = run_cot_baseline(base_pipe, gsm8k_ds, "gsm8k", args.max_samples)
    gsm8k_cod = run_cod_baseline(base_pipe, gsm8k_ds, "gsm8k", args.max_samples)
    gsm8k_clear = run_clear_baseline(base_pipe, base_pipe, base_pipe, gsm8k_ds, "gsm8k", args.max_samples)

    # ---- Alpaca ----
    print("\n=== Alpaca ===")
    alpaca_ours = run_alpaca(expert, amateur, kd_data,
                             os.path.join(args.results_dir, "alpaca"),
                             args.max_samples, args.epochs, args.iterations)

    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train").train_test_split(test_size=0.1, seed=42)["test"]
    alpaca_cot = run_cot_baseline(base_pipe, alpaca_ds, "alpaca", args.max_samples)
    alpaca_cod = run_cod_baseline(base_pipe, alpaca_ds, "alpaca", args.max_samples)
    alpaca_clear = run_clear_baseline(base_pipe, base_pipe, base_pipe, alpaca_ds, "alpaca", args.max_samples)

    # ---- Consolidated table ----
    table = {
        "GSM8K": {
            "Ours":  gsm8k_ours,
            "CoT":   gsm8k_cot,
            "CoD":   gsm8k_cod,
            "CLEAR": gsm8k_clear,
        },
        "Alpaca": {
            "Ours":  alpaca_ours,
            "CoT":   alpaca_cot,
            "CoD":   alpaca_cod,
            "CLEAR": alpaca_clear,
        },
    }

    out_path = os.path.join(args.results_dir, "consolidated_results.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"\n=== Results saved to {out_path} ===")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--student", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--kd_dataset", default="data/300_sample.jsonl")
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    main(p.parse_args())
