"""
Generate a balanced mix of correct and INCORRECT solutions for teacher labeling.

Why: GSM8K reference solutions are correct by construction. Labeling them gives
~zero error labels, so the student would learn "every step is fine" and score 0
on error detection. Error-detection training needs flawed solutions — so we
sample them from a small generator model at temperature and keep a mix, split
by whether the final answer matches GT (Math-Shepherd-style).

Output JSONL is label_pipeline-compatible: {problem, solution, gt_answer}
plus provenance fields {model_answer, solution_correct}.

Usage:
    # Real run via local oMLX (set OMLX_API_KEY first):
    python -m scripts.generate_solutions \
        --input data/raw/gsm8k_train.jsonl --output data/raw/gsm8k_sampled.jsonl \
        --backend omlx --k 4

    # Local dev smoke (0.5B on MPS):
    python -m scripts.generate_solutions \
        --input data/raw/gsm8k_train.jsonl --output /tmp/sampled.jsonl \
        --backend local --dev_mode --max_problems 2 --k 2
"""
import re
import json
import argparse
from tqdm import tqdm

GEN_PROMPT = (
    "Solve this problem step by step. Write each step on its own line starting "
    "with 'Step 1:', 'Step 2:', etc. End with a final line exactly of the form "
    "'Final answer: <number>'.\n\nProblem: {problem}"
)


def extract_final_answer(text: str) -> str:
    m = re.search(r'(?:final answer|answer)\s*[:=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return nums[-1].replace(",", "") if nums else ""


def answers_match(a: str, b: str) -> bool:
    try:
        from math_verify import parse, verify
        if verify(parse(a), parse(b)):
            return True
    except Exception:
        pass

    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return bool(a) and a.strip() == str(b).strip()


def make_generator(backend: str, omlx_url: str, dev_mode: bool, temperature: float):
    if backend == "omlx":
        from models.omlx_client import OmlxClient
        client = OmlxClient(api_url=omlx_url)
        print(f"oMLX models available: {client.list_models()}")
        return lambda prompt: client.chat(prompt, max_tokens=512, temperature=temperature)
    # Native Hugging Face generator — fully replacing vLLM
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "google/gemma-4-26B-A4B-it"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    
    print(f"Native HF fallback: Loading {name} with CPU offload...")
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_memory={0: "22GB", 1: "22GB", "cpu": "200GB"},
        trust_remote_code=True
    )
    model.eval()

    def gen(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(formatted, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                                 temperature=temperature, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with {problem, gt_answer}.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["omlx", "local"], default="omlx")
    parser.add_argument("--omlx_url", default=None)
    parser.add_argument("--k", type=int, default=4, help="Max samples per problem.")
    parser.add_argument("--need_correct", type=int, default=1, help="Keep up to N correct per problem.")
    parser.add_argument("--need_incorrect", type=int, default=1, help="Keep up to N incorrect per problem.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--dev_mode", action="store_true")
    args = parser.parse_args()

    generate = make_generator(args.backend, args.omlx_url, args.dev_mode, args.temperature)

    with open(args.input) as f:
        problems = [json.loads(l) for l in f if l.strip()][: args.max_problems]

    def process_problem(p):
        problem, gt = p["problem"], str(p.get("gt_answer", ""))
        gt_solution = p.get("gt_solution", "")
        local_kept = []
        got_c, got_i = 0, 0
        for _ in range(args.k):
            if got_c >= args.need_correct and got_i >= args.need_incorrect:
                break
            text = generate(GEN_PROMPT.format(problem=problem)).strip()
            if not text:
                continue
            model_answer = extract_final_answer(text)
            correct = answers_match(model_answer, gt)
            if correct and got_c >= args.need_correct:
                continue
            if not correct and got_i >= args.need_incorrect:
                continue
            local_kept.append({
                "problem": problem,
                "solution": text,
                "gt_answer": gt,
                "gt_solution": gt_solution,
                "model_answer": model_answer,
                "solution_correct": correct,
            })
            got_c += correct
            got_i += (not correct)
        return local_kept, got_c, got_i

    kept, n_correct, n_incorrect = [], 0, 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=32 if args.backend == "omlx" else 1) as executor:
        futures = [executor.submit(process_problem, p) for p in problems]
        for future in tqdm(as_completed(futures), total=len(futures), desc="problems"):
            local_kept, c, i = future.result()
            kept.extend(local_kept)
            n_correct += c
            n_incorrect += i

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    total = len(kept)
    print(f"Kept {total} solutions ({n_correct} correct / {n_incorrect} incorrect) → {args.output}")
    if total and (n_incorrect == 0 or n_correct == 0):
        print("WARNING: one-sided mix — adjust --temperature/--k or generator model.")


if __name__ == "__main__":
    main()
