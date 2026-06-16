"""
Download and convert real datasets for the SLFD pipeline. All free HF downloads.

Two independent outputs:

  --processbench   ProcessBench -> eval JSONL with GOLD step-error labels.
                   Format eval expects: {problem, steps:[{text, is_error}]}.
                   No teacher needed: ProcessBench already labels the first error.

  --train_source gsm8k   GSM8K -> labeling INPUT JSONL for the teacher.
                   Format label_pipeline expects: {problem, solution, gt_answer}.
                   The teacher then scores/critiques each step offline.

Usage:
    # Eval set (math config, all examples):
    python -m scripts.download_data --processbench --config math \
        --output data/processbench_test.jsonl

    # Training source (300 GSM8K problems for the teacher to label):
    python -m scripts.download_data --train_source gsm8k --n 300 \
        --output data/raw/gsm8k_train.jsonl
"""
import os
import re
import json
import argparse


def _write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}")


def build_processbench(config: str, output: str, n: int | None):
    """Qwen/ProcessBench -> {problem, steps:[{text, is_error}]}.

    ProcessBench gives `label` = index of the first erroneous step (-1 if the
    whole solution is correct). We mark exactly that step as the error, which
    matches both the per-step F1 signal and the first-error-step metric.
    """
    from datasets import load_dataset
    ds = load_dataset("Qwen/ProcessBench", split=config)

    # ProcessBench has no GT columns. Join GT (final answer + full worked
    # solution) by problem text from the source dataset, so the teacher's
    # privileged conditions can be measured (eval_teacher / probe_privilege):
    #   gsm8k -> GSM8K test split
    #   math  -> Hendrycks MATH (train+test, 100% coverage of the 955 problems)
    # gt_map maps problem text -> {"answer", "solution"}.
    gt_map = {}
    if config == "gsm8k":
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        gt_map = {ex["question"].strip(): {
            "answer": _extract_gsm8k_answer(ex["answer"]),
            "solution": re.sub(r"\n?####.*$", "", ex["answer"]).strip(),
        } for ex in gsm}
    elif config == "math":
        for sp in ("train", "test"):
            for ex in load_dataset("nlile/hendrycks-MATH-benchmark", split=sp):
                p = (ex.get("problem") or "").strip()
                if p:
                    gt_map[p] = {"answer": ex.get("answer") or "",
                                 "solution": ex.get("solution") or ""}
    elif config == "olympiadbench":
        # Join from Hothan/OlympiadBench text+MM math configs (~96% coverage).
        # final_answer / solution can be lists — flatten to strings.
        def _flat(x):
            return ", ".join(map(str, x)) if isinstance(x, list) else (str(x) if x else "")
        for c in ("OE_TO_maths_en_COMP", "OE_MM_maths_en_COMP",
                  "TP_TO_maths_en_COMP", "TP_MM_maths_en_COMP"):
            try:
                src = load_dataset("Hothan/OlympiadBench", c, split="train")
            except Exception:
                continue
            for ex in src:
                p = (ex.get("question") or "").strip()
                if p:
                    gt_map[p] = {"answer": _flat(ex.get("final_answer")),
                                 "solution": _flat(ex.get("solution"))}

    rows, joined = [], 0
    for i, ex in enumerate(ds):
        if n and i >= n:
            break
        steps = ex.get("steps") or []
        label = ex.get("label", -1)
        if label is None:
            label = -1
        gt = gt_map.get(ex.get("problem", "").strip(), {})
        joined += bool(gt.get("answer"))
        rows.append({
            "problem": ex.get("problem", ""),
            "steps": [{"text": s, "is_error": (label >= 0 and j == label)}
                      for j, s in enumerate(steps)],
            "first_error_label": label,
            "gt_answer": gt.get("answer", ""),
            "gt_solution": gt.get("solution", ""),
        })
    if gt_map:
        print(f"GT joined for {joined}/{len(rows)} problems")
    _write_jsonl(output, rows)


def _extract_gsm8k_answer(answer: str) -> str:
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else ""


def build_gsm8k(output: str, n: int | None, split: str):
    """openai/gsm8k -> {problem, solution, gt_answer} for teacher labeling."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for i, ex in enumerate(ds):
        if n and i >= n:
            break
        answer = ex.get("answer", "")
        # Drop the "#### N" final-answer line from the step text; keep it as gt.
        solution = re.sub(r"\n?####.*$", "", answer).strip()
        rows.append({
            "problem": ex.get("question", ""),
            "solution": solution,
            "gt_answer": _extract_gsm8k_answer(answer),
        })
    _write_jsonl(output, rows)


def build_math_train(output: str, n: int | None, split: str = "train"):
    """Hendrycks MATH -> {problem, solution, gt_answer, gt_solution} training
    source. Carries gt_solution so labeling can run SOLUTION-privilege."""
    from datasets import load_dataset
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
    rows = []
    for i, ex in enumerate(ds):
        if n and i >= n:
            break
        p = (ex.get("problem") or "").strip()
        if not p:
            continue
        sol = ex.get("solution") or ""
        rows.append({"problem": p, "solution": sol,
                     "gt_answer": ex.get("answer") or "", "gt_solution": sol})
    _write_jsonl(output, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processbench", action="store_true",
                        help="Download ProcessBench eval set (gold step labels).")
    parser.add_argument("--config", default="math",
                        help="ProcessBench config: gsm8k | math | olympiadbench | omnimath")
    parser.add_argument("--train_source", choices=["gsm8k", "math"], default=None,
                        help="Download a training source for teacher labeling.")
    parser.add_argument("--split", default="train", help="Split for the training source.")
    parser.add_argument("--n", type=int, default=None, help="Cap number of examples.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.processbench:
        build_processbench(args.config, args.output, args.n)
    elif args.train_source == "gsm8k":
        build_gsm8k(args.output, args.n, args.split)
    elif args.train_source == "math":
        build_math_train(args.output, args.n, args.split)
    else:
        raise SystemExit("Pass --processbench or --train_source {gsm8k,math}.")


if __name__ == "__main__":
    main()
