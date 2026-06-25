"""
Offline teacher labeling pipeline.
Takes (problem, solution, gt_answer) and produces step-level labels.
Outputs JSONL: {problem, solution, steps: [{text, score, feedback, is_error}]}

Usage:
    python -m data.label_pipeline \
        --input data/raw/math_shepherd_sample.jsonl \
        --output data/labeled/math_shepherd_labeled.jsonl \
        --max_samples 500

    # Local Apple Silicon dev mode (smaller models):
    python -m data.label_pipeline --input ... --output ... --dev_mode

    # Use oMLX server (localhost:8000) instead of loading teacher locally:
    python -m data.label_pipeline --input ... --output ... --use_omlx
"""
import json
import os
import re
import argparse
from tqdm import tqdm
from models.teacher import TeacherModel, _parse_score
from data.step_segmentation import segment_steps

# Reasoning ("thinking") teachers emit a chain of thought before the answer —
# either inside <think>...</think> tags or as plain prose. We must (a) give them
# enough token budget to reach the "Score:" line, and (b) strip any think block
# before parsing so leftover reasoning text can't confuse the score/feedback
# extraction. The cap is a ceiling, not a fixed cost: greedy decoding stops at
# EOS once the model emits its Score/Feedback, so a generous budget is ~free
# except on the rare step where the model rambles.
OMLX_LABEL_MAX_TOKENS = int(os.environ.get("OMLX_LABEL_MAX_TOKENS", "1024"))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks. If a think block is left unclosed (the
    model ran out of budget mid-reasoning) keep only the text after the last
    </think>, or empty if it never closed — there is no parseable answer there."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        text = ""
    return text.strip()


def label_step_omlx(problem, solution_prefix, step_text, gt_answer, client=None,
                    api_url="http://localhost:8000/v1",
                    max_tokens=OMLX_LABEL_MAX_TOKENS, gt_solution=None) -> dict:
    """Call oMLX server for step labeling instead of loading model locally.

    Privilege level (strongest first): gt_solution (full worked reference) >
    gt_answer (bare final number) > neither (GT-free, for the privileged gap)."""
    from models.omlx_client import OmlxClient
    client = client or OmlxClient(api_url=api_url)
    if gt_solution is not None:
        prompt = TeacherModel.STEP_EVAL_PROMPT_SOLUTION.format(
            problem=problem, solution_prefix=solution_prefix,
            step_text=step_text, gt_solution=gt_solution,
        )
    elif gt_answer is not None:
        prompt = TeacherModel.STEP_EVAL_PROMPT.format(
            problem=problem, solution_prefix=solution_prefix,
            step_text=step_text, gt_answer=gt_answer,
        )
    else:
        prompt = TeacherModel.STEP_EVAL_PROMPT_NO_GT.format(
            problem=problem, solution_prefix=solution_prefix, step_text=step_text,
        )
    content = _strip_think(client.chat(prompt, max_tokens=max_tokens, temperature=0.0))
    score = _parse_score(content)
    parts = content.split("Feedback:", 1)
    feedback = parts[1].strip().split("\n")[0].strip() if len(parts) > 1 else ""
    return {
        "score": score,
        "feedback": feedback,
        "is_error": (score < 0.0) if score is not None else None,
        "parse_failed": score is None,
    }


def label_solution_omlx(problem, steps, gt_answer, gt_solution=None,
                        api_url="http://localhost:8000/v1") -> list[dict]:
    from models.omlx_client import OmlxClient
    client = OmlxClient(api_url=api_url)
    labels = []
    prefix = ""
    for step in steps:
        labels.append(label_step_omlx(problem, prefix, step, gt_answer,
                                      client=client, gt_solution=gt_solution))
        prefix += step + "\n"
    return labels


def label_file(input_path: str, output_path: str, max_samples: int = None,
               dev_mode: bool = False, use_omlx: bool = False,
               omlx_url: str = "http://localhost:8000/v1", privilege: str = "solution", local_model: str = None):
    """privilege controls what the teacher sees while labeling — the core
    'privileged vs no-GT' distillation comparison:
      solution -> full worked reference solution (richest privilege)
      answer   -> bare final answer
      none     -> nothing (GT-free labels)
    """
    if use_omlx:
        print(f"# Labeling via oMLX at {omlx_url} (privilege={privilege})")
        teacher = None
    else:
        teacher = TeacherModel(model_name=local_model, dev_mode=dev_mode)

    def process_line(line):
        sample = json.loads(line)
        problem = sample.get("problem", sample.get("prompt", ""))
        solution = sample.get("solution", sample.get("original_answer", ""))
        gt_answer = sample.get("answer", sample.get("gt_answer", ""))
        gt_solution = sample.get("gt_solution", "")

        if privilege == "solution":
            pa, ps = None, (gt_solution or None)
        elif privilege == "answer":
            pa, ps = (gt_answer or None), None
        else:
            pa, ps = None, None

        steps = segment_steps(solution)
        if use_omlx:
            labels = label_solution_omlx(problem, steps, pa, gt_solution=ps, api_url=omlx_url)
        else:
            labels = teacher.label_solution(problem, steps, pa)

        return {
            "problem": problem,
            "solution": solution,
            "gt_answer": gt_answer,
            "privilege": privilege,
            "steps": [
                {"text": step, **label}
                for step, label in zip(steps, labels)
            ],
        }

    i = -1
    with open(input_path) as fin:
        lines = fin.readlines()
    if max_samples:
        lines = lines[:max_samples]

    def sample_key(sample: dict) -> str:
        return sample.get("problem", sample.get("prompt", ""))

    existing_problems = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_problems.add(sample_key(json.loads(line)))
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: found {len(existing_problems)} existing labels in {output_path}")

    with open(output_path, "a") as fout:
        if use_omlx:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for line in lines:
                    sample = json.loads(line)
                    if sample_key(sample) in existing_problems:
                        continue
                    futures.append(executor.submit(process_line, line))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    fout.write(json.dumps(future.result()) + "\n")
                    fout.flush()
            i = len(lines) - 1
        else:
            for i, line in enumerate(tqdm(lines)):
                sample = json.loads(line)
                if sample_key(sample) in existing_problems:
                    continue
                fout.write(json.dumps(process_line(line)) + "\n")
                fout.flush()

    print(f"Finished processing into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dev_mode", action="store_true",
                        help="Use smaller models for local Apple Silicon development.")
    parser.add_argument("--use_omlx", action="store_true",
                        help="Call oMLX server (OMLX_URL) instead of loading teacher locally.")
    # Respect OMLX_URL by default so a remote/tunneled teacher works without
    # passing --omlx_url. (Previously hardcoded :8000 → 404 when served elsewhere.)
    parser.add_argument("--omlx_url", default=os.environ.get("OMLX_URL", "http://localhost:8000/v1"))
    parser.add_argument("--privilege", choices=["solution", "answer", "none"], default="solution",
                        help="What the teacher sees while labeling (privileged vs no-GT comparison).")
    parser.add_argument("--local_model", default=None,
                        help="Override the default teacher model name (e.g. google/gemma-2-2b-it)")
    args = parser.parse_args()

    mode = "oMLX API" if args.use_omlx else ("DEV (small models)" if args.dev_mode else "PROD")
    print(f"# label_pipeline starting — mode: {mode}, privilege: {args.privilege}")

    omlx_url = os.environ.get("OMLX_URL") or args.omlx_url
    label_file(args.input, args.output, args.max_samples,
               dev_mode=args.dev_mode, use_omlx=args.use_omlx, omlx_url=omlx_url,
               privilege=args.privilege, local_model=args.local_model)
