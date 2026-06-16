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


def label_solution_omlx(problem, steps, gt_answer, api_url="http://localhost:8000/v1") -> list[dict]:
    from models.omlx_client import OmlxClient
    client = OmlxClient(api_url=api_url)
    labels = []
    prefix = ""
    for step in steps:
        labels.append(label_step_omlx(problem, prefix, step, gt_answer, client=client))
        prefix += step + "\n"
    return labels


def label_file(input_path: str, output_path: str, max_samples: int = None,
               dev_mode: bool = False, use_omlx: bool = False,
               omlx_url: str = "http://localhost:8000/v1"):
    if use_omlx:
        print(f"# Labeling via oMLX server at {omlx_url} (teacher not loaded locally)")
        teacher = None
    else:
        teacher = TeacherModel(dev_mode=dev_mode)

    i = -1
    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            problem = sample.get("problem", sample.get("prompt", ""))
            solution = sample.get("solution", sample.get("original_answer", ""))
            gt_answer = sample.get("answer", sample.get("gt_answer", ""))

            steps = segment_steps(solution)
            if use_omlx:
                labels = label_solution_omlx(problem, steps, gt_answer, api_url=omlx_url)
            else:
                labels = teacher.label_solution(problem, steps, gt_answer)

            record = {
                "problem": problem,
                "solution": solution,
                "gt_answer": gt_answer,
                "steps": [
                    {"text": step, **label}
                    for step, label in zip(steps, labels)
                ],
            }
            fout.write(json.dumps(record) + "\n")

    print(f"Labeled {min(i+1, max_samples or i+1)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dev_mode", action="store_true",
                        help="Use smaller models for local Apple Silicon development.")
    parser.add_argument("--use_omlx", action="store_true",
                        help="Call oMLX server (localhost:8000) instead of loading teacher locally.")
    parser.add_argument("--omlx_url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    mode = "oMLX API" if args.use_omlx else ("DEV (small models)" if args.dev_mode else "PROD")
    print(f"# label_pipeline starting — mode: {mode}")

    label_file(args.input, args.output, args.max_samples,
               dev_mode=args.dev_mode, use_omlx=args.use_omlx, omlx_url=args.omlx_url)
