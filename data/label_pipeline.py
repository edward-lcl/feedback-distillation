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
import argparse
from tqdm import tqdm
from models.teacher import TeacherModel, _parse_score
from data.step_segmentation import segment_steps


def label_step_omlx(problem, solution_prefix, step_text, gt_answer, api_url="http://localhost:8000/v1") -> dict:
    """Call oMLX server for step labeling instead of loading model locally."""
    import requests
    prompt = TeacherModel.STEP_EVAL_PROMPT.format(
        problem=problem, solution_prefix=solution_prefix,
        step_text=step_text, gt_answer=gt_answer,
    )
    resp = requests.post(f"{api_url}/chat/completions", json={
        "model": "default",  # oMLX serves whatever is loaded
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.0,
    }, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    score = _parse_score(content)
    parts = content.split("Feedback:", 1)
    feedback = parts[1].strip().split("\n")[0].strip() if len(parts) > 1 else ""
    return {"score": score, "feedback": feedback, "is_error": score < 0.0}


def label_solution_omlx(problem, steps, gt_answer, api_url="http://localhost:8000/v1") -> list[dict]:
    labels = []
    prefix = ""
    for step in steps:
        labels.append(label_step_omlx(problem, prefix, step, gt_answer, api_url=api_url))
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
