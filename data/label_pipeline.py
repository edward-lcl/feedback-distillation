"""
Offline teacher labeling pipeline.
Takes (problem, solution, gt_answer) and produces step-level labels.
Outputs JSONL: {problem, solution, steps: [{text, score, feedback, is_error}]}

Usage:
    python -m data.label_pipeline \
        --input data/raw/math_shepherd_sample.jsonl \
        --output data/labeled/math_shepherd_labeled.jsonl \
        --max_samples 500
"""
import json
import argparse
from tqdm import tqdm
from models.teacher import TeacherModel
from data.step_segmentation import segment_steps


def label_file(input_path: str, output_path: str, max_samples: int = None):
    teacher = TeacherModel()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            problem = sample.get("problem", sample.get("prompt", ""))
            solution = sample.get("solution", sample.get("original_answer", ""))
            gt_answer = sample.get("answer", sample.get("gt_answer", ""))

            steps = segment_steps(solution)
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
    args = parser.parse_args()
    label_file(args.input, args.output, args.max_samples)
