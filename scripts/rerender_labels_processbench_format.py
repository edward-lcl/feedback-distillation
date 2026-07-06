"""Re-render generated teacher labels into ProcessBench gold-label format.

Completes the provenance x format design (PAPER_FRAMING.md, experiment #2):
the same-source GSM8K cell showed generated labels stay weak when the source
problems are held fixed, but the generated rows still differ from the gold
rows in labeling convention, not just provenance. This script removes the
convention differences so ONLY the label source (teacher judgment vs human
annotation) varies:

  gold convention (results/diagnostics/processbench_gsm8k_gold_train400_steps.jsonl):
    - exactly the FIRST error step per solution is flagged; later steps are
      kept and labeled non-error
    - score is binary (+1.0 correct / -1.0 error), feedback is the literal
      string "Correct." / "Error."
  generated convention (data/labeled/math_*_gsm8k400.jsonl):
    - every step judged independently (errors cascade past the first)
    - graded scores in {-1,-0.5,0,0.5,1}, free-text teacher feedback

Input: unflattened labeled records {problem, solution, gt_answer,
steps: [{text, score, feedback, is_error}]} — grouped per candidate solution,
NOT per problem text (ProcessBench GSM8K repeats 25 problem texts across the
400 candidates). Output: the same records re-rendered, plus a flattened
training file via data.flatten_labels (identical flattening path to the
files Saksham already trained on).
"""

import argparse
import json

from data.flatten_labels import flatten_labeled_records


def rerender_record(record: dict) -> dict:
    steps = record["steps"]
    first_error = next(
        (i for i, s in enumerate(steps)
         if bool(s.get("is_error", float(s.get("score", 0.0)) < 0.0))),
        None,
    )
    new_steps = []
    for i, s in enumerate(steps):
        is_error = first_error is not None and i == first_error
        new_steps.append({
            "text": s["text"],
            "score": -1.0 if is_error else 1.0,
            "feedback": "Error." if is_error else "Correct.",
            "is_error": is_error,
        })
    out = dict(record)
    out["steps"] = new_steps
    out["teacher_first_error"] = first_error
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="unflattened labeled JSONL (one record per candidate solution)")
    parser.add_argument("--output", required=True,
                        help="re-rendered unflattened JSONL")
    parser.add_argument("--output_steps", required=True,
                        help="flattened per-step training JSONL")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    rerendered = [rerender_record(r) for r in records]

    with open(args.output, "w") as f:
        for r in rerendered:
            f.write(json.dumps(r) + "\n")
    flat = flatten_labeled_records(rerendered)
    with open(args.output_steps, "w") as f:
        for row in flat:
            f.write(json.dumps(row) + "\n")

    n_sol = len(rerendered)
    n_with_err = sum(1 for r in rerendered if r["teacher_first_error"] is not None)
    n_rows = len(flat)
    n_err_rows = sum(r["is_error"] for r in flat)
    print(f"{n_sol} solutions ({n_with_err} with a teacher-flagged error), "
          f"{n_rows} step rows, error rate {n_err_rows / n_rows:.3f} "
          f"(was per-step independent; now first-error-only)")


if __name__ == "__main__":
    main()
