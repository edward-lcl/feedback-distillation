"""Build the same-source GSM8K labeling input (HANDOFF_SAKSHAM.md steps 1-2).

Takes the 400 ProcessBench GSM8K source problems (the exact ones behind the
gold-label rows) and recovers each problem's official worked reference
solution from openai/gsm8k so the Gemma-4 teacher can label the same
candidates with full privilege. Output rows: {problem, solution, gt_answer,
gt_solution}, where solution is the ProcessBench candidate with steps joined
by blank lines (round-trips through segment_steps' paragraph split).
"""

import argparse
import json
import re
import sys

CALC_RE = re.compile(r"<<[^>]*>>")


def normalize(text: str) -> str:
    return " ".join(text.split())


def load_gsm8k_reference_map():
    from datasets import load_dataset

    ref = {}
    for split in ("test", "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            ref.setdefault(normalize(row["question"]), row["answer"])
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processbench_gsm8k.jsonl")
    parser.add_argument("--output", default="data/gsm8k400_for_labeling.jsonl")
    args = parser.parse_args()

    ref = load_gsm8k_reference_map()
    matched, missed, out = 0, [], []
    with open(args.input) as fin:
        for line in fin:
            sample = json.loads(line)
            key = normalize(sample["problem"])
            gt = ref.get(key)
            if gt is None:
                missed.append(sample["problem"][:80])
                continue
            matched += 1
            # Strip calculator annotations; keep the worked steps and the
            # final "#### N" line (the teacher prompt treats gt_solution as
            # the full worked reference).
            gt_solution = CALC_RE.sub("", gt).strip()
            steps = [" ".join(s["text"].split("\n")) for s in sample["steps"]]
            out.append({
                "problem": sample["problem"],
                "solution": "\n\n".join(steps),
                "gt_answer": sample["gt_answer"],
                "gt_solution": gt_solution,
                "first_error_label": sample.get("first_error_label"),
            })

    with open(args.output, "w") as fout:
        for row in out:
            fout.write(json.dumps(row) + "\n")

    print(f"matched {matched}, missed {len(missed)} -> {args.output}")
    if missed:
        print("MISSED PROBLEMS (first 5):", *missed[:5], sep="\n  ", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
