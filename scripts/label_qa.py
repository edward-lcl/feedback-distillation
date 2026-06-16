"""
QA report for teacher-labeled data — run BEFORE training on it.

Surfaces the failure modes that silently ruin a training run:
  - parse-failure rate (unparseable teacher replies → dropped labels)
  - class balance (% error steps — near-zero means the student can't learn
    error detection; that's the all-correct-solutions trap)
  - score distribution (degenerate teachers emit one value everywhere)
  - critique quality proxy (empty/short feedback on error steps)

Usage:
    python -m scripts.label_qa --input data/labeled/gsm8k_labeled.jsonl
"""
import json
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    n_solutions = n_steps = n_parse_failed = n_error = 0
    n_solutions_with_error = 0
    empty_fb_on_error = fb_len_total = fb_count = 0
    score_buckets = Counter()

    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_solutions += 1
            has_error = False
            for st in rec.get("steps", []):
                n_steps += 1
                score = st.get("score")
                if st.get("parse_failed") or score is None:
                    n_parse_failed += 1
                    continue
                bucket = round(score * 2) / 2  # 0.5-wide buckets in [-1, 1]
                score_buckets[bucket] += 1
                fb = (st.get("feedback") or "").strip()
                if fb:
                    fb_len_total += len(fb)
                    fb_count += 1
                if st.get("is_error"):
                    n_error += 1
                    has_error = True
                    empty_fb_on_error += (not fb)
            n_solutions_with_error += has_error

    valid = n_steps - n_parse_failed
    print(f"solutions:               {n_solutions}")
    print(f"steps:                   {n_steps}")
    print(f"parse failures:          {n_parse_failed} ({n_parse_failed / max(1, n_steps):.1%})")
    print(f"valid labeled steps:     {valid}")
    print(f"error steps:             {n_error} ({n_error / max(1, valid):.1%} of valid)")
    print(f"solutions w/ >=1 error:  {n_solutions_with_error} ({n_solutions_with_error / max(1, n_solutions):.1%})")
    print(f"avg feedback length:     {fb_len_total / max(1, fb_count):.0f} chars ({fb_count} non-empty)")
    print(f"error steps w/o feedback:{empty_fb_on_error}")
    print("score distribution:")
    for b in sorted(score_buckets):
        c = score_buckets[b]
        print(f"  {b:+.1f}: {'#' * max(1, int(40 * c / max(score_buckets.values())))} {c}")

    # Red flags
    if n_parse_failed / max(1, n_steps) > 0.10:
        print("\nRED FLAG: >10% parse failures — fix the teacher prompt/format first.")
    if valid and n_error / valid < 0.05:
        print("\nRED FLAG: <5% error steps — training data is nearly all-correct; "
              "the student cannot learn error detection from this. Generate flawed "
              "solutions (scripts.generate_solutions) instead of labeling reference ones.")
    if len(score_buckets) == 1:
        print("\nRED FLAG: degenerate score distribution (single value).")


if __name__ == "__main__":
    main()
