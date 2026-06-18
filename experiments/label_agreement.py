"""How much do the privileged and no-GT teacher labels actually differ?

If the two labeled sets barely disagree, "no transfer to the student" is
explained directly: the student had near-identical training targets either way.
This reads the two labeled files (same source solutions, same order) and reports
per-step agreement on the `is_error` decision plus the score gap.

    python -m experiments.label_agreement \
        --priv data/labeled/math_priv.jsonl --nogt data/labeled/math_nogt.jsonl

Labeled JSONL format (from data/label_pipeline.py):
    {problem, solution, steps: [{text, score, feedback, is_error, parse_failed}]}
"""
import json
import argparse


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv", default="data/labeled/math_priv.jsonl")
    ap.add_argument("--nogt", default="data/labeled/math_nogt.jsonl")
    ap.add_argument("--out", default="results/diagnostics/label_agreement.json")
    args = ap.parse_args()

    P, N = _load(args.priv), _load(args.nogt)
    if len(P) != len(N):
        print(f"WARNING: row count mismatch priv={len(P)} nogt={len(N)} — aligning by index up to min.")
    n_rows = min(len(P), len(N))

    steps = agree = both_err = neither = priv_only = nogt_only = 0
    parse_fail = 0
    score_gap_sum = 0.0
    flipped_solutions = 0
    for p, q in zip(P[:n_rows], N[:n_rows]):
        ps, qs = p.get("steps", []), q.get("steps", [])
        sol_flipped = False
        for a, bb in zip(ps, qs):
            ae, be = a.get("is_error"), bb.get("is_error")
            if a.get("parse_failed") or bb.get("parse_failed") or ae is None or be is None:
                parse_fail += 1
                continue
            steps += 1
            if a.get("score") is not None and bb.get("score") is not None:
                score_gap_sum += abs(float(a["score"]) - float(bb["score"]))
            if ae == be:
                agree += 1
                both_err += (ae is True)
                neither += (ae is False)
            else:
                sol_flipped = True
                if ae and not be:
                    priv_only += 1
                else:
                    nogt_only += 1
        flipped_solutions += sol_flipped

    s = max(1, steps)
    res = {
        "n_solutions": n_rows,
        "n_steps_scored": steps,
        "parse_failures_skipped": parse_fail,
        "step_agreement_rate": round(agree / s, 4),
        "disagree_priv_flags_error_only": priv_only,
        "disagree_nogt_flags_error_only": nogt_only,
        "both_flag_error": both_err,
        "neither_flags_error": neither,
        "mean_abs_score_gap": round(score_gap_sum / s, 4),
        "solutions_with_any_disagreement": flipped_solutions,
        "frac_solutions_with_disagreement": round(flipped_solutions / max(1, n_rows), 4),
    }
    print(json.dumps(res, indent=2))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")
    if res["step_agreement_rate"] > 0.95:
        print("⚠️ priv and no-GT labels agree on >95% of steps — the null may simply be that "
              "privilege barely changed the training targets. That's the explanation, not a bug.")


if __name__ == "__main__":
    main()
