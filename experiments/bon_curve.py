"""Best-of-N vs N curve from a scored candidate pool (no model loading).

Consumes the JSONL written by `experiments.bon_paired --scores_file` (rows:
{problem, gt_answer, candidates, answers, correct, scores: {name: [float]}})
or any pool with a flat "scores": [float] list for a single verifier. For each
N in --ns it takes the FIRST N candidates (generation order, so subsets are
nested) and reports majority_vote@N, oracle pass@N, and prm_rerank@N per
verifier. pass@1 is the N=1 column.

    python -m experiments.bon_curve \
        --scores_file results/bon_paired_regrade/scored_pool.jsonl \
        --ns 1 2 4 8 --out results/bon_paired_regrade/bon_curve.json

Emits JSON plus a pgfplots-ready coordinate block per series on stdout.
"""
import json
import argparse
import collections

from scripts.generate_solutions import answers_match


def curve_point(rows: list[dict], n: int, verifiers: list[str]) -> dict:
    n_maj = n_oracle = total = 0
    n_rr = {v: 0 for v in verifiers}
    for r in rows:
        cands = r["candidates"][:n]
        if not cands:
            continue
        total += 1
        answers = r["answers"][:n]
        correct = r["correct"][:n]
        votes = collections.Counter(a for a in answers if a)
        n_maj += (answers_match(votes.most_common(1)[0][0], str(r["gt_answer"]))
                  if votes else 0)
        n_oracle += any(correct)
        for v in verifiers:
            scores = (r["scores"][v] if isinstance(r["scores"], dict)
                      else r["scores"])[:n]
            n_rr[v] += correct[max(range(len(cands)), key=lambda i: scores[i])]
    t = max(1, total)
    point = {
        "N": n,
        "n_problems": total,
        "majority_vote": round(n_maj / t, 4),
        "oracle_pass@N": round(n_oracle / t, 4),
    }
    for v in verifiers:
        point[f"prm_rerank_{v}"] = round(n_rr[v] / t, 4)
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_file", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scores_file) if l.strip()]
    if isinstance(rows[0]["scores"], dict):
        verifiers = sorted(rows[0]["scores"])
    else:
        verifiers = ["prm"]
        for r in rows:
            r["scores"] = {"prm": r["scores"]}
    max_n = min(len(r["candidates"]) for r in rows)
    ns = [n for n in sorted(set(args.ns)) if n <= max_n]
    if set(args.ns) - set(ns):
        print(f"NOTE: dropped N values above the pool size ({max_n}): "
              f"{sorted(set(args.ns) - set(ns))}")

    points = [curve_point(rows, n, verifiers) for n in ns]
    result = {"scores_file": args.scores_file, "verifiers": verifiers,
              "points": points}
    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"Saved -> {args.out}")

    for series in (["majority_vote", "oracle_pass@N"]
                   + [f"prm_rerank_{v}" for v in verifiers]):
        coords = " ".join(f"({p['N']},{p[series]})" for p in points)
        print(f"% pgfplots {series}: {coords}")


if __name__ == "__main__":
    main()
