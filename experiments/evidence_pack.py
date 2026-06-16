"""Evidence pack + bootstrap CIs for the privilege probe.

Runs the no-GT and +solution conditions on a dataset with PER-SAMPLE logging,
then computes what the paper needs for rigor + qualitative support:
  - F1 per condition and the solution gap, with a paired bootstrap 95% CI
  - by-MATH-level breakdown of the gap (does it peak mid-difficulty?)
  - flip examples: problems where +solution localizes the gold first error
    that the no-GT teacher misses (with the teacher's feedback at that step)

    OMLX_MODEL=<teacher> python -m experiments.evidence_pack \
        --dataset data/processbench_math_shuffled.jsonl --max_samples 150
"""
import os
import json
import random
import argparse

from tqdm import tqdm
from data.label_pipeline import label_step_omlx
from models.omlx_client import OmlxClient


def first_flagged(client, problem, steps, gt_solution):
    """Scan steps; return (first index flagged as error, feedback at that step)."""
    prefix = ""
    for j, s in enumerate(steps):
        lab = label_step_omlx(problem, prefix, s, None, client=client, gt_solution=gt_solution)
        if lab.get("is_error"):
            return j, lab.get("feedback", "")
        prefix += s + "\n"
    return -1, ""


def f1_from(records, key):
    nc = ne = ch = eh = 0
    for r in records:
        if r["gold"] == -1:
            nc += 1; ch += (r[key] == -1)
        else:
            ne += 1; eh += (r[key] == r["gold"])
    ca = ch / nc if nc else 0.0
    ea = eh / ne if ne else 0.0
    return (2 * ca * ea / (ca + ea)) if (ca + ea) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    ap.add_argument("--max_samples", type=int, default=150)
    ap.add_argument("--omlx_url", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--results_dir", default="results/evidence_pack")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.dataset) if l.strip()][: args.max_samples]

    # Best-effort MATH difficulty level join (Levels 1–5).
    level_map = {}
    try:
        from datasets import load_dataset
        for sp in ("train", "test"):
            for ex in load_dataset("nlile/hendrycks-MATH-benchmark", split=sp):
                p = (ex.get("problem") or "").strip()
                if p:
                    level_map[p] = ex.get("level")
    except Exception:
        pass

    client = OmlxClient(api_url=args.omlx_url)
    client.list_models()
    print("teacher:", client.model)

    records = []
    for s in tqdm(samples):
        prob = s["problem"]
        gold = s.get("first_error_label", -1)
        steps = [st["text"] for st in s["steps"]]
        sol = s.get("gt_solution") or None
        p_nogt, _ = first_flagged(client, prob, steps, None)
        p_sol, fb = first_flagged(client, prob, steps, sol)
        records.append({"gold": gold, "level": level_map.get(prob.strip()),
                        "pred_nogt": p_nogt, "pred_sol": p_sol,
                        "problem": prob[:240], "sol_feedback_at_pred": fb})

    f1_nogt, f1_sol = f1_from(records, "pred_nogt"), f1_from(records, "pred_sol")
    gap = f1_sol - f1_nogt

    # Paired bootstrap CI on the gap (resample problems with replacement).
    rng = random.Random(0)
    n = len(records)
    gaps = []
    for _ in range(args.boot):
        samp = [records[rng.randrange(n)] for _ in range(n)]
        gaps.append(f1_from(samp, "pred_sol") - f1_from(samp, "pred_nogt"))
    gaps.sort()
    lo, hi = gaps[int(0.025 * args.boot)], gaps[int(0.975 * args.boot)]

    by_level = {}
    for lv in sorted({str(r["level"]) for r in records if r["level"]}):
        sub = [r for r in records if str(r["level"]) == lv]
        by_level[lv] = {"n": len(sub),
                        "gap": round(f1_from(sub, "pred_sol") - f1_from(sub, "pred_nogt"), 4)}

    flips = [{"problem": r["problem"], "gold_step": r["gold"],
              "feedback": r["sol_feedback_at_pred"]}
             for r in records
             if r["gold"] >= 0 and r["pred_sol"] == r["gold"] and r["pred_nogt"] != r["gold"]]

    out = {"n": n, "f1_nogt": round(f1_nogt, 4), "f1_sol": round(f1_sol, 4),
           "gap": round(gap, 4), "gap_ci95": [round(lo, 4), round(hi, 4)],
           "by_level": by_level, "n_flips": len(flips), "flips": flips[:10]}
    print(json.dumps({k: out[k] for k in
                      ["n", "f1_nogt", "f1_sol", "gap", "gap_ci95", "by_level", "n_flips"]}, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    json.dump(out, open(f"{args.results_dir}/evidence_pack.json", "w"), indent=2)
    with open(f"{args.results_dir}/per_sample.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved -> {args.results_dir}/ (evidence_pack.json + per_sample.jsonl)")


if __name__ == "__main__":
    main()
