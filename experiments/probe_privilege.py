"""Privilege probe: does *richer* privilege restore the privileged gap?

The Phase-0 gate found the bare GT *answer* buys ~0 over no-GT on GSM8K (a strong
teacher self-verifies easy arithmetic). This probe adds a third condition — the
full worked *reference solution* as privilege — and runs all three through the
same first-error metric (experiments.eval_teacher.evaluate_condition) so the F1s
are directly comparable.

    no_gt          : teacher sees problem + steps only
    with_answer    : + the final numeric answer        (thin privilege)
    with_solution  : + the full worked reference soln   (rich privilege)

A positive gap_solution with a ~0 gap_answer means the SLFD premise holds — the
privilege just has to be substantial.

Usage:
    OMLX_MODEL=gemma-4-26b-a4b-it-MLX-4bit .venv/bin/python -m experiments.probe_privilege \
        --dataset data/processbench_gsm8k_shuffled.jsonl --max_samples 50
"""
import os
import re
import json
import argparse

from experiments.eval_teacher import evaluate_condition
from data.label_pipeline import label_step_omlx
from models.omlx_client import OmlxClient


def _extract_solution(answer: str) -> str:
    """GSM8K answer field -> worked solution with the '#### N' line removed."""
    return re.sub(r"\n?####.*$", "", answer).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processbench_gsm8k_shuffled.jsonl")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--omlx_url", default=None)
    ap.add_argument("--results_dir", default="results/teacher_eval")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.dataset) if l.strip()][: args.max_samples]

    # GT answer + worked solution are embedded by scripts.download_data (gsm8k
    # and math configs both carry gt_answer + gt_solution). No re-join needed.
    have_ans = sum(1 for s in samples if s.get("gt_answer"))
    have_sol = sum(1 for s in samples if s.get("gt_solution"))
    print(f"GT present: answer {have_ans}/{len(samples)} | solution {have_sol}/{len(samples)}")
    if have_sol < len(samples):
        print("WARNING: some problems lack gt_solution — those fall back to no-GT in the "
              "solution condition, biasing gap_solution downward.")

    client = OmlxClient(api_url=args.omlx_url)
    try:
        client.list_models()
    except Exception as e:
        raise SystemExit(f"Cannot reach oMLX: {e}")
    print("teacher model:", client.model)

    sol_by_problem = {s["problem"]: (s.get("gt_solution") or None) for s in samples}
    lab_nogt = lambda p, pre, st, gt: label_step_omlx(p, pre, st, None, client=client)
    lab_ans = lambda p, pre, st, gt: label_step_omlx(p, pre, st, gt, client=client)
    lab_sol = lambda p, pre, st, gt: label_step_omlx(
        p, pre, st, None, client=client, gt_solution=sol_by_problem.get(p))

    res = {}
    res["no_gt"] = evaluate_condition(lab_nogt, samples, False, "no_gt")
    res["with_answer"] = evaluate_condition(lab_ans, samples, True, "with_answer")
    res["with_solution"] = evaluate_condition(lab_sol, samples, True, "with_solution")
    res["gap_answer_f1"] = round(res["with_answer"]["f1"] - res["no_gt"]["f1"], 4)
    res["gap_solution_f1"] = round(res["with_solution"]["f1"] - res["no_gt"]["f1"], 4)

    print(json.dumps(res, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "privilege_probe.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
