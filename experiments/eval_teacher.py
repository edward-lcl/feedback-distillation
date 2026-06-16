"""
Phase-0 gate: how good is the teacher's grounding, actually?

Runs the teacher on ProcessBench (gold first-error labels) in two conditions:
  with_gt  — privileged prompt (GT answer included): the DISTILLATION CEILING.
  no_gt    — same prompt minus the answer: measures the PRIVILEGED GAP.

Metric (official ProcessBench protocol):
  correct_acc = on all-correct solutions, fraction where teacher flags nothing
  error_acc   = on erroneous solutions, fraction where teacher's first flagged
                step == gold first-error index
  f1          = harmonic mean of the two

Decision rule: if with_gt F1 is weak, fix teacher prompting/grounding before any
training run. If (with_gt - no_gt) is small, the privileged framing needs work.

Usage:
    # Real teacher via local oMLX (set OMLX_API_KEY first):
    python -m experiments.eval_teacher \
        --dataset data/processbench_gsm8k.jsonl --backend omlx --max_samples 50

    # Local dev smoke (1.5B teacher on MPS):
    python -m experiments.eval_teacher \
        --dataset data/processbench_gsm8k.jsonl --backend local --dev_mode --max_samples 3
"""
import json
import argparse
from tqdm import tqdm


def _first_flagged(labeler, problem, steps, gt_answer) -> tuple[int, int]:
    """Scan steps in order; return (first index the teacher flags as error, parse_failures).
    -1 if no step flagged. Stops at the first flag (all the metric needs)."""
    prefix = ""
    parse_failures = 0
    for j, step_text in enumerate(steps):
        label = labeler(problem, prefix, step_text, gt_answer)
        if label.get("parse_failed"):
            parse_failures += 1
        elif label["is_error"]:
            return j, parse_failures
        prefix += step_text + "\n"
    return -1, parse_failures


def evaluate_condition(labeler, samples, gt_key: bool, desc: str) -> dict:
    correct_total = correct_hit = 0
    error_total = error_hit = 0
    parse_failures = total_steps_scanned = 0

    for s in tqdm(samples, desc=desc):
        gt = s.get("gt_answer", "") if gt_key else None
        steps = [st["text"] for st in s["steps"]]
        pred_first, pf = _first_flagged(labeler, s["problem"], steps, gt)
        parse_failures += pf
        total_steps_scanned += len(steps)
        gold = s.get("first_error_label", -1)
        if gold == -1:
            correct_total += 1
            correct_hit += (pred_first == -1)
        else:
            error_total += 1
            error_hit += (pred_first == gold)

    correct_acc = correct_hit / correct_total if correct_total else 0.0
    error_acc = error_hit / error_total if error_total else 0.0
    f1 = (2 * correct_acc * error_acc / (correct_acc + error_acc)
          if (correct_acc + error_acc) > 0 else 0.0)
    return {
        "f1": round(f1, 4),
        "correct_acc": round(correct_acc, 4),
        "error_acc": round(error_acc, 4),
        "n_correct_samples": correct_total,
        "n_error_samples": error_total,
        "parse_failure_rate": round(parse_failures / max(1, total_steps_scanned), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="ProcessBench JSONL from scripts.download_data (gsm8k config has gt_answer joined).")
    parser.add_argument("--backend", choices=["omlx", "local"], default="omlx")
    parser.add_argument("--omlx_url", default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--skip_no_gt", action="store_true",
                        help="Only measure the with-GT ceiling (halves the calls).")
    parser.add_argument("--results_dir", default="results/teacher_eval")
    args = parser.parse_args()

    import os
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f if l.strip()][: args.max_samples]
    with_gt_samples = [s for s in samples if s.get("gt_answer")]
    if not with_gt_samples:
        print("WARNING: no gt_answer in dataset — with-GT condition unavailable. "
              "Use the gsm8k ProcessBench config (GT is auto-joined there).")

    if args.backend == "omlx":
        from data.label_pipeline import label_step_omlx
        from models.omlx_client import OmlxClient
        client = OmlxClient(api_url=args.omlx_url)
        try:
            print(f"oMLX models available: {client.list_models()}")
        except Exception as e:
            raise SystemExit(f"Cannot reach oMLX server: {e}\n"
                             "Set OMLX_API_KEY (and OMLX_URL if not :8000).")
        labeler = lambda p, pre, st, gt: label_step_omlx(p, pre, st, gt, client=client)
    else:
        from models.teacher import TeacherModel
        teacher = TeacherModel(dev_mode=args.dev_mode)
        labeler = teacher.label_step

    results = {}
    if with_gt_samples:
        results["with_gt"] = evaluate_condition(labeler, with_gt_samples, True, "with_gt")
    if not args.skip_no_gt:
        results["no_gt"] = evaluate_condition(labeler, samples, False, "no_gt")
    if "with_gt" in results and "no_gt" in results:
        results["privileged_gap_f1"] = round(results["with_gt"]["f1"] - results["no_gt"]["f1"], 4)

    print(json.dumps(results, indent=2))
    out = os.path.join(args.results_dir, "teacher_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
