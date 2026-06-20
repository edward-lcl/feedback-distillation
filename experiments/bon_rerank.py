"""Best-of-N re-ranking — the GT-free student PRM as a test-time verifier.

This measures what a process reward model is actually *for*: improving a
reasoning model's answers at inference, cheaply, without ground truth.

Pipeline: for each problem, sample N candidate solutions from a generator
(the "policy"), score each with the student PRM (aggregate its per-step
`score_step` outputs), pick the highest-scoring one, and compare final-answer
accuracy against:
  pass@1         — take the first sample (no verifier)
  majority_vote  — self-consistency over the N answers
  prm_rerank     — pick the PRM's top-scored candidate  (our verifier)
  oracle_pass@N  — any of the N is correct (the ceiling)

If prm_rerank > majority_vote, the distilled GT-free verifier adds value.

    OMLX_MODEL=<generator> python -m experiments.bon_rerank \
        --dataset data/processbench_math_shuffled.jsonl \
        --checkpoint checkpoints/priv_critique.pt --n 8 --max_samples 100

NOTE: final-answer matching uses `math_verify` (symbolic) via answers_match,
with a numeric/string fallback.
"""
import os
import json
import argparse
import collections

import torch

from scripts.generate_solutions import extract_final_answer, answers_match, GEN_PROMPT, make_generator
from data.step_segmentation import segment_steps
from models.student import StudentModel


def solution_score(student, problem: str, solution: str, agg: str = "min") -> float:
    """Aggregate the PRM's per-step scores into one solution score.
    min = the weakest step (Math-Shepherd-style); mean = average confidence."""
    steps = segment_steps(solution)
    if not steps:
        return -1e9
    prefix, scores = "", []
    for s in steps:
        with torch.no_grad():
            scores.append(float(student.score_step(problem, prefix, s).item()))
        prefix += s + "\n"
    return min(scores) if agg == "min" else sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSONL with problem + gt_answer.")
    ap.add_argument("--checkpoint", default=None,
                    help="Student PRM checkpoint; untrained PRM if omitted (smoke only).")
    ap.add_argument("--student_model", default=None,
                    help="Student base model (must match the checkpoint's base, e.g. for the capacity sweep).")
    ap.add_argument("--n", type=int, default=8, help="Candidates sampled per problem.")
    ap.add_argument("--agg", choices=["min", "mean"], default="min")
    ap.add_argument("--backend", choices=["omlx", "local"], default="omlx")
    ap.add_argument("--omlx_url", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--results_dir", default="results/bon")
    ap.add_argument("--dev_mode", action="store_true")
    args = ap.parse_args()

    generate = make_generator(args.backend, args.omlx_url, args.dev_mode, args.temperature)
    student = StudentModel(args.student_model, dev_mode=args.dev_mode)
    if args.checkpoint:
        c = torch.load(args.checkpoint, map_location="cpu")
        student.model.load_state_dict(c["model"], strict=False)
        student.score_head.load_state_dict(c["score_head"])
        print(f"Loaded PRM checkpoint: {args.checkpoint}")
    else:
        print("WARNING: no checkpoint — untrained PRM (re-ranking is meaningless; smoke only).")

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rows = [r for r in rows if r.get("gt_answer")][: args.max_samples]
    print(f"{len(rows)} problems with gt_answer; N={args.n}, agg={args.agg}")

    n_p1 = n_maj = n_prm = n_oracle = total = 0
    for r in rows:
        problem, gt = r["problem"], str(r["gt_answer"])
        from concurrent.futures import ThreadPoolExecutor
        prompt = GEN_PROMPT.format(problem=problem)
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            futures = [pool.submit(generate, prompt) for _ in range(args.n)]
            cands = [f.result().strip() for f in futures]
        cands = [t for t in cands if t]
        if not cands:
            continue
        total += 1
        answers = [extract_final_answer(t) for t in cands]
        correct = [answers_match(a, gt) for a in answers]

        n_p1 += correct[0]
        votes = collections.Counter(a for a in answers if a)
        n_maj += answers_match(votes.most_common(1)[0][0], gt) if votes else 0
        scores = [solution_score(student, problem, t, args.agg) for t in cands]
        n_prm += correct[max(range(len(cands)), key=lambda i: scores[i])]
        n_oracle += any(correct)

    res = {
        "n_problems": total, "N": args.n, "agg": args.agg,
        "pass@1": round(n_p1 / max(1, total), 4),
        "majority_vote": round(n_maj / max(1, total), 4),
        "prm_rerank": round(n_prm / max(1, total), 4),
        "oracle_pass@N": round(n_oracle / max(1, total), 4),
    }
    print(json.dumps(res, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    json.dump(res, open(f"{args.results_dir}/bon_results.json", "w"), indent=2)
    print(f"Saved -> {args.results_dir}/bon_results.json")
    print("Verifier adds value iff prm_rerank > majority_vote (ceiling = oracle_pass@N).")


if __name__ == "__main__":
    main()
