"""Same-pool paired Best-of-N: re-rank ONE shared candidate set with BOTH the
privileged and no-GT student PRMs, and compare them head-to-head.

Why this exists: the first Phase-3 run scored `priv` and `nogt` on *separate*
candidate generations (`results/bon_priv` vs `results/bon_nogt`), so the
comparison wasn't apples-to-apples. Here we generate the pool once, score every
candidate with both verifiers, and report:
  - shared baselines: pass@1, majority_vote, oracle_pass@N (identical for both)
  - prm_rerank_priv vs prm_rerank_nogt on the SAME candidates
  - a PAIRED McNemar exact test on the two re-rankers' per-problem correctness

    OMLX_MODEL=<generator> python -m experiments.bon_paired \
        --dataset data/processbench_math_shuffled.jsonl \
        --priv checkpoints/priv_critique.pt --nogt checkpoints/nogt_critique.pt \
        --n 8 --max_samples 1000

NOTE: final-answer matching uses the simple matcher from generate_solutions;
swap in a symbolic MATH checker for publication-grade accuracy (shared TODO).
"""
import os
import json
import math
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor

import torch

from scripts.generate_solutions import extract_final_answer, answers_match, GEN_PROMPT, make_generator
from experiments.bon_rerank import solution_score
from models.student import StudentModel


def _load_student(checkpoint, dev_mode):
    s = StudentModel(dev_mode=dev_mode)
    c = torch.load(checkpoint, map_location="cpu")
    s.model.load_state_dict(c["model"], strict=False)
    s.score_head.load_state_dict(c["score_head"])
    return s


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on discordant pairs, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--priv", required=True, help="Privileged student PRM checkpoint.")
    ap.add_argument("--nogt", required=True, help="No-GT student PRM checkpoint.")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--agg", choices=["min", "mean"], default="min")
    ap.add_argument("--backend", choices=["omlx", "local"], default="omlx")
    ap.add_argument("--omlx_url", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--results_dir", default="results/bon_paired")
    ap.add_argument("--dev_mode", action="store_true")
    args = ap.parse_args()

    generate = make_generator(args.backend, args.omlx_url, args.dev_mode, args.temperature)
    priv = _load_student(args.priv, args.dev_mode)
    nogt = _load_student(args.nogt, args.dev_mode)
    print(f"Loaded both PRMs: priv={args.priv}  nogt={args.nogt}")

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rows = [r for r in rows if r.get("gt_answer")][: args.max_samples]
    print(f"{len(rows)} problems; N={args.n}, agg={args.agg} — SAME pool scored by both verifiers")

    n_p1 = n_maj = n_oracle = total = 0
    n_priv = n_nogt = 0
    b = c = 0  # discordant pairs: b = priv right & nogt wrong; c = priv wrong & nogt right
    for r in rows:
        problem, gt = r["problem"], str(r["gt_answer"])
        prompt = GEN_PROMPT.format(problem=problem)
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            cands = [f.result().strip() for f in [pool.submit(generate, prompt) for _ in range(args.n)]]
        cands = [t for t in cands if t]
        if not cands:
            continue
        total += 1
        answers = [extract_final_answer(t) for t in cands]
        correct = [answers_match(a, gt) for a in answers]

        n_p1 += correct[0]
        votes = collections.Counter(a for a in answers if a)
        n_maj += (answers_match(votes.most_common(1)[0][0], gt) if votes else 0)
        n_oracle += any(correct)

        # both verifiers pick from the SAME candidate list
        sp = [solution_score(priv, problem, t, args.agg) for t in cands]
        sn = [solution_score(nogt, problem, t, args.agg) for t in cands]
        cp = correct[max(range(len(cands)), key=lambda i: sp[i])]
        cn = correct[max(range(len(cands)), key=lambda i: sn[i])]
        n_priv += cp
        n_nogt += cn
        if cp and not cn:
            b += 1
        elif cn and not cp:
            c += 1

    t = max(1, total)
    res = {
        "n_problems": total, "N": args.n, "agg": args.agg, "shared_pool": True,
        "pass@1": round(n_p1 / t, 4),
        "majority_vote": round(n_maj / t, 4),
        "oracle_pass@N": round(n_oracle / t, 4),
        "prm_rerank_priv": round(n_priv / t, 4),
        "prm_rerank_nogt": round(n_nogt / t, 4),
        "paired_discordant_priv_only": b,
        "paired_discordant_nogt_only": c,
        "mcnemar_p_two_sided": round(_mcnemar_exact_p(b, c), 4),
    }
    print(json.dumps(res, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    json.dump(res, open(f"{args.results_dir}/bon_paired_results.json", "w"), indent=2)
    print(f"Saved -> {args.results_dir}/bon_paired_results.json")
    verdict = ("priv > nogt" if n_priv > n_nogt else "nogt >= priv")
    print(f"Head-to-head on shared pool: {verdict} "
          f"(priv {res['prm_rerank_priv']} vs nogt {res['prm_rerank_nogt']}, "
          f"McNemar p={res['mcnemar_p_two_sided']}). Transfer is real only if priv > nogt AND p<0.05.")


if __name__ == "__main__":
    main()
