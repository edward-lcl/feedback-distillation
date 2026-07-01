#!/usr/bin/env python3
"""Read-only monitor for Phase B capacity-gate runs.

This is intentionally non-invasive: it only reads the runner log and result
JSONs, then prints the current state plus a simple pivot/continue hint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


STEP_RE = re.compile(r"\bstep\s+(\d+):")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def latest_step(log_text: str) -> int | None:
    steps = [int(m.group(1)) for m in STEP_RE.finditer(log_text)]
    return max(steps) if steps else None


def current_phase(log_text: str) -> str:
    phases = [line.strip() for line in log_text.splitlines() if line.startswith("== ")]
    return phases[-1] if phases else "unknown"


def metric_line(name: str, data: dict | None) -> str:
    if not data:
        return f"{name}: pending"
    keys = ["roc_auc", "pr_auc", "f1", "pred_error_rate", "first_error_acc"]
    parts = []
    for key in keys:
        val = data.get(key)
        if val is not None:
            parts.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
    warnings = data.get("warnings") or []
    suffix = f" warnings={len(warnings)}" if warnings else ""
    return f"{name}: " + ", ".join(parts) + suffix


def bon_line(name: str, data: dict | None) -> str:
    if not data:
        return f"{name}: pending"
    return (
        f"{name}: MV={data['majority_vote']:.4f}, "
        f"priv={data['prm_rerank_priv']:.4f}, "
        f"nogt={data['prm_rerank_nogt']:.4f}, "
        f"p={data['mcnemar_p_two_sided']:.4f}"
    )


def recommendation(priv: dict | None, nogt: dict | None, quick_bon: dict | None,
                   nan_count: int, step: int | None) -> str:
    if quick_bon:
        best = max(quick_bon["prm_rerank_priv"], quick_bon["prm_rerank_nogt"])
        margin = best - quick_bon["majority_vote"]
        if margin > 0:
            return "CONTINUE: quick BoN clears majority vote; force full BoN and replicate seeds."
        if margin >= -0.01:
            return "CONTINUE SHORT: quick BoN is close to MV; try a longer cap before pivoting."
        return "PIVOT: quick BoN is clearly below MV; try more capacity/data or a training fix."

    if priv and nogt:
        best_auc = max(priv.get("roc_auc") or 0, nogt.get("roc_auc") or 0)
        if best_auc >= 0.63:
            return "CONTINUE: ProcessBench AUC is in the prior baseline range; run quick BoN next."
        if best_auc >= 0.58:
            return "CONTINUE SHORT: some signal exists; try a larger max_steps cap before BoN."
        return "PIVOT: AUC is weak after this cap; change capacity/training before spending BoN time."

    if step and nan_count / max(step, 1) > 0.25:
        return "WATCH: high non-finite-loss rate; consider BATCH_SIZE=1 or a scoring-only stability probe."
    return "WAIT: no eval result yet; keep watching until priv/nogt ProcessBench JSONs appear."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Run tag, e.g. Qwen_Qwen2.5-3B-Instruct_seed0_ms100")
    ap.add_argument("--log", default=None,
                    help="Runner log path. Defaults to phaseb_capacity_<tag>.log if present.")
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    root = Path(args.results_root)
    log_path = Path(args.log) if args.log else Path(f"phaseb_capacity_{args.tag}.log")
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    step = latest_step(log_text)
    nan_count = log_text.count("non-finite loss")

    priv_dir = root / "ablation" / f"{args.tag}_priv_critique"
    nogt_dir = root / "ablation" / f"{args.tag}_nogt_critique"
    ci_path = root / "ablation" / f"{args.tag}_transfer_ci.json"
    quick_dir = root / "bon_paired" / f"{args.tag}_quick_m200"
    full_dir = root / "bon_paired" / f"{args.tag}_full_m1000"

    priv = load_json(priv_dir / "processbench_results.json")
    nogt = load_json(nogt_dir / "processbench_results.json")
    ci = load_json(ci_path)
    quick_bon = load_json(quick_dir / "bon_paired_results.json")
    full_bon = load_json(full_dir / "bon_paired_results.json")

    print(f"tag: {args.tag}")
    print(f"log: {log_path if log_path.exists() else 'missing'}")
    print(f"phase: {current_phase(log_text)}")
    print(f"latest_step: {step if step is not None else 'pending'}")
    print(f"non_finite_loss_count: {nan_count}")
    print(metric_line("priv", priv))
    print(metric_line("nogt", nogt))
    if ci:
        print(
            "transfer_ci: "
            f"gap={ci['gap_model_a_minus_model_b']:.4f}, "
            f"ci95=[{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}], "
            f"p2={ci['p_two_sided']:.4f}"
        )
    else:
        print("transfer_ci: pending")
    print(bon_line("quick_bon", quick_bon))
    print(bon_line("full_bon", full_bon))
    print("recommendation:", recommendation(priv, nogt, quick_bon, nan_count, step))


if __name__ == "__main__":
    main()
