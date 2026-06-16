# Handoff — Henry (research / analysis + paper)

_Updated 2026-06-16. Async-friendly, no compute needed. Paper: https://www.overleaf.com/2555239245xpdcmsxkrzgx · Dashboard: https://feedback-distillation.exe.xyz_

Your Overleaf draft is already strong (Related Work, the position, and the GSM8K/MATH/cross-family results are in). This runbook covers what's **changed** since you drafted it, and what's left to land.

## The spine — now sharper (and one thing to FIX)

The finding is a **tractability sweet spot**, not a monotonic difficulty effect:

> Privileged supervision helps the teacher **only where it both NEEDS the reference (can't self-verify) and can USE it (problem is tractable)**. Verified Δ solution-gap F1:
> - **GSM8K (easy): ≈ 0** — strong teacher self-verifies; privilege redundant.
> - **MATH (hard): +0.07** (N=150; error recall 0.58→0.65) — the sweet spot.
> - **OlympiadBench (hardest): ≈ 0** (−0.03, OE-only verified) — teacher can't use an olympiad-level reference.
> Plus **richness matters**: a bare answer is inert; only the full worked solution helps. Cross-family **confirmed** (Qwen-27B, +0.082 on MATH).

**FIX in your draft:** §2.3 currently says *"on hard problems it opens. Difficulty is the moderator."* — that's the old monotonic reading and our OlympiadBench result **contradicts it**. Reframe to the sweet spot. (Edward staged the OlympiadBench LaTeX block + the exact §1.4 / §2.3 edits.)

## What's settled — do NOT relitigate
| | choice |
|---|---|
| Teacher | Gemma-4-26b class (official ckpt for the reported run) |
| Dataset | ProcessBench **MATH** primary; GSM8K + OlympiadBench as the difficulty tiers |
| Privilege signal | **full worked solution** (a bare answer is inert) |
| Eval | ProcessBench-style first-error F1, GT-free at student test time |

## Your mission (in priority order)
**A. Related work & positioning** — already drafted; keep. Novelty = (1) score+critique as a distillation target, (2) the privilege **sweet-spot** characterization (when privilege can be distilled at all).

**B. Results narrative** — update to the **3-tier sweet spot** (GSM8K ≈0 → MATH +0.07 → OlympiadBench ≈0) + the richness panel + cross-family. The sweet-spot contrast (not the monotonic claim) is the load-bearing figure.

**C. Add the downstream-verifier section (new — §2.6).** The frontier/impact angle, not in the draft yet: the GT-free student PRM used as a test-time verifier (best-of-N re-rank) → final-answer accuracy vs majority vote. Numbers come from Saksham's Phase 3 (`bon_rerank.py`); a stub is fine for now.

**D. Interpret the evidence pack (incoming).** Edward is generating, from a per-sample probe log:
- **By-tier / by-level breakdown** — note the prediction has CHANGED: not "gap widens with level," but a sweet spot (rises GSM8K→MATH, falls at OlympiadBench).
- **Flip examples** — concrete steps the teacher catches only with the solution; categorize error types, pick 2–3 for the paper.
- **Bootstrap CIs** on every gap (so +0.07 ships with a confidence interval).

## Inputs you have
- Paper (Overleaf): https://www.overleaf.com/2555239245xpdcmsxkrzgx
- Dashboard (live, "Path to submission" shows the full runway): https://feedback-distillation.exe.xyz
- Result JSONs: `results/teacher_eval*/privilege_probe.json` (GSM8K, MATH N=50/150, Qwen-27B, OlympiadBench OE)
- Eval data: `data/processbench_math_shuffled.jsonl`
- Evidence pack + CIs: _being delivered by Edward (per-sample log → by-tier + flips + CIs)._

## Deliverable
The Overleaf draft, updated to the sweet-spot framing, with Related Work + the 3-tier results + the downstream-verifier section. Target venue flexible (COLM/AAAI/workshop) — quality over the date.
