# Handoff — Henry (research / analysis + paper)

_Updated 2026-06-16. Async-friendly, no compute needed. Paper: https://www.overleaf.com/2555239245xpdcmsxkrzgx · Dashboard: https://feedback-distillation.exe.xyz_

Your Overleaf draft is already strong (Related Work, the position, and the GSM8K/MATH/cross-family results are in). This runbook covers what's **changed** since you drafted it, and what's left to land.

## The spine — now sharper (and one thing to FIX)

The finding is a **tractability sweet spot**, not a monotonic difficulty effect:

> Privileged supervision helps the teacher **only where it both NEEDS the reference (can't self-verify) and can USE it (problem is tractable)**. Verified Δ solution-gap F1:
> - **GSM8K (easy): ≈ 0** — strong teacher self-verifies; privilege redundant.
> - **MATH (hard): +0.05** (N=400, **95% CI [0.01, 0.09] — significant**; error recall 0.61→0.66) — the sweet spot.
> - **OlympiadBench (hardest): ≈ 0** (−0.03, OE-only verified) — teacher can't use an olympiad-level reference.
> Plus **richness matters**: a bare answer is inert; only the full worked solution helps. Cross-family **confirmed** (Qwen-27B, +0.082 on MATH).

> **Headline number to cite: +0.05, 95% CI [0.01, 0.09] (N=400, significant).** The earlier N=150 +0.07 was underpowered (CI [−0.0005, 0.144] barely included zero) — supersede it everywhere.

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

**B. Results narrative** — update to the **3-tier sweet spot** (GSM8K ≈0 → MATH +0.05 [CI 0.01, 0.09] → OlympiadBench ≈0) + the richness panel + cross-family. The sweet-spot contrast (not the monotonic claim) is the load-bearing figure.

**C. Add the downstream-verifier section (new — §2.6).** The frontier/impact angle, not in the draft yet: the GT-free student PRM used as a test-time verifier (best-of-N re-rank) → final-answer accuracy vs majority vote. Numbers come from Saksham's Phase 3 (`bon_rerank.py`); a stub is fine for now.

**D. Interpret the evidence pack (DELIVERED — N=400, `results/evidence_pack_n400/` + `results/mechanism/`).** Defensible numbers, ready to drop into the paper:
- **Overall MATH gap: +0.051, bootstrap 95% CI [0.010, 0.093]** (paired over problems) — significant. This is the headline.
- **By-MATH-level (the sweet spot replicates WITHIN MATH):** L1 **−0.13** (n32, too easy) → L2 +0.03 (n60) → **L3 +0.11 (peak, n91)** → L4 +0.05 (n101) → L5 +0.04 (n116). Peaks at intermediate levels, collapses at L1 — mirrors the cross-dataset GSM8K→MATH→OlympiadBench curve. **This is your cleanest paper figure.**
- **Mechanism (`mechanism_analysis.py`) — rescue-of-self-verification-failures, gated by tractability:**
  - *Gate 1 (needs help):* of 227 error problems, no-GT **missed 89** → +solution **rescues 29 (33%)**; no-GT **hit 138** → +solution **breaks only 17 (12%)**. The gain concentrates where the teacher can't self-verify.
  - *Gate 2 (can use):* rescue rate falls monotonically with reference length — short **0.37** → mid **0.33** → long **0.28**. The reference helps only when followable. Moderator = self-verification-failure × tractability.
- **Flip examples** — 29 saved in `results/evidence_pack_n400/per_sample.jsonl` (`flip`/`broke` flags); categorize error types, pick 2–3 for the paper.

## Inputs you have
- Paper (Overleaf): https://www.overleaf.com/2555239245xpdcmsxkrzgx
- Dashboard (live, "Path to submission" shows the full runway): https://feedback-distillation.exe.xyz
- Result JSONs: `results/teacher_eval*/privilege_probe.json` (GSM8K, MATH N=50/150, Qwen-27B, OlympiadBench OE)
- Eval data: `data/processbench_math_shuffled.jsonl`
- Evidence pack + CIs: **delivered** — `results/evidence_pack_n400/{evidence_pack.json,per_sample.jsonl}` (overall gap + CI + by-level + flips) and `results/mechanism/mechanism.json` (gate1/gate2). Regenerate with `python -m experiments.evidence_pack` then `python -m experiments.mechanism_analysis`.

## Deliverable
The Overleaf draft, updated to the sweet-spot framing, with Related Work + the 3-tier results + the downstream-verifier section. Target venue flexible (COLM/AAAI/workshop) — quality over the date.
