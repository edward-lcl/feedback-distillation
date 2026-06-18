# Handoff — Henry (research / analysis + paper)

_Updated 2026-06-16. Async-friendly, no compute needed. Paper: https://www.overleaf.com/2555239245xpdcmsxkrzgx · Dashboard: https://feedback-distillation.exe.xyz_

Your Overleaf draft is already strong (Related Work, the position, and the GSM8K/MATH/cross-family results are in). This runbook covers what's **changed** since you drafted it, and what's left to land.

## 2026-06-17 — status sync (your draft is in `main`)

Your push (`61c0ec0`) is merged: the compiled 4-page draft (`paper/SLFD_draft.pdf`) now has **Related Work §1.1–1.4 + 3-tier sweet-spot Results (Table 2) + §2.7 downstream verifier**, and it compiles. Nicely done — and thank you for keeping **§2.7 honest** (prelim PRM re-rank 32.0 < majority 32.5, reported "only to fix the protocol," deferred pending Saksham's re-score). The §2.3 monotonic→sweet-spot FIX is in. So mission A/B are essentially done and C is correctly stubbed.

**Two fixes in the Overleaf source before anyone cites it:**
1. **§2.3 prose typo — `+0.5 F1` should be `+0.05`.** Table 2 is right ([0.01, 0.09]); the sentence reads 10× too big. This is the headline number, so it matters most.
2. **Spelling typos** in §1.1–1.2: `whcih`, `wcih` (→ "which"), `close to uors` (→ "ours"), `expert-amateur literate` (→ "literature").

**Remaining to land (from your own dashboard task row):**
- ☐ paste 2–3 flip cases from `results/evidence_pack_n400/per_sample.jsonl` (the `flip`/`broke` rows) into the Results.
- ☐ fill the OlympiadBench gap cell in Table 2 with the verified OE-only number (−0.03) rather than just "≈ 0 (n.s.)".
- ☐ §2.7 real numbers — **gated on Saksham's threshold-free re-score** (ROC/PR-AUC) + symbolic checker + larger N. Keep it a stub until then; see HANDOFF_SAKSHAM.md "READ FIRST".
- ☐ **Consolidate the tables** (Edward's call). The draft has 6 tables and for our venue level that reads as over-split — agents tend to over-engineer this. Fold the per-condition results into one consolidated table (esp. Tables 3+4 GSM8K/MATH → a single conditions table; consider merging the cross-family panel too). One readable table beats five small ones.

**Green light to keep drafting now** — the preliminary results already paint the picture, so build out the full narrative and slot remaining numbers in as they land (per Edward). The Foerster guide above is the structure to follow.

**Worth a read before the next drafting pass:** Jakob Foerster's "How to ML Paper" — https://www.jakobfoerster.com/how-to-ml-paper — a tight guide on structuring the narrative (claim → evidence → ablation) that maps well onto where the draft is now.

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
**A. Related work & positioning — ✅ DONE (§1.1–1.4).** Novelty = (1) score+critique as a distillation target, (2) the privilege **sweet-spot** characterization (when privilege can be distilled at all).

**B. Results narrative — ✅ DONE (§2.3, Table 2).** Restructured to the **3-tier sweet spot** (GSM8K ≈0 → MATH +0.05 [CI 0.01, 0.09] → OlympiadBench ≈0) + the richness panel + cross-family. The sweet-spot contrast (not the monotonic claim) is the load-bearing figure. (Remaining: OlympiadBench cell number + flip cases — see sync block above.)

**C. Add the downstream-verifier section — ✅ STUBBED (now §2.7, not §2.6).** Honest negative as intended; real numbers gated on Saksham's re-score. The frontier/impact angle, not in the draft yet: the GT-free student PRM used as a test-time verifier (best-of-N re-rank) → final-answer accuracy vs majority vote. Numbers come from Saksham's Phase 3 (`bon_rerank.py`); a stub is fine for now. **⚠️ Don't write this as a positive result yet:** the first run had prm_rerank **32.0 < majority_vote 32.5** (N=200, within noise), AND the Phase 2 student PRM it relies on isn't validated — the reported "privilege transfers" F1 gap (0.197 vs 0.037) was a fixed-threshold artifact (nogt recall ≈2.6%; see HANDOFF_SAKSHAM.md "READ FIRST"). Both need the threshold-free re-score (ROC/PR-AUC) + symbolic checker + larger N before this section claims anything. Keep it a placeholder until then.

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
