# Handoff — Henry (research / analysis + paper)

_Updated 2026-06-16. Async-friendly, no compute needed. Paper: https://www.overleaf.com/2555239245xpdcmsxkrzgx · Dashboard: https://feedback-distillation.exe.xyz_

## 2026-06-30 — your draft is the spine of a merged paper now; here's the status

Quick catch-up since your last context (2026-06-18): while you were heads-down, Saksham
ran a second, independent track (`saksham/phaseb-capacity-gate-main-integration`, merged
into `main` 2026-07-01) on a **Qwen2.5-3B** verifier, asking a different question — is the
student-transfer null (your §2.7) a capacity problem or a supervision problem? He found
that under the *identical* training/eval path, ProcessBench-style gold labels (GSM8K,
OmniMath) train a *competent* verifier (0.73–0.79 ROC-AUC), while our generated teacher
labels stay weak (0.55–0.63) at that scale too — pointing at supervision
distribution/format mismatch, not raw capacity, as the reason your null doesn't distill.
He wrote this up as its own paper (`colm2026_prm_mismatch_overleaf__2_.pdf`, targeting the
**COLM Efficient Reasoning Workshop, deadline 2026-07-12**), with no idea your draft
existed in the form it's in now.

Separately, someone finally ran the **B3 positive control** your own §2.7 "Boundary of the
claim" paragraph named as a prerequisite (2026-06-22): a student trained on a deliberately
weak teacher scores significantly worse than one trained on Gemma-4 (+0.05 ROC-AUC, 95% CI
excluding zero). That closes the "is the pipeline just insensitive" gap you flagged.

**Verdict on the two drafts, for what it's worth: yours is the stronger paper** — it has
the one real, significant, mechanistically-explained positive result in the whole project
(the sweet spot). Saksham's is a strong, complementary diagnostic that happens to answer
exactly what your own draft said it needed. So rather than pick one, we merged them:
**`paper/merged_draft.tex`** (compiles clean via `tectonic`) keeps your Related Work and
§2.1–2.5 (sweet spot) essentially verbatim — just the two known typos fixed — leads with
it as the paper's spine exactly as you framed it, and extends your §2.6/§2.7 with the
positive control (finally in prose, not just a log entry) and a new section presenting
Saksham's mismatch result as the answer to your own "capacity or supervision?" question.
Plan doc with the full section-by-section mapping: `PAPER_MERGE_PLAN.md`.

**What's actually left for you:**
- ☐ **Table consolidation** — still not done. You flagged this yourself on 2026-06-16 and
  it's more relevant now, not less: the merged draft is up to 10 tables. Your call on
  which to fold (Tables 1+2 GSM8K/MATH conditions was your original suggestion; the new
  Phase B tables in §6 have the same "too many small tables" problem).
- ☐ **Prose/editorial pass across the merge seam** — I (Edward, with an agent) stitched
  the sections together mechanically; it needs a human read for tone and transitions,
  especially the "Our position" paragraph (now three contributions, not two) and the
  Conclusion.
- ☐ **Confirm the framing decisions** — sweet spot leads, null+positive-control is
  §5, mismatch diagnosis is §6, shared Limitations at the end. Push back if you'd
  structure it differently.
- ☐ **Venue** — this is now explicitly targeting COLM ER (2026-07-12), not the flexible
  "COLM/AAAI/workshop" framing from your original brief. Says something about how much
  editorial latitude there is on length/formatting; check the CFR before a final pass.
- ☐ **Getting it to you** — `paper/merged_draft.tex` and `PAPER_MERGE_PLAN.md` exist
  locally right now, not yet pushed anywhere you can see them. Ask Edward for the current
  location (git branch or Overleaf) before you start editing.

Everything below this point is your original context (2026-06-16 through 2026-06-18) —
still accurate for your section, kept for reference.

Your Overleaf draft is already strong (Related Work, the position, and the GSM8K/MATH/cross-family results are in). This runbook covers what's **changed** since you drafted it, and what's left to land.

## 2026-06-17 — status sync (your draft is in `main`)
> ⏩ **Superseded — see the `2026-06-18` block below.** §2.7 is no longer a deferred stub: the re-score is done and it's a **verified negative** (no transfer). Treat the `2026-06-18` block + mission C as current; this block is kept as history.

Your push (`61c0ec0`) is merged: the compiled 4-page draft (`paper/SLFD_draft.pdf`) now has **Related Work §1.1–1.4 + 3-tier sweet-spot Results (Table 2) + §2.7 downstream verifier**, and it compiles. Nicely done — and thank you for keeping **§2.7 honest** (prelim PRM re-rank 32.0 < majority 32.5, reported "only to fix the protocol," deferred pending Saksham's re-score). The §2.3 monotonic→sweet-spot FIX is in. So mission A/B are essentially done and C is correctly stubbed.

**Two fixes in the Overleaf source before anyone cites it:**
1. **§2.3 prose typo — `+0.5 F1` should be `+0.05`.** Table 2 is right ([0.01, 0.09]); the sentence reads 10× too big. This is the headline number, so it matters most.
2. **Spelling typos** in §1.1–1.2: `whcih`, `wcih` (→ "which"), `close to uors` (→ "ours"), `expert-amateur literate` (→ "literature").

**Remaining to land (from your own dashboard task row):**
- ☐ paste 2–3 flip cases from `results/evidence_pack_n400/per_sample.jsonl` (the `flip`/`broke` rows) into the Results.
- ☐ fill the OlympiadBench gap cell in Table 2 with the verified OE-only number (−0.03) rather than just "≈ 0 (n.s.)".
- ☑ §2.7 numbers are **in** — the re-score + N=1000 run are done (see below). §2.7 is now a **verified honest negative**, not a stub.
- ☐ **Consolidate the tables** (Edward's call). The draft has 6 tables and for our venue level that reads as over-split — agents tend to over-engineer this. Fold the per-condition results into one consolidated table (esp. Tables 3+4 GSM8K/MATH → a single conditions table; consider merging the cross-family panel too). One readable table beats five small ones.

**Green light to keep drafting now** — the preliminary results already paint the picture, so build out the full narrative and slot remaining numbers in as they land (per Edward). The Foerster guide above is the structure to follow.

**Worth a read before the next drafting pass:** Jakob Foerster's "How to ML Paper" — https://www.jakobfoerster.com/how-to-ml-paper — a tight guide on structuring the narrative (claim → evidence → ablation) that maps well onto where the draft is now.

## 2026-06-18 — paper now needs a reframe (verified student-transfer NULL)

I read the committed `paper/SLFD_draft.pdf`. Concrete tasks, in priority order:

1. **§2.7 — replace "we defer the result" with the verified negative.** The re-score + N=1000 run are done (numbers in `results/RESULTS.md`). Update Table 6 + prose to:
   - step-level `roc_auc`: **no-GT 0.641 ≥ priv 0.631 ≥ priv_scoreonly 0.624** (non-degenerate run);
   - downstream Best-of-N (N=1000): **no-GT re-rank 0.373 ≥ priv 0.349; neither beats majority vote (~0.39).**
   Conclusion: privileged supervision does **not** transfer into the small student PRM at this scale.
2. **Reframe the contribution (§1.4 / abstract / title framing).** As written, contribution #1 implies the student distillation *works*. It doesn't. Lead with the **teacher-level** privilege×difficulty×richness sweet spot (the validated result) as the headline; present the student-transfer **null + its diagnosis** (the diagnostic threads) as an honest secondary finding. Don't imply the distillation succeeds.
3. **Fix the headline typo — STILL PRESENT.** §2.3 prose reads "**+0.5 F1**" — must be **+0.05** (Table 2 is right at [0.01, 0.09]). This is the headline number.
4. **Spelling/grammar still in the draft:** `whcih`→which (§1.1), `uors`→ours + `priviledge`→privileged (§1.2), `literate`→literature + `GMS8K`→GSM8K (§1.4), `fails monotonically`→falls (§2.5), `is not longer usable`→no longer (§2.3), `challenge MATH`→challenging (§1.1).
5. **Consolidate tables** (you flagged this; agents over-split): 6 tables is too many for our level — fold Tables 3+4 (GSM8K/MATH conditions) into one conditions table, consider merging Table 5 (cross-family) too.
6. **Fill the OlympiadBench cell** in Table 2 with the verified OE-only number (−0.03), not just "≈ 0 (n.s.)".
7. **Paste 2–3 flip cases** from `results/evidence_pack_n400/per_sample.jsonl`.

(Note re submitting the 22nd workshop: the reframe in #1–2 is the gating item — the paper currently promises a transfer result it doesn't have.)

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

**C. Downstream-verifier section (§2.7) — ✅ VERIFIED NEGATIVE (2026-06-18). Write it as a clean null.** The N=1000 run is done and validated (labeling confirmed through the served Gemma-4 teacher). The result: **privilege does NOT transfer into the student PRM.**
- **Step-level (`roc_auc`, the right metric):** no-GT **0.641** ≥ priv 0.631 ≥ priv_scoreonly 0.624. Non-degenerate run (no silent collapse), so it's a clean comparison.
- **Downstream Best-of-N (N=1000):** no-GT verifier `prm_rerank` **0.373** ≥ priv 0.349; **neither beats majority vote** (~0.39).
- The earlier "0.197 vs 0.037 privilege transfers" was a **fixed-threshold F1 artifact** and does **not** reproduce; do not cite it.

**How to frame it:** the **teacher-level** privilege/sweet-spot result is the spine and stands. §2.7 reports the honest finding that this teacher advantage does **not** distill into a better small (1.5B) student PRM at this scale — no-GT is equal-or-better, and neither verifier beats majority vote. That's a legitimate, interesting negative. The live diagnosis threads make it a contribution rather than a dead end:
1. Gemma-4 privilege probe (confirm priv≠nogt labels for the labeling teacher);
2. same-pool paired Phase 3 (one shared candidate set + paired significance — `bon_priv`/`bon_nogt` were separate generations);
3. why no transfer — train/eval distribution shift, 1.5B capacity, or priv-vs-no-GT label agreement.
Full numbers + the open threads: `results/RESULTS.md`. Keep §2.7 honest; don't oversell either direction.

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
