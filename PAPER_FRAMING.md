# PAPER_FRAMING.md — canonical framing for the COLM 2026 submission

**This file is doctrine.** Every human and every agent writing paper text, related
work, Slack summaries, reviews, or rebuttals for this project follows the framing
below. If you (or your agent) think a different framing is better, PR this file
first — don't fork the narrative in a draft. Last updated: 2026-07-05 (frontier
scan). Owner: Edward.

## The one-sentence pitch

> A cheap, format-matched supervision check tells you whether your generated
> process labels can support the verifier you want — before you pay for a bigger
> student or a training-recipe search.

The paper is an **efficient-reasoning paper about training-data supervision**,
not a PRM-SOTA paper and not a distillation-method paper.

## The three claims, in load-bearing order

1. **Teacher sweet spot (§4, the validated spine):** privileged (answer-aware)
   step-error localization helps a teacher only where it both *needs* the
   reference (can't self-verify) and can *use* it (reference short enough).
   Inverted U across GSM8K/MATH/OlympiadBench, replicated within MATH levels and
   across model families.
2. **Verified transfer null (§5):** that advantage does NOT distill into a small
   answer-blind student — with a positive control proving the pipeline detects
   teacher-quality differences, so the null is not pipeline insensitivity.
3. **Mismatch diagnosis (§6):** format-matched gold process labels train a
   competent verifier under the identical path; generated teacher labels don't,
   at 1.5B and 3B. Supervision distribution/format, not student capacity.

## Terminology rule — REQUIRED differentiation (added 2026-07-05)

In 2026, "privileged teacher" is a loaded term: the OPSD / on-policy
self-distillation family (OPSD arXiv:2601.18734; position-weighted OPSD
arXiv:2605.21606; HDPO arXiv:2603.23871) uses answer-conditioned privileged
teachers for *successful* policy distillation. **Our result does not contradict
that literature and must never read as if it does.** The differentiation
paragraph (drop-in, keep the substance if you rewrite):

> Recent on-policy self-distillation methods condition a teacher on the
> ground-truth solution and distill its token-level guidance into the same
> model's policy, with strong gains (OPSD; position-weighted OPSD). Our setting
> differs on both axes that matter: we distill privileged *step-error labels*,
> off-policy, into a separate answer-blind *verifier*. Our diagnosis — that
> off-policy, format-mismatched supervision fails to transfer — is consistent
> with the mechanism that motivates the on-policy line (cf. Rethinking On-Policy
> Distillation, arXiv:2604.13016), observed here from the verifier side.

## Must-cite additions (frontier scan 2026-07-05)

| Paper | Why |
|---|---|
| OPSD (arXiv:2601.18734) | privileged-teacher terminology collision; differentiate |
| Position-weighted OPSD (arXiv:2605.21606) | nearest 2026 "where do teacher signals fail" relative |
| Noise-aware PRM training (arXiv:2601.12748) | pre-empt "would denoising rescue it?" — our gold-matched control isolates distribution, not noise; say this explicitly |

Nice-to-cite (one line each, cut first under length pressure): Scan
(arXiv:2509.16548), uPRM (arXiv:2605.10158), Strong Teacher Not Needed
(arXiv:2605.23857), Adaptive Generate-Rank-Verify (arXiv:2605.17609), Trust but
Verify survey (arXiv:2508.16665).

## Framing rules

1. **Lead efficiency, not apology.** The diagnostic costs ~0.3 GPU-hours vs ~15
   GPU-hours for the recipe search it replaces — this belongs in the abstract
   and intro, not just the conclusion. CFP topic 1 is literally "pipelines for
   creating high-quality training data under resource constraints."
2. **Sweet spot = verification-budget allocation rule.** Spend privileged
   labeling only inside the tractability band; redundant below, wasted above.
3. **The null is a *verified negative*, stated with pride** — positive control +
   CI + diagnosis. Never hedge it into vagueness; never oversell it as "privilege
   can't work" (scope: this student scale, this label format, off-policy).
4. **Never claim SOTA-PRM anything.** Qwen2.5-Math-7B-PRM800K is an external
   calibration point, full stop.
5. **Numbers freeze discipline:** table numbers come from committed artifacts in
   `results/` — cite the artifact path in PRs that change any paper number. The
   generated-label rows are 4-seed aggregates (priv BCE 0.5475, no-GT rank
   0.6062) + the same-source GSM8K control (generated priv BCE 0.5494,
   generated no-GT BCE 0.6183, GSM8K gold 0.7515). The run-to-run variance
   footnote (seed-0 rerun 0.477 vs canonical 0.550, both degenerate) is still
   useful as §6.4 instability evidence.

## Experiments the frontier now expects (priority order)

1. **Same-source GSM8K cell** — DONE 2026-07-05. Labels ran on Edward's Mac;
   training ran on Saksham's cluster. Result: generated labels remain weak when
   source problems are held fixed (priv 0.5494, no-GT 0.6183 vs GSM8K gold
   0.7515 ROC-AUC). This rules out source distribution as the sole explanation
   for GSM8K, but provenance and generated-label format/semantics remain
   coupled.
2. **Format-rerender cell** — re-render generated teacher labels into
   ProcessBench-style format, train once: completes the provenance×format 2×2.
   If format rescues → diagnosis sharpens to "format"; if not → "distribution."
3. **Gold-3B downstream best-of-N vs majority vote** + the BoN-vs-N figure
   (accuracy vs N, gold-trained vs teacher-trained verifier) — makes the paper
   legible as efficient-reasoning.
4. PRMBench companion numbers (arXiv:2501.03124) — first thing to cut.

## Where things live

- Combined paper: Overleaf (link in dashboard + Slack). `paper/main.tex` in-repo
  is Saksham's standalone COLM-styled draft — reuse its style block for the port.
- Deadline: **2026-07-19 AoE** (COLM 2026 Workshop on Efficient Reasoning;
  non-archival; double-blind — check anonymization on every artifact).
- Dashboard: `scripts/build_dashboard.py` → `dashboard/index.html` (CI-rebuilt);
  frontier/reading-list section renders from `FRONTIER` in that file.
