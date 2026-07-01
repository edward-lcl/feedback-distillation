# Paper merge plan — combining Henry's sweet-spot draft + Saksham's mismatch draft

Context: two independent paper drafts currently exist and neither author knows about
the other's.

- `paper/SLFD_draft.pdf` (Henry, Overleaf, last touched ~2026-06-20). Headline: the
  teacher-level privilege × difficulty × richness sweet spot (validated, mechanistic,
  cross-family). Ends with an explicit "Boundary of the claim" section naming what's
  needed to make §2.6/§2.7 (the 1.5B transfer null) a strong result rather than a flat
  null: (i) a strong-vs-weak-teacher positive control, (ii) something that resolves
  "does it fail to transfer because of privilege, or because the student/pipeline is
  just weak."
- `colm2026_prm_mismatch_overleaf__2_.pdf` (Saksham, local only, sent 2026-06-30).
  Headline: under a fixed Qwen2.5-3B verifier/training/eval path, ProcessBench-style
  gold labels transfer (0.75-0.85 ROC-AUC) while generated teacher labels don't
  (0.55-0.63 ROC-AUC). No mention of the teacher-level sweet spot at all.

**The two items Henry's draft asks for are both now done, just not written into either
draft:**
1. B3 positive control (weak teacher vs. Gemma-4 teacher, 1.5B student): student
   quality tracks teacher quality, +0.048 ROC-AUC gap, 95% CI [0.020, 0.077],
   significant. Currently only logged in `HANDOFF_SAKSHAM.md` (2026-06-22 entry) —
   not in RESULTS.md, not in either paper draft.
2. Saksham's Phase B result is a stronger answer than what Henry asked for: instead of
   "does a bigger/better-trained student on the *same* labels beat majority vote,"
   it shows the *same* training path produces a competent verifier when supervision
   is format/distribution-compatible, and stays weak on the current generated labels
   regardless of student scale (1.5B or 3B). That's a cleaner isolation of "supervision
   is the bottleneck" than a capacity/data scale-up would have been.

## Proposed merged structure

1. **Intro** — motivating question: does privileged (answer-aware) teacher supervision
   help step-error localization, and can that advantage be distilled into a small,
   deployable, answer-blind verifier? (Henry's framing, lightly extended.)

2. **Related work** — Henry's §1.1-1.4 mostly as-is. Add Saksham's novelty/scooping
   list (ProcessLID, RetrievalPRM, FreePRM, uPRM, ThinkPRM, Qwen PRM800K) to the
   generated/weak-supervision-PRM paragraph — these are directly relevant once the
   mismatch result is in the paper and currently uncited in Henry's draft.

3. **Contribution 1 — the teacher-level sweet spot** (Henry §2.1-2.5, unchanged).
   This stays the validated spine: GSM8K ≈0, MATH +0.05 [0.01,0.09] significant,
   OlympiadBench ≈0, mechanism = rescue-of-self-verification-failure × tractability,
   cross-family replication (Qwen-27B +0.082).

4. **Contribution 2 — does it distill? A diagnosed negative** (Henry §2.6-2.7, extended):
   - Step-level + downstream BoN null as currently written (no-GT ≥ priv, neither
     beats majority vote).
   - Diffuse-churn mechanism (69% label agreement, symmetric disagreement) as currently
     written.
   - **New:** insert the B3 positive control here. This directly answers Henry's own
     "Boundary of the claim" ask #1 — it's the missing sentence, not a new section.
   - **New:** the N=1000 shared-pool paired BoN Edward attempted locally is close but
     crashed on an auth bug before completing (see Open Items below) — if finished,
     replaces the currently-cited N=200 McNemar test (p=0.14, underpowered per Henry's
     own caveat) with a properly powered N=1000 version.

5. **Contribution 3 — diagnosing the null further: capacity vs. supervision** (new,
   from Saksham's Phase B work, reframed as answering Henry's ask #2 rather than as a
   standalone paper):
   - Frame: "Section 4 shows privilege doesn't distill even though the pipeline is
     provably sensitive to teacher quality (§3.X) — so is the 1.5B student simply too
     small, or is the generated-label supervision itself the problem?"
   - Under an identical Qwen2.5-3B training/eval path, ProcessBench-style gold labels
     (GSM8K, OmniMath source configs) transfer strongly (0.75-0.85 ROC-AUC,
     sequence-cluster bootstrap CIs, 4 seeds each) while the same generated
     privileged/no-GT teacher labels remain weak (0.55-0.63 ROC-AUC) — at both 1.5B
     and 3B scale.
   - Public Qwen2.5-Math-7B-PRM800K baseline (0.84 ROC-AUC) as external calibration —
     not a SOTA claim.
   - Conclusion: the bottleneck is not (only) student capacity — it's a supervision
     distribution/format mismatch in the generated-label pipeline specifically.

6. **Limitations** (merge both drafts' honesty sections):
   - Henry's: N=200 paired test underpowered (resolve if the N=1000 rerun finishes).
   - Saksham's: the gold-vs-generated contrast confounds provenance, source-problem
     distribution, and label format — the missing controlled cell is generated labels
     on the *same* GSM8K/OmniMath source problems. Also: generated-label baselines use
     1 seed vs. 4 for gold rows; OlympiadBench is high-variance (boundary diagnostic,
     not headline).
   - New, from the merge itself: the 1.5B (teacher-null) and 3B (Phase B) tracks aren't
     a controlled capacity ablation — they used different loss recipes and eval N, so
     "capacity doesn't fix it" is suggestive across tracks, not a clean single ablation.

7. **Conclusion** — privilege helps the teacher in a well-characterized tractability
   window; that advantage does not distill into a small verifier even though the
   pipeline is demonstrably sensitive to supervision quality; the reason is closer to
   supervision distribution/format mismatch than to raw student capacity.

## Status as of the draft in `paper/merged_draft.tex` (compiles clean via tectonic)

Resolved by direct investigation, not left as placeholders:

- **B3 positive control**: written into prose + §5.4, with the number
  recomputed from currently-committed data (see "Checkpoint provenance"
  below for why it isn't a straight copy of the historical log entry).
- **Flip-case table**: populated with 3 real, sourced rescue cases from
  `results/evidence_pack_n400/per_sample.jsonl` (37 total rescue cases exist;
  these 3 were picked to span error types, not for effect size).
- **Abstract ROC-AUC range**: was overstated (`0.75–0.85`); the actual
  single-seed range across all 8 gold-source seeds is `0.7256–0.7869`.
  Fixed to `0.73–0.79` (headline) `+ up to 0.83` (post-hoc ensemble
  diagnostic, correctly scoped as such).
- **Phase B (3B) numbers spot-checked against raw JSONs**: every number
  quoted in §6 of the merged draft traces to a uniquely-named,
  single-commit file (`results/diagnostics/...`), and the sequence-CI /
  calibration JSONs self-document their exact input paths
  (`model_a`/`model_b`/`scores` fields). No provenance ambiguity found on
  this track — verified to 4 decimal places on GSM8K seed0, OmniMath seed0,
  Qwen PRM800K, and both generated-label baselines.

### Checkpoint provenance issue found on the 1.5B track (real finding, not a typo)

The 2026-06-22 positive-control log entry (`RUNBOOK_PHASE_B.md`,
commit `ae912cf`) cites "priv_critique achieved ROC-AUC 0.624" as the
comparison point. That exact number does not match any git-committed
`results/ablation/priv_critique/processbench_results.json`:
- 2026-06-18 (commit `7c54a9d`, the commit Henry's Table 4 is sourced from,
  N=400 eval): **0.6309**
- 2026-06-22, same day as the B3 entry, a *different* re-evaluation logged
  in `RUNBOOK_PHASE_B.md` itself: **0.6288**
- 2026-06-23 (commit `eeb21d0`, full N=1000 eval, current committed value):
  **0.6251**
- The only file that actually reads ~0.624 is `priv_scoreonly`
  (**0.6236**, unchanged since 2026-06-18)

Root cause: `checkpoints/priv_critique.pt` and
`results/ablation/priv_critique/` are shared, non-namespaced paths that get
silently overwritten every time that ablation is retrained — flagged
already in `slfd plan` (2026-06-22) as a live risk, evidently realized here.
The "0.624" citation is most likely a mislabeling of `priv_scoreonly` in
that log entry, not a fabrication or a materially different run — but it
can't be proven from committed artifacts, since none of the intermediate
retrains were preserved under distinct names.

**Fix applied in the draft**: rather than trust any historical number, I
recomputed the positive control directly from the currently-committed,
identically-sized (6,505-step) per-step score files for `priv_critique` and
`weak_student` (`experiments/transfer_ci.py --model_a
results/ablation/priv_critique/per_step_scores.json --model_b
results/positive_control/weak_student/per_step_scores.json --n_boot
10000`): **+0.0497 ROC-AUC, 95% CI [0.0206, 0.0783], p=0.0012** —
significant, and consistent in direction/magnitude with the original
+0.048 estimate. The draft cites this reproduced number and explains the
discrepancy in a footnote rather than silently using either the old or new
number without comment.

**Recommended fix to the codebase** (not yet done): namespace
`checkpoints/<name>.pt` and `results/ablation/<name>/` by a run tag
(the `SEED=`-as-namespace pattern already proposed in `slfd plan`) so this
can't happen again. Worth doing before any more ablation reruns — it cost
real time to reconstruct this provenance from git archaeology, and the next
person won't have that git history memorized.

## Open items before this is submission-ready

- [ ] **Decide on `scripts/run_b0_paired.sh`** (the crashed local N=1000 shared-pool
      paired BoN). Root cause is now diagnosed, not just "an auth error": the script's
      local generator step tried to hit `localhost:8000` expecting a lightweight
      generator (`Qwen3-4B-Instruct-2507-MLX-8bit`), but that port is occupied by the
      `launchd`-managed Gemma-4 labeling teacher, and the background job's environment
      didn't carry a valid `OMLX_API_KEY` for it. Two sub-decisions: (a) is it worth
      pointing the script at a correctly-authenticated endpoint and finishing it before
      the deadline, given (b) the checkpoints it already trained used `MAX_STEPS=2000`
      and scored lower (0.55/0.58) than the ones behind the paper's headline numbers
      (0.63/0.64) — so it likely needs a clean retrain with matching hyperparameters
      first, not just an unblock-and-rerun. This is a "does the marginal rigor gain
      justify a day of compute with 12 days left" call, not a technical blocker.
- [ ] **Run the missing controlled cell**: generated teacher labels on the same
      GSM8K/OmniMath source problems used by the gold-label rows. This is the one
      experiment that would actually isolate provenance from source-distribution —
      currently the paper's most important honestly-flagged limitation.
- [ ] Reconcile author/venue metadata — Saksham's original draft is COLM ER workshop,
      double-blind, anonymized; the merged draft has no venue markers yet
      (`\author{\todo{...}}` is the only remaining placeholder in the .tex).
- [x] Typos in Henry's draft — fixed during the merge (conjunctive, external privilege,
      first-error F1, "uniformly beneficial" → "conditionally beneficial", broken
      `§??` cross-reference now resolves via `\ref`).
- [ ] Table consolidation (already flagged as Edward's call in HANDOFF_HENRY.md).
- [ ] Figure out where the merged source actually lives — Henry's is on his Overleaf
      project; Saksham's PDF has no shared Overleaf link in the Slack message. Need a
      single source of truth before drafting further.

## Deadline

COLM Efficient Reasoning Workshop, 2026-07-12.
