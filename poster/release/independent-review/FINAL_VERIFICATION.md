# V2 focused independent verification

**Recommendation: 2. Release-ready after minor corrections.** All original major scientific/communication findings are resolved. There are **no remaining blockers or majors**. Remove the obsolete “pending verification” release-status text (N1); no new experiment, scientific rewrite, or redesign is required. The scientific review is closed, subject only to preserving the reviewed content in that cleanup.

## Reviewed artifact and independence

- Candidate: `poster/v2/before-you-scale-v2-review-candidate.zip`.
- ZIP SHA256: `c10caa65e83bb5691d5ba600bf1ee0e8565288db36bdecd1c9a41e4e4ae51d94`.
- Reviewed PDF: archive member `before-you-scale-v2/final-print.pdf`.
- **PDF SHA256:** `881acdc66487211298ccbe487cc3940a4d9b94a5aef09d095b74d5951e1672f8`.
- Reviewed 2026-10-09 against repository HEAD `e598eaa077df71463f6bb9ae9a23fd6c1a09f35c`.

I extracted the archive to a temporary directory, rendered the actual PDF, compared it with the original F1–F11 report, and recorded an initial assessment **before** opening `REVIEW_DISPOSITION.md`, `SCIENTIFIC_REVIEW.md`, or `VISUAL_REVIEW.md`. I then read those documents and checked their claims against independently computed results. Their reports were not used as validation evidence. The candidate, manuscript, and editable sources were not modified; this report is the only repository deliverable.

## F1–F11 disposition

Locations below refer to the packaged one-page PDF and packaged source.

| Finding | Disposition | Independently checked evidence / exact location | Required action |
|---|---|---|---|
| **F1 Active intervention** | **Resolved** | Navy panel, `poster.typ:76–98`: “problem, prefix and step text stay identical,” “Targets + sampling change together,” and “Their separate effects are not isolated.” Source/data checks confirm both input identity and changed flags. Example is explicitly conceptual; 1 is defined as an **error flag**, not the BCE correctness target. Feedback and magnitude are correctly described as unused by this score-only path. | None. Do not remove the sampling qualification or reinterpret flags as literal correctness targets. |
| **F2 Gold reference** | **Resolved** | Gold panel, `poster.typ:100–112`, and chart key: same-source reference; **2,082 vs 2,080** steps; input/segmentation differences; unrounded four-seed formula; explicit noncausal interpretation and no inferred CI. Ratio independently reproduces. No equivalence, gold-level performance, or pure label-only counterfactual claim appears. | None. |
| **F3 Teacher CI** | **Resolved** | Lower-right MATH bar: independent PDF crop clearly shows the full vertical interval and **both caps**, including the cap inside the bar. Figure code uses z-order 5 above the bar; estimate 0.0506 and endpoints [0.0095, 0.0925] match evidence. | None. |
| **F4 Accessibility** | **Resolved** | Header defines PRM; card 02 defines no-GT; card 03 names Qwen2.5-3B and fixed recipe/evaluation; hero heading identifies the benchmark/task; navy example explains the active change; footer states the unproven downstream outcome. The six-question story is now self-contained for an AI researcher. | None. This is a reviewer walkthrough, not a measured 60-second novice study. |
| **F5 Seeds / uncertainty** | **Resolved** | Main chart uses raw circles, transformed squares, separated/staggered seed strips, and gold diamonds/range/dashed mean. Shapes remain distinguishable in an independently rendered grayscale image. Legend explicitly distinguishes observations from CIs. Numeric seed positions and means match the artifacts. | None. Legend is close-reading text (~21.9 pt), but unobscured and not a release impediment. |
| **F6 Downstream limitation** | **Resolved** | Bold footer: “Improved first-error ranking; downstream reasoning gains not established.” Separate-verifier/off-policy and non-capacity-ablation scope remain visible. Consistent with manuscript §7. | None. No downstream run is needed to release this scoped claim. |
| **F7 Teacher pattern scope** | **Resolved** | Lower-right heading says “OBSERVED TEACHER PATTERN IN THESE PROBES”; caption distinguishes sample sizes/protocols and states “no universal difficulty law is established.” Unsupported GSM8K non-significance wording is removed; no invented interval appears. | None. |
| **F8 GPU-hour accounting** | **Resolved** | Compute panel: “One 500-step run,” ≈0.3 GPU-hours versus preliminary sweep ≈15; explicit setup-specific paper estimates, not end-to-end savings or full-study cost. | None. Independent runtime measurement remains unavailable, but is no longer represented as established. |
| **F9 Essential readability** | **Resolved** | Actual teacher protocol/CI captions export at **25 pt**, replacing 16.625-pt caveats; chart labels are enlarged. Key intervention, gold, and downstream qualifications are visible without clipping. | None for scientific/digital release. **External dependency:** physical viewing/print proof, only for production acceptance. |
| **F10 Branding** | **Resolved** | Top-right original Algoverse raster is byte-identical to V1, proportionally placed at 216 pt width: **192 effective ppi**. No fabricated redraw or detail-generating upsample. This implements the original smaller-placement alternative. | None for digital release. Higher-resolution official art is optional production improvement, not a scientific gate. |
| **F11 Affiliation / provenance** | **Partially resolved; external remainder only** | Header and metadata list Edward's **Xi’an Jiaotong-Liverpool University (XJTLU)** affiliation; CSU is absent, satisfying the explicit author instruction. Exact public-PDF byte identity remains unverified and is not asserted. | No further affiliation approval needed. **External dependency:** publication-version reconciliation if desired; does **not** block this poster's scientific/digital release. |

## Critical evidence independently verified

**Intervention:** I compared bundled fingerprint rows against the actual repository JSONL fields, recomputing exact and whitespace-normalized input hashes. All five source-file hashes and all four code-excerpt hashes match the manifest; excerpt lines also match their complete repository source files. This checks the bundle's asserted evidence against original inputs rather than trusting its `check.py` output.

For both generated conditions, all 2,080 `(problem, solution_prefix, step_text)` rows are identical in order before/after transformation. From the 400 unflattened records per condition, I independently reconstructed “keep the first flagged error” and matched every transformed flag:

| Condition | Raw error flags | First-error flags | Changed flags |
|---|---:|---:|---:|
| No-GT | 425 | 171 | 254 |
| Privileged | 571 | 194 | 377 |

`training/slfd_trainer.py:143–170` partitions balanced pools using `is_error`; lines 193–245 use the same flags for BCE, with **correctness target = 1 − error flag**. `experiments/train_slfd.py:30–38` disables LM feedback for score-only; `scripts/run_gold_scorehead_gate.sh:77–94` selects that path. Thus the illustrated `0,1,1 → 0,1,0` flags are technically accurate without claiming those are the literal BCE target bits. Targets and sampling remain unseparated; the revised poster says so.

**Gold:** Independently confirmed 2,082 gold versus 2,080 generated rows and 15 generated / 17 gold unmatched input rows after whitespace normalization. V2 acknowledges rather than hides this limitation. It does not weaken the verified identity of the raw→transformed comparison.

**Quantities:** Recomputed ROC-AUC and average precision from the repository's 20 corresponding `per_step_scores.json` files; every pair matches its metric JSON. Bundled metric JSON bytes also match repository HEAD. Independently obtained four-seed mean AUCs:

| Raw no-GT | First-error no-GT | Raw privileged | First-error privileged | Gold |
|---:|---:|---:|---:|---:|
| 0.6182525361 | 0.7365938255 | 0.5494073282 | 0.6762036852 | 0.7514507421 |

`(0.736593825527593 − 0.6182525360752396) / (0.7514507421249089 − 0.6182525360752396) = 0.8884600848770006`, correctly rounded to **88.8%**. Arrow deltas round to +0.1183 and +0.1268. These are mean-per-seed AUC quantities; no ensemble significance is smuggled into their display.

**Manuscript:** The archive contains the original review but no detailed manuscript PDF. I checked the previously verified local nine-page manuscript, `Downloads/171_Before_You_Scale_The_Cost_-1.pdf`, SHA256 `8c0f8acadb203fd6a153c84002dd46de5a10fecd1914d89eff6f015b6f10368e`, including transfer/setup and limitations on pages 4 and 6. Existing manuscript-level gold-equivalence and positive-control snapshot discrepancies remain outside this correction; V2 does not repeat the problematic precise inferences. No new bootstrap or model training was performed.

## Regression and production checks

**Independently passed:** ZIP integrity and all 92 manifest hashes; source compilation to a separate temporary output; **exact extracted-text and 72-dpi pixel equality** between that compilation and the packaged PDF; one 2448×2448-pt page; all fonts embedded; no text spans outside page bounds; visual inspection for panel overlap/clipping; full color, grayscale, and enlarged teacher rendering; vector charts/QR with only the logo rasterized; XJTLU/CSU checks; rendered QR decoding at 150 dpi to `https://openreview.net/forum?id=gKNwyhDxPk`, matching the PDF hyperlink.

No new contradictory caption, incorrect number, misleading uncertainty encoding, materially ambiguous target sign, or unsupported scientific claim was found. The top-row failure means no observed student advantage in this setup; the later diagnostic identifies a supervision problem without claiming to isolate the entire historical 1.5B failure mechanism. Existing non-capacity-ablation language appropriately limits that inference.

Not newly verified: live public destination availability, public-PDF byte identity, physical phone scan, timed attendee study, printer color/bleed/PDF-X acceptance, runtime ledger, or downstream gains. Builder reports of these checks were not silently promoted to independent passes. These limits do not prevent the scoped scientific/digital release.

## Genuinely new issue

**N1 — Minor, release-status cleanup only.** Bottom acknowledgement line (`poster.typ:142`) still says “V2 candidate · pending focused independent verification”; document metadata (`poster.typ:6`) says “Pending independent re-review.” This was appropriate for the candidate but will be stale after this report. Before final distribution, remove the pending-review phrase and use neutral release metadata. Preserve the scientific text, figures, values, and affiliations. This does **not** invalidate the science or require another substantive review; the orchestrator can confirm the intended text-only diff, regenerate, and inspect the footer.

**Final decision:** scientific explanations approved; **release-ready after that minor correction**. No original blocker or major remains. Physical production and public-paper reconciliation are separate external dependencies, not reasons to reopen this scientific review.
