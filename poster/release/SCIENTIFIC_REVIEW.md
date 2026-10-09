# V2 scientific review of rendered candidate

**Verdict: previously identified scientific issues are addressed in this candidate; pending focused verification by the independent reviewer. This is not release approval.**

Reviewed actual `final-print.pdf` extracted text, complete rendered preview, and a fresh PDF crop of the navy intervention panel. Reviewed PDF SHA256: `881acdc66487211298ccbe487cc3940a4d9b94a5aef09d095b74d5951e1672f8`. The coordinator may subsequently rebuild; this hash identifies the reviewed render. The page is 2448 × 2448 pt (34 × 34 in). This bounded review uses direct training-code/input checks documented in EVIDENCE_AUDIT.md; it does not repeat model training or infer new significance.

## Findings against actual poster

- **F1:** The navy panel shows binary error flags `0,1,1 → 0,1,0`, explicitly defines 1 as error flag, keeps problem/prefix/step text fixed, and says target relabeling changes balanced sampling. Their separate effects are not isolated. Feedback wording and score magnitude are identified as unused by score-only training. The diagram is marked conceptual. The sign is scientifically correct: these are error flags, not numeric correctness-logit BCE targets (which are their complements). Later 0 does not certify mathematical correctness.
- **F2:** Gold is a same-source reference, not an identical-input counterfactual. The chart/caption plus reference panel state slight input/segmentation differences, including 2,082 versus 2,080 steps. The 88.8% expression is explicitly descriptive, uses unrounded four-seed means, and disclaims a causal fraction of failure and an inferred CI. No equivalence-to-gold assertion appears.
- **F3:** Full preview shows both MATH confidence-interval caps above the bar. Stored interval remains the evidence source; no new uncertainty is claimed.
- **F4:** PRM is defined in the header; no-GT is explained by absent reference-solution access; ProcessBench is described as the benchmark locating first reasoning errors; fixed student Qwen2.5-3B-Instruct appears in card 03. The conceptual companion now explains the target change instead of repeating endpoint values. No timed novice study was performed.
- **F5:** Distinct circle/square markers and offset seed observations distinguish raw and first-error conditions; gold diamonds and range expose its descriptive seed variation. Caption says seed observations are not confidence intervals. No bootstrap p-values are attached to these per-seed means.
- **F6:** The prominent footer says downstream reasoning gains are not established; the 1.5B-to-3B contrast remains explicitly non-capacity-ablation.
- **F7/F9:** Teacher panel calls this an observed pattern in the probes, states MATH N=400 with paired bootstrap CI, GSM8K N=50 without an available CI, and separate Olympiad protocol. It disclaims a universal difficulty law. Caveats are in an enlarged layout caption, visibly separate from axes; physical viewing distance remains untested.
- **F8:** Cost panel clearly specifies one 500-step run ≈0.3 GPU-hours and preliminary sweep ≈15 GPU-hours, setup-specific paper estimates, not end-to-end savings or full-study cost.
- **F10:** Official logo remains; no scientific concern introduced by retaining raster branding. Effective resolution/print appearance are production limitations.
- **F11:** Actual PDF lists Edward's Xi’an Jiaotong-Liverpool University (XJTLU) affiliation; CSU is absent, consistent with the subsequent explicit author instruction. Other affiliations remain. No public-PDF byte-identity assertion appears.

All requested headline numbers are present unchanged: 0.726→0.777; 0.631 versus 0.641; 0.6183→0.7366; 0.5494→0.6762; gold 0.7515; 88.8%; ≈0.3 and ≈15 GPU-hours. Teacher F1 and student ROC-AUC remain distinctly labeled.

No new blocking scientific defect was identified in this rendered candidate. Outstanding limitations remain the manuscript/public-revision identity, unresolved manuscript-level ensemble-gold assertion and positive-control numerical snapshot, lack of downstream-gain evidence, and physical print/attendee acceptance. The poster avoids presenting those unresolved issues as established results. PDF font/link/QR and isolated-build checks belong to the coordinator's technical review, not this scientific verdict.
