# Independent scientific QA

Reviewed 2026-10-09 by the evidence auditor, independently of poster production.

## Verdict: PASS for scientific content, with disclosed provenance limits

Reviewed editable `poster.typ`, figure-generation implementation, figure-data snapshot, extracted text of `final-print.pdf`, and rendered `final-preview.png`. This is a scientific-content gate; final QR decoding, dimensional/font checks and print-service acceptance belong to the integrated review.

### Verified in the rendered poster

- Teacher first-error F1 0.726 → 0.777 agrees with the N=400 evidence pack and final local manuscript.
- Initial 1.5B privileged/no-GT ROC-AUC 0.631/0.641 agrees with the historical 400-solution artifacts; these are bundled under `data/historical/` with Git/SHA256 provenance.
- Main figure's four means 0.6183, 0.7366, 0.5494, 0.6762, gold 0.7515, deltas +0.1183/+0.1268, and individual-seed locations agree with original metrics. Axis is explicitly ROC-AUC and numeric scale is visible. Seed dots are not represented as CIs.
- **Corrected during independent QA:** both layout directions initially described 88.8% as a mean-score quantity. The final source and final-PDF extraction instead correctly define it as `(PB − raw)/(gold − raw)` from unrounded four-seed mean AUCs, with no CI. Exact value 88.8460%.
- Main chart does not mix ensemble-bootstrap p-values with differences between seed-mean AUCs. The detailed ensemble comparisons are retained separately in `EVIDENCE_AUDIT.md`.
- Controlled diagnostic identifies 400 GSM8K training traces, 500-step LoRA/BCE, fixed 3B student and 1000 MATH evaluation solutions.
- Conceptual example is explicitly not an experimental sample. It correctly shows first-error retention, binary score replacement, and later errors becoming non-error labels, with an explicit warning that “Correct.” is a convention rather than a mathematical truth claim.
- Supporting teacher chart uses the actual MATH paired bootstrap CI and no invented GSM8K interval. OlympiadBench is separated in text with its protocol caveat; there is no interpolated invented difficulty curve.
- Compute bars have a 1:50 length ratio matching ≈0.3/≈15 GPU-hours. Text says “Reported” and uses approximation marks. The poster makes no measured-runtime or universal-savings claim.
- Scope note explicitly excludes a clean 1.5B-to-3B capacity ablation, universal diagnosis and SOTA claim. The main claim is restricted to “this failure.”
- Public OpenReview title and author order match the poster. The live abstract agrees with the headline science and costs; receipt: `provenance/scrape_0.json`.

### Findings resolved before this review

1. Wrong 88.8% estimand explanation replaced in both designs.
2. Ambiguous “Fix the student” changed to “Hold the student fixed.”
3. Training/evaluation counts and 500-step recipe added.
4. Fragile downstream best-of-N and inconsistent precise positive-control statistics omitted.
5. The latest manuscript's claim that both rerender ensembles tie gold is not repeated; the no-GT ensemble gold-gap CI excludes zero.

### Remaining limitations requiring no new poster data

- ≈0.3/≈15 GPU-hours are manuscript-reported estimates; no independent runtime ledger was located.
- Historical initial-student and current 3B results use different protocols; the poster correctly states the limitation.
- Positive-control sensitivity is supported, but its exact numerical snapshot differs between manuscript and stored receipt. The poster uses no precise control statistic.
- Public paper PDF scraping failed; its byte identity and every detailed statistic were not independently matched against the local July 20 manuscript. The live public abstract and author listing were verified.
- Affiliation text is inherited from the original submitted poster; public OpenReview verifies author names/order, not these affiliation strings.

See `EVIDENCE_AUDIT.md` for the full claim ledger, exact paths, formulas and unresolved manuscript discrepancies. No source manuscript or submitted poster was changed.
