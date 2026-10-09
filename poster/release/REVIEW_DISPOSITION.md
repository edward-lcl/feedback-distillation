# V2 independent-review disposition

2026-10-09. Candidate revision for focused independent verification; **not release approval**. Read the complete `poster/independent-review/REPORT.md`, its pre-correction `FIRST_ASSESSMENT.md`, stored evidence receipt and relevant training code/data. The author's V2 request authorizes scientifically qualified replacement wording, not changed numbers or a manuscript revision. A subsequent explicit author instruction resolves F11 by removing CSU from Edward's affiliation.

| Finding | Disposition | V2 response and evidential basis |
|---|---|---|
| F1 — active intervention | **Accept** | Replace generic cosmetic-rendering explanation with binary first-error targets at fixed reasoning inputs. Trainer BCE consumes `is_error`; score-only disables feedback LM, and balanced pool membership also uses `is_error`. No-GT changes 254 / 2,080 flags; privileged 377 / 2,080. Explain target/sampling contributions are unseparated. Diagram uses error flags; actual correctness-logit BCE targets are their complements. |
| F2 — gold counterfactual | **Accept** | Gold is labeled a same-source reference with small input/segmentation differences (2,082 vs 2,080 rows). Exact generated raw→first-error inputs match; gold does not. Retain 88.8% solely as the descriptive measured-gap ratio, with explicit noncausal qualification. No new aligned-control experiment is authorized or needed for this revision. |
| F3 — hidden teacher CI | **Accept; preserve fix** | Preserve high z-order whiskers above bars, with both caps visible. Estimate .0506 and stored 95% interval [.0095,.0925] unchanged. |
| F4 — first-minute explanation | **Accept** | Define process reward model as verifier scoring reasoning steps; define no-GT as teacher without a reference solution, expand ProcessBench as a first-error benchmark, name Qwen2.5-3B. Repurpose dark companion panel for the active target transformation. Later non-error targets are not certificates of mathematical correctness. This is a design goal, not a measured novice-reading-time claim. |
| F5 — grayscale and seed observations | **Accept** | Distinguish raw and first-error observations through marker shape and offset, including nearly overlapping seed points. Expose gold seed variation as descriptive observations. Seed spread is not a confidence interval; no ensemble-bootstrap statistic is attached to per-seed mean arrows or gap ratio. |
| F6 — downstream scope | **Accept** | Add visible “Improved first-error ranking; downstream reasoning gains not established.” Manuscript §7 and unresolved best-of-N evaluation support this limitation. Preserve off-policy and non-capacity-ablation scope. |
| F7 — teacher pattern | **Accept, scoped** | Present an observed pattern in the studied probes, not a general difficulty law or zero-benefit claim outside MATH. GSM8K N=50, MATH N=400 and separate Olympiad protocol do not isolate difficulty as a causal factor. No invented uncertainty intervals. |
| F8 — compute unit | **Accept** | State ≈0.3 GPU-hours is one 500-step diagnostic run, compared with ≈15 GPU-hours preliminary sweep. Both are setup-specific manuscript estimates; no end-to-end savings or full-study cost claim. |
| F9 — teacher caveat type | **Accept** | Promote essential sample/protocol/uncertainty caveats to readable poster captions and trim repetition. Render/crop checks can assess layout; physical print distance remains untested. |
| F10 — logo quality | **Modify** | Retain the supplied official raster asset, preserving proportions; no verified vector original was supplied. V2 reduces placement from 374 to 216 pt, raising effective resolution from 110.9 to 192 ppi without recovering or inventing detail. Vector redrawing/upsampling is not an authorized substitute. |
| F11 — affiliation / provenance | **Accept; affiliation resolved by author** | Author explicitly requests Xi'an Jiaotong-Liverpool University (XJTLU) only for Edward; remove CSU, preserve other affiliations. No invented expansion. Public-page title/authors/QR checks remain distinct from byte identity of the local detailed manuscript and public PDF; the latter is unverified. |

## Candidate wording constraints

“With reasoning inputs held fixed, converting generated error flags into first-error training targets improves transfer to ProcessBench-MATH. This target change also alters balanced sampling; their separate contributions are not isolated.”

“88.8% of the measured no-GT gap to the same-source gold reference” is descriptive arithmetic:

`(0.7365938255275931 − 0.6182525360752396) / (0.7514507421249089 − 0.6182525360752396)`.

It is not a fraction of original failure causally explained, not an equivalence test, and not confidence coverage. The five four-seed means and every seed observation remain unchanged. Mean per-seed AUC, AUC of four-seed mean scores, and bootstrap mean paired gaps remain distinct estimands.

## Evidence disagreements resolved by qualification

The reviewer is supported on the two central issues. The old rerender code docstring's claim that only label source differs relative to gold is contradicted by actual input/segmentation comparisons. Historical framing's generic rendering language omits the active binary-target/sampling change. Neither comment overrides the data or training code.

The manuscript's claim that both rerendered mean-score ensembles are statistically tied with gold remains unsupported: stored gold-minus-no-GT interval [.0112818,.0350949] excludes zero. Even an interval containing zero would not establish equivalence. V2 introduces no gold-equivalence claim and does not edit the accepted manuscript. The positive-control precise numerical-snapshot discrepancy also remains documented, without adding a new precise positive-control interval.

## Remaining judgments / limitations

- Independent reviewer should verify the revised causal and reference wording, chart semantics, and complete rendered layout. Compilation alone does not resolve findings.
- Author/publication-version identity of the local anonymous detailed PDF and current public PDF remains unverified; the public paper page and its QR remain the appropriate destination.
- Higher-resolution official branding would improve printing if an authorized original becomes available.
- Physical viewing distance, print proof, printer-specific bleed/CMYK/PDF-X, and downstream reasoning gains are not established by this revision.
- Exact affiliations need no further clarification for Edward after the explicit correction; other author affiliations are preserved.

All V1 and independent-review source files remain unchanged. See `EVIDENCE_AUDIT.md` for provenance and `REVIEW.md` for actual final validation outcomes; layout actions above are the implemented revision specification, not claims of physical acceptance.
