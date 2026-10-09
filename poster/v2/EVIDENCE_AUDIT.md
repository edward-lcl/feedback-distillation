# V2 scientific evidence audit — candidate, not release approval

Audit date: 2026-10-09. Repository HEAD inspected: `e598eaa077df71463f6bb9ae9a23fd6c1a09f35c`. No manuscript or submitted poster was changed. Machine-readable exact values and artifact paths: `data/verified_claims.json`.

## Source authority

| Source | Identification | Use |
|---|---|---|
| `~/Downloads/171_Before_You_Scale_The_Cost_-1.pdf` | 9 pages; PDF creation July 20 2026; SHA256 `8c0f8acadb203fd6a153c84002dd46de5a10fecd1914d89eff6f015b6f10368e` | Locally verified detailed scientific manuscript; public PDF byte identity unverified. Contains completed same-source/rerender experiments, Table 4 and §6.2. |
| `~/Downloads/171_Before_You_Scale_The_Cost_.pdf` | Identical SHA256 | Duplicate, not a separate revision. |
| `paper/slfd_colm_er_final.pdf` | 12 pages; PDF creation July 19 2026; SHA256 `a5ca3176855d0849c06ab23230a481d6b25825939f1f21be7f686b8bbf0ae866` | Older repository manuscript; supplemental provenance, not final layout/section authority. |
| `~/Downloads/COLM_2026_ER_Workshop_Poster_Saksham_Kapoor.pdf` | PowerPoint export, October 6 2026; 960 × 540 pt; SHA256 `22e2399305a6b3f83bab8baecb811334f7990dd631da6bd40c0814c6b5287f63` | Submitted design and attribution reference; numerical interpretation audited below. |
| `PAPER_FRAMING.md` | Canonical framing, already includes completed format rerender | Canonical historical framing. Author-approved V2 candidate qualifies its broad mechanism/capacity language using the actual diagnostic path; this file and the accepted manuscript remain unchanged. |
| Committed `results/` JSONs and evaluation scripts | Exact values rather than manually transcribed chart labels | Primary numerical authority; reveal a manuscript statistical overstatement below. |

Authority was established from actual text, experiment coverage, hashes and metadata, not file modification order. The latest manuscript still carries an anonymous under-review header; poster attribution comes from the submitted poster. Public page confirms author order. During V2 the author explicitly instructed removal of CSU from Edward's affiliation and use of Xi'an Jiaotong-Liverpool University (XJTLU) only; other affiliations are preserved. Public manuscript revision identity remains unverified.

## Claim-to-evidence ledger

All manuscript sections/tables below refer to the 9-page Downloads PDF.

| Claim / V2 candidate wording | Manuscript | Primary evidence | Statistical interpretation |
|---|---|---|---|
| “Answer-aware worked solutions improve MATH teacher first-error F1: 0.726 → 0.777.” | §3, Table 1 | `results/evidence_pack_n400/evidence_pack.json`: 0.7262 → 0.7768; N=400; `experiments/evidence_pack.py` | Paired problem bootstrap, observed Δ=0.0506; stored 95% CI [0.0095,0.0925]. This is teacher F1, not student thresholded F1. |
| “Observed teacher pattern in these probes: MATH benefits from reference access.” | §3, Table 1 | GSM8K: `results/teacher_eval/privilege_probe.json`, 0.8889 → 0.8372, N=50. MATH above. Olympiad OE: `results/teacher_eval_olympiadbench_oe/privilege_probe.json`, within-run Δ=-0.0289 | Do not invent GSM/Olympiad CIs. Manuscript reports n.s.; Olympiad uses a separate scoring protocol and should not be plotted as directly comparable absolute F1. Its teacher sample size is not the two training seeds of a separate student experiment. |
| “The initial 1.5B student did not gain from privilege: 0.631 vs 0.641 ROC-AUC.” | §4, Table 2 | Historical committed artifacts at `7c54a9d:results/ablation/priv_critique/processbench_results.json` = 0.6309052658278843; `7c54a9d:results/ablation/nogt_critique/processbench_results.json` = 0.6405774990954429 | Ntrain=1000, Neval=400, 2589 steps. HEAD files were replaced by 1000-solution evaluations and must not be used to reconstruct these exact cells. No inference of equivalence or universal privilege failure. |
| “Holding generated reasoning inputs fixed, first-error targets improve no-GT mean ROC-AUC 0.6183 → 0.7366.” | §6.2, Table 4 | `results/diagnostics/generated_nogt_gsm8k400_to_math1000_qwen3b_bce_bal_seed{0,1,2,3}/processbench_results.json`; PB version inserts `_pbformat` after `gsm8k400` | Arithmetic mean of four per-seed AUCs. Exact mean difference +0.1183412894523535, displayed +0.1183. No bootstrap CI belongs on this difference. |
| “First-error targets improve privileged mean ROC-AUC 0.5494 → 0.6762.” | §6.2, Table 4 | Same family replacing `nogt` with `priv` | Four-seed mean. Exact mean difference +0.1267963569604578, displayed +0.1268. |
| “Same-source GSM8K gold reference: 0.7515 mean ROC-AUC; inputs/segmentation differ slightly.” | §5–6, Table 4 | `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{0,1,2,3}/processbench_results.json` | Four-seed mean 0.7514507421249089. Plot ranges or seed points as seed variability, not confidence intervals. |
| “The improvement is 88.8% of the measured no-GT gap to this same-source gold reference.” | Derived from Table 4; explicit in `paper/generated/format_rerender_gsm8k400_table.md` and `results/RESULTS.md` | Exact three means above; formula below | Descriptive ratio, not a bootstrap statistic, confidence level, causal fraction of all PRM failure, or accuracy percentage. |
| “One 500-step diagnostic run ≈0.3 GPU-hours; preliminary recipe sweep ≈15 GPU-hours. Setup-specific reported estimates.” | Abstract, §1, §7; older final tex lines 104–106 | Manuscript-reported estimates: 500-step LoRA diagnostic, roughly 50-run preliminary sweep | Runtime accounting artifact was not found. Mark estimates as paper-reported, not independently timed. ≈50× is an arithmetic cost ratio, not realized end-to-end savings or a universal guarantee. |
| “Keep reasoning inputs fixed; retain only the first teacher-flagged error as an error target. This also changes balanced sampling; separate contributions are not isolated.” | §6.2; code qualification | `scripts/rerender_labels_processbench_format.py::rerender_record`; `training/slfd_trainer.py:143–170,198–242`; `experiments/train_slfd.py:30–35`; `scripts/run_gold_scorehead_gate.sh:88` | Active score-only BCE channel is binary `is_error`, not score magnitude or feedback wording. Later non-error targets do not certify mathematical correctness. The BCE scorer predicts correctness, so its numeric loss target is the complement of the error flag. |

## Exact seed values and controlled generated-label comparison

All main diagnostic cells use Qwen2.5-3B-Instruct, scalar score head, LoRA, BCE, balanced batches, 500 steps, batch size 2, 400 GSM8K candidate solutions, and serial batch-size-1 evaluation on ProcessBench-MATH 1000 solutions / 6505 steps / 594 error steps. Raw and first-error-target generated files have identical ordered (problem, prefix, step) inputs, 2,080 rows each. Gold is a same-source reference with 2,082 rows and small segmentation/input differences; it is not an identical-input label-only counterfactual. Balanced positive/negative pool membership changes with target relabeling. The experiment does not isolate target semantics from that sampling consequence.

| Condition | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| Gold | .743887 | .776024 | .760250 | .725641 | .751451 |
| Raw no-GT | .635036 | .591701 | .661418 | .584855 | .618253 |
| PB no-GT | .701080 | .755452 | .735351 | .754492 | .736594 |
| Raw privileged | .532630 | .559856 | .468021 | .637122 | .549407 |
| PB privileged | .789145 | .601620 | .727130 | .586919 | .676204 |

Exact no-GT recovery:

`(.7365938255275931 - .6182525360752396) / (.7514507421249089 - .6182525360752396) = .8884600848770013 = 88.8460%`, rounded to **88.8%**.

This is a descriptive fraction of the measured gap to the reference, not the fraction of original failure causally explained by target convention. Gold's input/segmentation differences prevent a pure label-only interpretation. This uses unrounded artifact means; substituting the four-decimal printed means shifts the last decimal. Privileged recovery is approximately 62.8%, also descriptive. No uncertainty interval for either ratio has been computed.

## Reconciliation of the submitted poster's estimands

The submitted poster lists single-checkpoint mean AUC endpoints next to deltas and p-values from a different analysis. Both sets of values are real, but placing them in the same row without identifying their estimands is misleading.

1. **Mean per-seed AUC:** `mean_s AUC(y, score_s)`. Used for the main rerender plot and 88.8% recovery.
2. **AUC of mean scores:** `AUC(y, mean_s score_s)`. `experiments/make_score_ensemble.py` forms four-checkpoint score averages. AUC is nonlinear, so this is not quantity 1.
3. **Stored bootstrap gap:** `experiments/sequence_transfer_ci.py` resamples 1000 whole solutions with replacement, computes paired AUC gaps for the mean-score artifacts, then stores the **mean of 5000 bootstrap gaps**, percentile CI, and a two-sided plus-one tail p-value. Consequently even this stored gap is not exactly the simple difference of the displayed full-sample ensemble AUCs. Bootstrap uncertainty here concerns evaluation sequences, not training-seed variation.

Committed inference files under `results/diagnostics/`:

| File | Compared score artifacts | Bootstrap mean gap | 95% CI | Two-sided p |
|---|---|---:|---|---:|
| `pbformat_nogt_4seed_mean_vs_raw_nogt_4seed_mean_sequence_ci.json` | PB no-GT minus raw no-GT | +.126193 | [.104296,.147594] | .00039992 |
| `pbformat_priv_4seed_mean_vs_raw_priv_4seed_mean_sequence_ci.json` | PB privileged minus raw privileged | +.2070 | [.1831,.2305] | .00039992 |
| `gsm8k_4seed_mean_vs_pbformat_priv_4seed_mean_sequence_ci.json` | Gold minus PB privileged | +.0044 | [-.0080,.0172] | .4983 |
| `gsm8k_4seed_mean_vs_pbformat_nogt_4seed_mean_sequence_ci.json` | Gold minus PB no-GT | +.0232 | [.0113,.0351] | .00039992 |

Score files live under `results/diagnostics/ensembles/{gsm8k_4seed_mean,pbformat_nogt_gsm8k400_4seed_mean,pbformat_priv_gsm8k400_4seed_mean,raw_nogt_gsm8k400_4seed_mean,raw_priv_gsm8k400_4seed_mean}/per_step_scores.json`; exact file pointers are embedded in each CI artifact and copied into `data/verified_claims.json`.

**Poster correction:** retain per-seed means together; if statistics are included, put them in a separately labeled mean-score-ensemble analysis. Do not “fix” +.1262 to +.1183 while retaining its p-value. Do not attach a bootstrap footnote to 88.8%.

## Inconsistencies and limitations

- **Latest manuscript §6.2 overstates a tie:** it says both PB-format mean-score artifacts are statistically tied with gold. The committed no-GT CI excludes zero. The privileged comparison interval includes zero, but failure to reject is not proof of equivalence. Sequence bootstrap treats solutions as clusters; it does not account for repeated problem strings across solutions (955 distinct problems / 1,000 solutions). The poster avoids the erroneous joint-tie claim.
- **Capacity:** 1.5B uses a score-and-critique path and a different evaluation slice from 3B. Their contrast is not a clean capacity ablation. The controlled 3B comparison demonstrates learnability under matched supervision in this setup; it does not rule out all capacity effects.
- **Downstream best-of-N:** Table 3 has a string-matching fallback caveat; repo framing marks pre-July-6 downstream runs superseded pending symbolic regrade. V2 omits these fragile ancillary numbers. Improved first-error ranking does not establish downstream reasoning gains; §7 leaves the 3B best-of-N validation incomplete.
- **Positive control:** manuscript §4 reports +.0497 [ .0206,.0783 ] p=.0012; `results/positive_control/transfer_ci.json` is actually a text receipt despite its extension and records +.0481 [.0200,.0767] using 10,000 step-bootstrap samples. Both support coarse pipeline sensitivity; numerical snapshot mismatch remains unresolved. Do not put a new precise positive-control CI on the poster.
- **Teacher sweet spot:** cross-dataset protocols and small easy/hardest samples constrain a universal inverted-U interpretation; use as supporting context, not a universal law or fitted curve.
- **Public claims:** no SOTA claim, no implication that format is the only PRM bottleneck, no guarantee of 50× savings, and no assertion that rerendering creates fresh human-correct judgments.
- **Timing:** approximate costs have manuscript provenance but lack independently verified runtime logs. This is explicitly disclosed wherever the cost comparison is used.

## Prior audit validation (retained provenance)

The original poster audit reports: read both Downloads manuscript contents and metadata; compared SHA256 hashes; read submitted poster text; inspected latest manuscript sections and generated tables; recomputed all five main seed means and recovery ratio from committed metric JSONs; inspected score-ensemble and sequence-cluster-bootstrap implementations; recovered historical 1.5B metrics from Git; inspected rerender implementation and teacher evidence pack. No new models, data, significance tests, confidence intervals, or measurements were invented. Full model training was not rerun.

## Public-source cross-check

The prior coordinator's website scrape (V1 `../provenance/scrape_0.json`; fresh independent-review receipt bundled in `provenance/independent-review/public-check.json`) includes the public [OpenReview page](https://openreview.net/forum?id=gKNwyhDxPk), published August 7 and last modified August 11, 2026, visible to Everyone under CC BY 4.0. The prior audit inspected its returned title, four authors, abstract and submission number 171. The abstract corroborates the key teacher improvement, transfer null, four-seed gold 0.7515, rerender 0.6183 → 0.7366, and approximate costs. The public PDF could not be fetched via the coordinator's scrape attempt, so byte identity between that public revision and the July 20 Downloads manuscript is unverified. The Downloads PDF remains the locally inspected detailed manuscript; the public abstract is an additional agreement check, not proof that every appendix statistic is identical. Public page author order agrees with the submitted poster. Edward's affiliation is corrected to Xi'an Jiaotong-Liverpool University (XJTLU), with CSU removed by explicit author instruction during V2. Other submitted-poster affiliations remain preserved.

The two exact historical 1.5B metric JSONs are bundled without modification in `data/historical/priv_critique_processbench_results.json` and `data/historical/nogt_critique_processbench_results.json`. `data/historical/provenance.json` records the full source commit, original paths and SHA256 hashes. They allow verification of these reported cells after copying `poster/` without Git history.

## V2 intervention audit and reproducible evidence

The complete independent `REPORT.md`, `FIRST_ASSESSMENT.md`, and `evidence-check.json` are preserved under `provenance/independent-review/`. V2 independently inspected the rerender function, active trainer branch, ablation mapping and launch command and recomputed the following from the original local training inputs:

| Generated condition | Rows raw / first-error | Error flags raw → first-error | Changed targets | Input identity |
|---|---:|---:|---:|---|
| No-GT | 2,080 / 2,080 | 425 → 171 | 254 | Exact, same order |
| Privileged | 2,080 / 2,080 | 571 → 194 | 377 | Exact, same order |

Gold contains 2,082 steps and 207 error flags. After whitespace normalization, 15 generated and 17 gold input rows remain unmatched. This corroborates the independent review's segmentation caveat. No new training or label repair was performed.

Minimal self-contained verification: `python3 data/intervention_evidence/check.py`. Bundled records hash only the exact/whitespace-normalized input tuples and retain error flags; source SHA256s and algorithm are recorded in `data/intervention_evidence/manifest.json` and the generated receipt is `provenance/intervention_check.json`. Code excerpts carry original line numbers and full-source hashes. These enable direct identity/count verification without distributing generated free-text feedback. The hashes do not reconstruct raw reasoning text, and are not substitutes for full raw data in a new training replication.

The rerender script's historical docstring says only label source varies relative to gold. The actual gold input comparison contradicts that broad claim; V2 follows measured input identity instead of this stale explanatory comment. Its changes to score magnitude and feedback strings occur in files but are unused as training channels by the tested score-only BCE path. Conceptual V2 examples explicitly depict binary error flags, not invented experimental traces or a mathematically correct/incorrect oracle.

F1–F11 actions are recorded in `REVIEW_DISPOSITION.md`. The V2 interpretation is author-authorized candidate wording for independent verification. It does not amend the accepted manuscript, establish public PDF byte identity, or imply release/physical-print acceptance.
