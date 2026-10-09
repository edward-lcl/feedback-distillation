# Source inventory and authority

Inspected 2026-10-09. Local repository: `/Users/edward/Projects/feedback-distillation`, HEAD `e598eaa077df71463f6bb9ae9a23fd6c1a09f35c`. Initial working tree clean; this task writes only `poster/`.

| Material | Discovery and decision |
|---|---|
| Final local manuscript | `~/Downloads/171_Before_You_Scale_The_Cost_-1.pdf`, 9 pages, July 20 creation metadata. Its actual text includes the completed same-source and format-rerender experiments. SHA256 in `EVIDENCE_AUDIT.md`. Identical to `171_Before_You_Scale_The_Cost_.pdf` and `Step_Level_Feedback_Distillation-1.pdf`; duplicated downloads are not new revisions. |
| Submitted poster | `~/Downloads/COLM_2026_ER_Workshop_Poster_Saksham_Kapoor.pdf`, Oct 6 PowerPoint export, 960 × 540 pt, navy/teal/white. Kept unchanged. No matching editable poster source found in Downloads. |
| Other manuscript candidates | Downloads `saksham-merged-final.pdf`, `slfd_colm_er_final.pdf`, and repository `paper/` PDFs are earlier or differently paginated. They are not used to override the final 9-page manuscript. |
| Potential source files | Content scan of Downloads `.tex` found `main(8).tex` with final title and rerender values; `main(4).tex` is the older Step-Level Feedback Distillation draft. Neither is assumed to be byte-equivalent source of the final PDF. No matching `.typ` or poster `.pptx` was found. `Algoverse.zip` contains other research/template materials, not an identified source for Saksham's poster. |
| Reference poster | Downloads `SID 1st author Kiran (2).pdf`, same hash as its `(1)` and unnumbered copies. Used as a visual reference and genuine Algoverse logo source. Extracted image xref 19 plus alpha mask; no replacement logo invented. |
| Repository scientific assets | `paper/generated/format_rerender_gsm8k400_table.{md,tex}`, same-source tables, committed `results/diagnostics/`, teacher evidence pack, historical 1.5B evaluation artifacts. Figure inputs are copied with SHA256 manifests under `data/figure_inputs/`. |
| Evaluation logic | `evaluation/processbench.py`, `experiments/sequence_transfer_ci.py`, `experiments/make_score_ensemble.py`, and `scripts/rerender_labels_processbench_format.py` establish metric direction, clustering, estimands, and transformation semantics. |
| Framing | `PAPER_FRAMING.md` controls the efficient-supervision diagnosis. README is visibly stale and predates the final controlled experiments; not scientific authority. |

## Live public checks

- Official conference page, https://colm.cc/, explicitly lists workshop posters as **34 × 34 inches**. Full page extracted through DeepAPI; request `4e994925-f955-4ce2-a327-3238dfbf179b`, saved in `provenance/scrape_0.json`.
- Official workshop page, https://wdlctc.github.io/efficient-reasoning-2026/, confirms the venue. Retrieved in the same request.
- Public paper, https://openreview.net/forum?id=gKNwyhDxPk, is readable without login, lists the four authors in submitted-poster order, and its abstract confirms the final rerender numbers. Published Aug 7; modified Aug 11; marked Everyone and CC BY 4.0. QR targets this stable forum page. An attempted DeepAPI extraction of its linked PDF returned HTTP 404; public PDF bytes/revision identity are therefore not independently verified. Local final PDF plus artifacts govern the detailed audit.
- Repository https://github.com/edward-lcl/feedback-distillation is confirmed public by the dedicated DeepAPI GitHub endpoint, request `b1c6d22d-682f-4f55-a1f7-71ea1564c9b3`, saved in `provenance/public_repo.json`. Its README contains superseded research framing. The poster therefore prioritizes the paper QR, not a second code QR.

Five distinct DeepAPI searches are retained in `provenance/web_*.json`; full-page results, rather than snippets, establish the dimensions and destination. Account balance fields were removed from saved receipts.

## Unresolved discrepancies

See `EVIDENCE_AUDIT.md` for exact evidence: manuscript overstatement about both ensembles tying gold; positive-control numerical snapshot mismatch; historical versus current 1.5B evaluation slice; no independent runtime ledger for the approximate GPU-hour estimates. The poster does not repeat those unsupported inferences. Submitted affiliation shorthand `CSU` is preserved, not guessed or expanded. A vector Algoverse original would improve logo resolution.
