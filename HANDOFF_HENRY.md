# Handoff — Henry (research / analysis + paper)

_Last updated 2026-06-15. Async-friendly, no compute needed. Dashboard: https://feedback-distillation.exe.xyz_

## The one thing to internalize

We validated the project's core claim today, and it has a sharp, defensible spine:

> **Privileged supervision helps a step-error-detecting teacher only when the problem is hard *and* the privilege is rich.**
> - GSM8K (easy): privilege gap ≈ 0 at every level — a strong teacher already self-verifies grade-school arithmetic.
> - MATH (hard): a bare final *answer* still does nothing, but the full worked *solution* lifts error recall 0.58 → 0.65 (gap **+0.07**, N=150, Gemma teacher).

That's the paper. Everything below turns it into a submission.

## What's settled — do NOT relitigate

| | choice |
|---|---|
| Teacher | Gemma-4-26b class (official ckpt for the reported run) |
| Dataset | ProcessBench **MATH** (GSM8K too saturated to show the effect) |
| Privilege signal | **full worked solution** (a bare answer is inert) |
| Eval | ProcessBench-style first-error F1, GT-free at student test time |

## Your mission (in priority order)

**A. Related work & positioning (write).** Frame our contribution against process-reward-model literature, which treats step scoring as a *scalar* signal and the PRM as a frozen *scorer/ranker*. Our novelty: (1) the teacher's step **score+critique behavior as a distillation target**, and (2) the **privilege×difficulty×richness** result. Core papers:
- Math-Shepherd (arXiv:2312.08935) — automatic step supervision
- Lightman et al., "Let's Verify Step by Step" (arXiv:2305.20050) — process > outcome supervision
- GenPRM / ThinkPRM — *generative* PRMs that reason about step correctness
- CLEAR (arXiv:2504.07116) — contrastive expert/amateur feedback (our prior framing)
- LightReasoner (arXiv:2510.07962) — small models extracting signal from large ones

**B. Results narrative (write).** Turn the privilege×difficulty data into the paper's results story: privilege buys signal exactly where self-verification fails, and only when rich enough. The GSM8K-vs-MATH contrast is the key figure.

**C. Interpret the evidence pack (light analysis).** Edward will generate two artifacts for you:
- **By-level breakdown** — does the MATH solution-gap grow with MATH difficulty level (1→5)? (Predicts: gap widens with level.)
- **Flip examples** — concrete steps where solution-privilege catches an error that no-GT misses. Categorize the error types; pick 2–3 illustrative ones for the paper.

## Inputs you have

- Dashboard (live): https://feedback-distillation.exe.xyz
- Result JSONs in `results/teacher_eval*/privilege_probe.json` (GSM8K, MATH N=50/N=150)
- Eval data: `data/processbench_math_shuffled.jsonl` (GT answer + solution joined)
- Evidence pack: _to be delivered (by-level breakdown + flip examples)_

## Deliverable

A paper skeleton (Overleaf or Google Doc — your call) with **Related Work** and the **privilege×difficulty results section** drafted. Target venue is flexible (COLM/AAAI/workshop) — quality over the date.

## Open question for you to weigh in on

Cross-teacher replication (Qwen-27B) is running now — if it confirms the pattern holds beyond Gemma, the difficulty×richness claim is family-independent. If it diverges, we need to discuss framing. Check the dashboard's "cross-family check" card.
