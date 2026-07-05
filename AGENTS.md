# feedback-distillation — agent instructions

SLFD: step-level feedback distillation. Privileged (answer-aware) teacher labels
step errors in math reasoning; we study when that helps the teacher and whether
it distills into a small answer-blind student verifier. Currently in submission
sprint: **COLM 2026 Workshop on Efficient Reasoning, deadline 2026-07-19 AoE.**

## Non-negotiables for ALL agents and contributors

1. **Framing is centralized.** Before writing or editing ANY paper text, related
   work, abstract, Slack summary, or review response, read `PAPER_FRAMING.md`
   and follow it exactly. If you disagree, PR that file — do not introduce a
   different framing in a draft, comment, or commit message. This applies to
   every team member's agents equally.
2. **Numbers trace to artifacts.** Any number in the paper or dashboard must
   come from a committed file under `results/` (or be flagged as pending). Cite
   the artifact path in the PR description when you change a number.
3. **Metrics:** compare PRM cells on threshold-free ROC-AUC / PR-AUC, never on
   F1 at a fixed cutoff (silent score-head collapse banks the base rate — see
   `evaluation/processbench.py` health warnings). If eval prints a health
   warning, that cell's F1 is not a result.
4. **Honest negatives stay honest.** The transfer null is a verified negative
   with a positive control; do not soften it and do not overclaim it.
5. **Push runs to branches** — raw result JSONs (`results/**/processbench_results.json`,
   `per_step_scores.json`), not PDFs/screenshots. No hand-edited conclusions.
6. Data files under `data/` are gitignored by default; use `git add -f` only for
   small JSONL artifacts another teammate needs (say so in the PR).

## Key docs

- `PAPER_FRAMING.md` — canonical framing, must-cites, terminology rules
- `RESEARCH_ROADMAP.md` — phase structure and decision gates
- `HANDOFF_SAKSHAM.md` / `HANDOFF_HENRY.md` — per-person marching orders
- `RUNBOOK_PHASE_B.md` — Phase B commands
- Dashboard: `scripts/build_dashboard.py` (narrative constants at the top) →
  `dashboard/index.html`, CI-rebuilt on push; update the constants when project
  state changes (or use the `dashboard-update` skill in `.claude/skills/`).
