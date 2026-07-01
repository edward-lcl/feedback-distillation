
# Sprint Run Notes

**Date:** 2026-07-01
**Goal:** Fully address COLM 2026 Workshop Remaining Advice.

## 1. Non-GPU Tasks
- **Novelty Search:** *Done.* Search confirmed no 2025-2026 paper directly invalidates the novelty of our "format match vs source distribution" controlled diagnosis for PRMs. DG-PRM and DataPRM touch on distribution shifts, but our specific ProcessBench-controlled cell remains novel.
- **Efficiency Claim:** *Done.* Added 0.3 GPU-hours vs 15 GPU-hours comparison sentence to the conclusion in `main.tex`.
- **Nomenclature Cleanup:** *Done.* Replaced `gold` with `ProcessBench labels` (and related terms) in `experiments/summarize_phase_b_tables.py` and regenerated the tables.
- **Overleaf Hygiene:** *Done.* Verified no anonymity violations (Phase A, Phase B, feedback-distillation, paths, etc.) exist in `main.tex`.

## 2. Generated-Label Seeds (GPU)
*Pending GPU availability.*
- Privileged BCE (error weight 3): seeds 0-3
- No-GT rank-only (balanced batches): seeds 0-3

### Results / Contingency Tracking
*To be filled out based on seed variance.*
