
# Sprint Run Notes

**Date:** 2026-07-01
**Goal:** Fully address COLM 2026 Workshop Remaining Advice.

## 1. Non-GPU Tasks
- **Novelty Search:** *Done.* Search confirmed no 2025-2026 paper directly invalidates the novelty of our "format match vs source distribution" controlled diagnosis for PRMs. DG-PRM and DataPRM touch on distribution shifts, but our specific ProcessBench-controlled cell remains novel.
- **Efficiency Claim:** *Done.* Added 0.3 GPU-hours vs 15 GPU-hours comparison sentence to the conclusion in `main.tex`.
- **Nomenclature Cleanup:** *Done.* Replaced `gold` with `ProcessBench labels` (and related terms) in `experiments/summarize_phase_b_tables.py` and regenerated the tables.
- **Overleaf Hygiene:** *Done.* Verified no anonymity violations (Phase A, Phase B, feedback-distillation, paths, etc.) exist in `main.tex`.

## 2. Generated-Label Seeds (GPU)
*Done.*
- Privileged BCE (error weight 3): canonical seed 0 plus seeds 1-3 completed.
- No-GT rank-only (balanced batches): canonical seed 0 plus seeds 1-3 completed.

### Results / Contingency Tracking
- Privileged BCE 4-seed aggregate: ROC-AUC 0.5475 (0.5007-0.5883), PR-AUC 0.0966 (0.0881-0.1045), best F1 0.1853 (0.1750-0.1969).
- No-GT rank-only 4-seed aggregate: ROC-AUC 0.6062 (0.5390-0.6755), PR-AUC 0.1254 (0.0911-0.1672), best F1 0.2142 (0.1870-0.2509).
- Both generated-label conditions remain below the GSM8K+OmniMath 8-seed ProcessBench-label ensemble; ROC-AUC gaps are +0.2570 vs privileged BCE and +0.1749 vs no-GT rank-only.
- Tables regenerated in `paper/generated/`; `paper/main.tex` already reports the 0-3 seed aggregates.
