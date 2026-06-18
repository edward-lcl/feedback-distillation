# Research Roadmap — from "honest null" to an impactful, publishable result

_Owner: Edward · Updated 2026-06-18 · This is the experiment plan the team executes against._

## The claim we're building toward
> **Privileged, answer-aware supervision improves a PRM *teacher* in a tractability sweet spot — but that advantage does not survive distillation into a small, answer-blind verifier. We show when it transfers, when it doesn't, and why.**

Why it matters: the field assumes "better labeler → better student" (Math-Shepherd, GenPRM, etc.). A real teacher-capability gap that **fails to distill**, with a mechanism and a boundary, is the non-obvious, citable result — and it tells practitioners whether paying compute for GT-aware labeling pipelines is worth it for small verifiers.

## Where we are (verified, N=1000, Gemma-4 labeling)
- ✅ **Teacher sweet spot** validated: GSM8K ≈0 · MATH +0.05 [0.01,0.09] · OlympiadBench ≈0; richness matters; cross-family (Qwen +0.082); mechanism = rescue × tractability.
- ✅ **Student transfer = NULL**: no-GT ≥ priv on `roc_auc` (0.641 vs 0.631) and downstream re-rank (0.373 vs 0.349).
- ⚠️ **Gating weakness:** the student PRM is **below majority vote** (0.37 vs 0.39) and F1 ≈ 0.18. Until the student is a *competent* verifier, the transfer null is confounded with "we trained a weak PRM." **Fixing this is P0.**

---

## Phase A — Diagnose the null (P0, cheap, run now)
One command: `./scripts/run_diagnostics.sh` (after the N=1000 run; teacher env set). _Owner: Saksham._
- **A1 Label agreement** (`experiments/label_agreement.py`): how often do priv vs no-GT teacher labels actually differ? If they agree >95%, the null is "privilege barely changed the training targets" — explanation found.
- **A2 Same-pool paired BoN** (`experiments/bon_paired.py`): both PRMs re-rank ONE shared candidate set + exact McNemar. Replaces the separate-pool `bon_priv`/`bon_nogt`. Reports absolute acc + paired p.
- **A3 Gemma-4 privilege probe**: run the probe through the served Gemma-4 teacher (only the 9b probe was saved) — confirm priv≠no-GT at the labeling teacher.
- **Decision gate A:** which regime are we in — labels-barely-differ, distribution mismatch, or capacity? Routes Phase C.

## Phase B — Make the student a competent PRM (P0, gating)
The transfer question only matters if the student is a useful verifier. _Owner: Edward + Saksham._
- **B1** Scale training data (N_TRAIN 1k → 5k–10k), more epochs/LR sweep.
- **B2** Symbolic MATH answer-matcher in BoN eval (replace the naive string matcher — already flagged); re-run BoN.
- **B3** Consider a stronger/larger student base if 1.5B caps out.
- **Target / decision gate B:** student `prm_rerank > majority_vote` and `roc_auc` competitive with published small PRMs / ProcessBench leaderboard. **If it cannot beat MV even when scaled → reframe as a teacher-only paper** (the sweet-spot finding stands on its own).

## Phase C — Map the transfer boundary (P1, this is what makes it strong)
Turn a flat null into a *boundary* — far more publishable. _Owner: Saksham (runs) + Edward (design)._
- **C1 Student-capacity sweep:** distill priv vs no-GT into 0.5B / 1.5B / 7B. Does privilege start to transfer above some capacity? (If yes → a capacity story, more nuanced and stronger than a flat null.)
- **C2 Train-data scale sweep:** does the priv−nogt transfer gap open as data grows (1k → 10k → 30k)?
- **C3 Positive control (critical):** distill from a **strong** teacher (Gemma-4 + GT) vs a deliberately **weak** teacher (small model, or label-shuffled). Student quality MUST track teacher quality here. If it does → the pipeline is sensitive, so the *privilege* null is real, not insensitivity. If even strong-vs-weak doesn't move the student → the student/eval is the bottleneck (back to Phase B). **This single experiment is what convinces a skeptical reviewer the null means something.**

## Phase D — Rigor / validity (P1, do alongside C)
_Owner: Edward._
- **D1** Multiple seeds + bootstrap CIs on BOTH the teacher gap and the transfer gap; a short power analysis (can we even detect a +0.05-size transfer at N?).
- **D2** Distribution-matched eval: train on ProcessBench-style solutions (not just 9b-generated) so train/eval distributions match — pre-empts the #1 validity attack.
- **D3** Baseline panel: pass@1, majority vote, self-consistency, Math-Shepherd PRM, one off-the-shelf strong PRM. Position the GT-free student honestly.

## Phase E — Generalization & paper (P2)
_Owner: Henry + Edward._
- **E1** Multiple BoN generators (policy-agnostic verifier check).
- **E2** A second dataset/tier if time allows.
- **E3** Paper: consolidate tables, the capacity/boundary figure as the new load-bearing result, mechanism, honest baselines. (See `HANDOFF_HENRY.md`.)

---

## Reviewer / PI questions → which experiment answers it
| Question a reviewer/PI will ask | Answered by |
| :--- | :--- |
| "Your verifier loses to majority vote — why care?" | **Phase B** (make it competent) |
| "Is the null real or just underpowered?" | **D1** (seeds/CIs/power) + **C3** (positive control) |
| "Why doesn't privilege transfer?" | **A1** (label agreement) + **C1** (capacity) + **D2** (distribution) |
| "Does it generalize beyond 1.5B / one dataset / one generator?" | **C1, E1, E2** |
| "Is the teacher sweet-spot itself robust?" | **D1** (seeds/CIs on the teacher gap) |
| "How does this compare to existing PRMs?" | **D3** (baseline panel) |

## Threats to validity & mitigations
- **Weak student confounds the null** → Phase B (P0).
- **Underpowered null** → D1 + C3 positive control.
- **Train/eval distribution shift** → D2.
- **Single seed / size / generator** → C1, D1, E1.
- **Naive answer matcher** → B2 (symbolic checker).

## What success looks like (framing decision tree)
- **Student beats MV + privilege transfers above some capacity/data** → "when does privileged supervision distill" — strongest, most novel paper.
- **Student beats MV + privilege still doesn't transfer (positive control passes)** → "privileged supervision helps teachers but provably does not distill into small verifiers" — clean, counterintuitive, citable.
- **Student can't beat MV even scaled** → drop the downstream story; publish the **teacher-level sweet-spot + mechanism** as a self-contained contribution.

Any of the three is publishable. The first two are *impactful*. The job of Phases A–D is to find out which one we're in — fast.
