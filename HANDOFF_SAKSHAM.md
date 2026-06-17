# Handoff — Saksham (GPU box, 2×3090 / 48 GB)

**Status: FIRST RUN IN — REORIENTING (2026-06-17).** Phase 1 (probe) replicates; Phases 2–3 ran end-to-end on the `gemma-2-9b-it` fallback, but the headline is **NOT validated** — the reported Phase 2 gap is a fixed-threshold metric artifact (see READ FIRST below). Raw metrics: [results/RESULTS.md](results/RESULTS.md). Don't report any phase as "done" until the threshold-free re-score confirms it.

_Goal: reproduce the privilege × difficulty result with an **official** Gemma checkpoint at scale. Phase 1 (the probe) is below; Phase 2 (the full student run + ablations — the paper result) is at the bottom, now also ready._

---

## ⚠️ READ FIRST — reorientation (2026-06-17, after your first end-to-end run)

Great first pass — Phase 1 (probe) replicates and the pipeline runs end-to-end. Before we scale, three things, **in this order**:

**1. Always commit + push your run to a branch — don't send only a PDF.**
We need the raw result JSONs (they carry precision/recall/AUC that a summary table drops). After any run:
```bash
git checkout -b saksham/run-$(date +%m%d)        # or any descriptive name
git add results/ checkpoints/*.json data/*_shuffled.jsonl   # JSONs + configs, NOT big .pt weights
git commit -m "Phase 1/2/3 run: <teacher>, N_TRAIN=.. N_EVAL=.."
git push -u origin HEAD
```
Then drop the branch name in the channel. (Skip the multi-GB `.pt` weights; the `results/ablation/*/processbench_results.json` files are what we need.)

**2. Fix the metric before scaling — your Phase 2 gap is currently a threshold artifact, not capability.**
Charlotte's catch: `nogt_critique` F1 cratered to 0.037 but its `first_error_acc` (0.435) is identical to the privileged cells. That's not a contradiction — it's a metric artifact:
- The decision rule is a **fixed cutoff** (`score_logit < 0` ⇒ "error"). The no-GT student's score head is shifted so it almost never crosses 0 → predicts "no error" everywhere → recall→0 → **F1 collapses**.
- But `first_error_acc` counts a sequence correct when predicted-first == true-first, and "no error anywhere" (`pred_first=None`) **trivially matches every error-free sequence**. ProcessBench-MATH is ~43% error-free, so a silent model banks ~0.435 *for free*. All three cells sit at the base rate; only F1 sees the collapse.

So the 0.037 → 0.197 "privilege transfers" headline is, as measured, mostly **calibration at a fixed threshold**, not demonstrated capability transfer. **Re-score the existing checkpoints threshold-free — no retraining needed.** `evaluation/processbench.py` now also emits:
- `roc_auc`, `pr_auc` — ranking quality of the raw scores, no cutoff → **this is the apples-to-apples comparison**
- `error_recall`, `clean_specificity`, `pred_error_rate` — exposes silent collapse (`pred_error_rate≈0` ⇒ degenerate cell)
- `first_error_acc_errs`, `clean_seq_acc` — first-error accuracy split so a do-nothing model can't bank the base rate

Just re-run the eval step on your saved checkpoints (no relabel/retrain):
```bash
for d in priv_critique priv_scoreonly nogt_critique; do
  ./.venv/bin/python -m experiments.run_processbench --checkpoint checkpoints/$d.pt \
      --dataset data/processbench_math_shuffled.jsonl --max_samples 400 \
      --results_dir results/ablation/$d
done
```
The runner now prints a table with these columns. **Decision rule for the thesis: privilege transfers iff `roc_auc(priv) > roc_auc(nogt)` by a meaningful margin.** If the no-GT cell is just silent (`pred_error_rate≈0`, `error_recall≈0`), report that — the fix may be threshold calibration, not "privilege."

> 🛟 **The eval now self-checks.** `run_processbench` prints a loud `⚠️ EVAL HEALTH WARNING` (to stderr) and writes a `warnings` field into the JSON whenever a cell collapses (predicts ~no errors, or one-class slice). If you see that banner, **don't report that cell's F1 as a result** — it's a calibration artifact; use `roc_auc`. No banner + `✓ eval health OK` means the cell is well-behaved.

**3. Teacher topology — STOP trying to run Gemma-4 on your box.**
Gemma-4-26B-A4B has no working vLLM/CUDA path on 2×3090 right now (3D MoE experts vLLM can't tensor-parallel, custom GELU breaks Marlin, single 47 GB safetensors, on-the-fly bnb stripped). That's a real dead end — don't burn more time on it, and don't chase an "unquantized for validity" run (our headline came from a **4-bit** MLX teacher; 4-bit is fine). The id `gemma-4-26b-a4b-it` in the old handoff was an MLX-local alias, not a servable HF repo — our mistake.

Instead we **split the two teacher-dependent steps across two endpoints** (the code now supports this directly):

| Step | Endpoint | Runs on |
|---|---|---|
| **Generation** (expensive, 512-tok traces) | `GEN_OMLX_URL` / `GEN_OMLX_MODEL` | a **small vLLM model on YOUR 3090s** (e.g. `gemma-2-9b-it` or a 7B — well-supported, continuous batching) |
| **Labeling** (cheap, short scores — needs the privileged teacher) | `OMLX_URL` / `OMLX_MODEL` / `OMLX_API_KEY` | **Edward's MLX Gemma-4**, served over a tunnel |
| Student train + eval | — | your 3090s (trivial, 1.5B) |

Labeling is **sequential** (one request at a time), so it won't overload the served teacher. Env to export on your box:
```bash
# expensive generation → your local small model (vLLM, OpenAI-compatible)
export GEN_OMLX_URL=http://localhost:8000/v1
export GEN_OMLX_MODEL=google/gemma-2-9b-it
# cheap privileged labeling → Edward's served teacher (LIVE)
export OMLX_URL=https://teacher.elcl.systems/v1
export OMLX_MODEL=<exact served MLX model id — ask Edward>
export OMLX_API_KEY=<key from Edward, out-of-band>
export OMLX_TIMEOUT=600          # remote teacher; generous headroom
N_TRAIN=1000 N_EVAL=400 EPOCHS=2 ./scripts/run_student_ablation.sh
```
- **Student:** `Qwen/Qwen2.5-1.5B-Instruct` (unchanged — correct).
- Single-endpoint mode still works (set only `OMLX_URL`/`OMLX_MODEL` and both steps use it) — e.g. if you'd rather use `gemma-2-27b-it-bnb-4bit` as the teacher too. That's scientifically fine; cross-teacher (Qwen-27B +0.082) already showed the effect is teacher-agnostic. Just note which teacher made the labels.

**Sequencing:** (2) re-score current checkpoints threshold-free → confirm whether the gap survives → (3) point labeling at Edward's served teacher (or `gemma-2-27b` locally) → only then scale to N=1000. Don't 10× a metric we haven't trusted yet.

---

## 0. Get the code
```bash
git clone https://github.com/edward-lcl/feedback-distillation.git
cd feedback-distillation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CUDA torch, transformers, datasets, etc.
pip install vllm                          # the teacher server
```

## 1. Serve a teacher (OpenAI-compatible endpoint)
The code talks to any OpenAI-compatible `/v1` endpoint, so vLLM drops in. On 48 GB, a full-precision 27B won't fit — use a **4-bit/AWQ 27B** (tensor-parallel across both GPUs) or fall back to `gemma-2-9b-it`:
```bash
# example — pick a real checkpoint your box can fit:
vllm serve google/gemma-2-27b-it --quantization awq --tensor-parallel-size 2 \
     --port 8000 --api-key sk-local
# simpler/smaller fallback:
# vllm serve google/gemma-2-9b-it --tensor-parallel-size 2 --port 8000 --api-key sk-local
```

## 2. Point the client at it
```bash
export OMLX_URL=http://localhost:8000/v1
export OMLX_MODEL=google/gemma-2-27b-it    # must match the served model id exactly
export OMLX_API_KEY=sk-local               # omit if you served with no key
```

## 3. Run the experiment (one command)
```bash
./scripts/run_privilege_probe.sh           # PB_CONFIG=math  N=150  SEED=0 by default
```
It downloads ProcessBench MATH (GT answer + solution joined), shuffles (seed 0), and runs the 3-condition probe (no-GT / +answer / +full-solution).

## 4. What to report
From `results/teacher_eval_math_<model>/privilege_probe.json`:
- **`gap_solution_f1`** — clearly **> 0** confirms the result holds with the official checkpoint.
- `gap_answer_f1` — expected ≈ 0 (a bare answer is inert).

**Reference (local runs, MATH N=150):** Gemma-4-26b solution gap **+0.07**, Qwen-27B **+0.082**. If your official Gemma lands in that ballpark, the headline is locked with a reproducible, official model.

## Optional / nice-to-have
- Harder set: `PB_CONFIG=olympiadbench ./scripts/run_privilege_probe.sh` (may widen the gap).
- Higher N for tighter CIs: `N=300 ./scripts/run_privilege_probe.sh`.
- A second official family (e.g. a Qwen2.5/3 instruct) for an extra cross-family point.

## Phase 2 — the full student run [RAN — pending threshold-free re-score]
*Context: the three student ablations trained and evaluated end-to-end (see [results/RESULTS.md](results/RESULTS.md) for the F1/FEA table). But the reported privilege-transfer gap is a fixed-threshold artifact (READ FIRST §2) — re-score on ROC/PR-AUC before treating it as a result.*
The trainer is fixed (real LoRA + boundary-token score head), so the GT-free student pipeline is turnkey. One command runs both headline ablations:
```bash
N_TRAIN=300 N_EVAL=400 EPOCHS=2 ./scripts/run_student_ablation.sh
```
It labels MATH train data twice (privileged solution + no-GT), trains the cells (score-only vs score+critique), and evals on ProcessBench MATH. Report the printed table:
- **priv_critique vs priv_scoreonly** — does distilling the NL critique help?
- **priv_critique vs nogt_critique** — does the teacher's privilege transfer to the student?

Tiny local smoke first (no GPU teacher needed): `DEV=1 GEN_BACKEND=local N_TRAIN=4 N_EVAL=10 ./scripts/run_student_ablation.sh`

## Phase 3 — use the PRM as a test-time verifier [PRELIMINARY — did not beat majority vote]
*Context: `bon_rerank.py` (now ThreadPool-parallel) ran downstream — but prm_rerank 32.0 came in **below** majority_vote 32.5, and a 3pp lift over pass@1 on N=200 is within noise. Needs the symbolic MATH checker + larger N + CIs before it's a result. See [results/RESULTS.md](results/RESULTS.md).*
Once Phase 2 yields a student checkpoint, measure what the verifier is worth downstream:
```bash
OMLX_MODEL=<generator> ./.venv/bin/python -m experiments.bon_rerank \
    --dataset data/processbench_math_shuffled.jsonl \
    --checkpoint checkpoints/priv_critique.pt --n 8 --max_samples 200
```
Samples N candidate solutions per problem, re-ranks with the student PRM, and reports
**pass@1 / majority_vote / prm_rerank / oracle_pass@N**. Headline: `prm_rerank > majority_vote`
means the GT-free verifier improves reasoning at inference (a fraction of the teacher's cost).
(NOTE: answer-matching is the simple numeric/string matcher — swap in a symbolic MATH checker for final numbers.)
