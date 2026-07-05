# Handoff — Saksham (GPU box, 2×3090 / 48 GB)

## 2026-07-05 — same-source data is READY; train it on your cluster

**Deadline correction: COLM Efficient Reasoning is now 2026-07-19 AoE** (extended
from Jul 12). Non-archival, double-blind. Framing for anything you or your agents
write: **`PAPER_FRAMING.md` is doctrine** — read it before touching paper text.

The data-prep + labeling steps below (your steps 1–3) are **done** — they ran
locally on Edward's Mac against the local Gemma-4 teacher (no tunnel/endpoint
needed; that's why the oMLX URL request was closed without re-exposing the key):

- Input: `data/gsm8k400_for_labeling.jsonl` (400/400 ProcessBench GSM8K problems
  matched to `openai/gsm8k` references; builder: `scripts/build_gsm8k400_labeling_input.py`)
- Labels: `data/labeled/math_{priv,nogt}_gsm8k400.jsonl` + flattened
  `..._steps.jsonl` (runner: `scripts/run_gsm8k400_same_source_labeling.sh`,
  branch `data/same-source-gsm8k400` — check Slack for status)

**Your move (step 4 of your own plan):** on your cluster, per condition,
seeds 0–3 to match the gold rows:
`scripts/run_gold_scorehead_gate.sh TRAIN_DATASET=data/labeled/math_priv_gsm8k400_steps.jsonl`
(and `math_nogt_...`), **gold-source recipe** — not `bce_ew3`/`rank_bal` —
`RUN_TAG=generated_{priv,nogt}_gsm8k400_to_math1000`, eval on
`data/processbench_math_shuffled.jsonl`. Push raw JSONs to a branch.

Also needed from you (short): one sentence on why your seed-0 rerun diverged
from the canonical checkpoint (0.477 vs 0.550 priv BCE) — hardware/config/seed?
It becomes the run-to-run variance footnote. And if cluster time allows, the
**format-rerender cell** (generated labels re-rendered into ProcessBench-style
format, one training run) completes the provenance×format 2×2 — see
`PAPER_FRAMING.md` §experiments.

## 2026-06-30 — Phase B is merged; here's the one experiment worth running next

Your `saksham/phaseb-capacity-gate-main-integration` branch is merged into `main`
(2026-07-01). Your gold-vs-generated result is the strongest positive evidence in the
project — it's been merged with Henry's teacher-sweet-spot draft into one paper aimed at
**COLM Efficient Reasoning Workshop, deadline 2026-07-12**
(`paper/merged_draft.tex`, plan in `PAPER_MERGE_PLAN.md`). Your framing survives basically
untouched: §6 of the merged paper is your gold-vs-generated result, reframed as the answer
to a question Henry's draft asked but didn't have evidence for ("is the null a capacity
problem or a supervision problem?").

**The one thing worth your GPU time before submission: the missing controlled cell.**
Your own result card already names it — generated teacher labels on the *same*
GSM8K/OmniMath source problems used by the gold-label rows, holding source-distribution
fixed and varying only provenance (gold vs.\ generated). Right now the gold-vs-generated
contrast confounds three things at once (provenance, source distribution, format); this
is the one experiment that actually isolates one variable.

**Important: this is not ready to just launch — it needs a data-prep step first, and I
scoped exactly why.** `results/diagnostics/processbench_{gsm8k,omnimath}_gold_train400_steps.jsonl`
(your existing gold-label training files) only contain the *candidate* solution ProcessBench
asks a judge to evaluate, with per-step `is_error` gold flags — they do **not** contain a
reference worked solution. `data/label_pipeline.py` (the tool that calls the Gemma-4
teacher) needs `{problem, solution, gt_answer, gt_solution}` — it uses `solution` as the
candidate to label and `gt_solution`/`gt_answer` as what the teacher is privileged with.
`data/processbench_gsm8k.jsonl` (400 examples, already in the repo) has `gt_answer` but no
`gt_solution` — same gap for OmniMath, which isn't even downloaded locally
(`scripts/download_data.py` pulls it from `Qwen/ProcessBench`, config `omnimath`).

So the actual next step is a short glue script, not a training run:
1. For each of the 400 GSM8K ProcessBench source problems, match it back to
   `openai/gsm8k` (`main`/`test` split, already referenced in `scripts/download_data.py`)
   by problem text to recover the official reference solution (GSM8K answers include the
   full worked steps before the `####` line). OmniMath may not have an equivalently clean
   reference-solution field — check what `Qwen/ProcessBench`'s omnimath config actually
   ships before assuming this generalizes; GSM8K alone might be the only source where this
   is honestly doable without another data source.
2. Build `{problem, solution: <ProcessBench candidate, steps joined>, gt_answer, gt_solution: <matched GSM8K reference>}`
   JSONL, one row per source problem.
3. Run `data.label_pipeline` twice (`--privilege solution` and `--privilege none`) against
   Edward's Gemma-4 teacher to get `math_priv_gsm8k400.jsonl` / `math_nogt_gsm8k400.jsonl`
   — this step is cheap/sequential (labeling, not generation) and could run on Edward's Mac
   rather than your GPU box.
4. Train with `scripts/run_gold_scorehead_gate.sh TRAIN_DATASET=<the new file>` using the
   **same defaults** the gold-source runs used (don't reach for the generated-label track's
   `bce_ew3`/`rank_bal` variants here — apples-to-apples means matching the gold-source
   recipe, not the best generated-label recipe), `RUN_TAG=generated_priv_gsm8k400_to_math1000`,
   eval on the existing `data/processbench_math_shuffled.jsonl`.
5. Compare against `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{0..3}`
   (holds source fixed, varies provenance) and against
   `results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0` /
   `generated_rank_nogt_to_math1000_qwen3b_seed0` (holds provenance fixed, varies source).

If step 1 turns out to be more than an afternoon (e.g., OmniMath has no usable reference
field), GSM8K-only is still a real, publishable single-cell result — don't block on doing
both sources.

---

**Status: N=1000 RUN COMPLETE & VERIFIED (2026-06-18) — privilege does NOT transfer to the student.** Labeling confirmed to run through the served **Gemma-4** teacher (~32k requests), generation on local `gemma-2-9b`, our threshold-free eval. Result: no-GT student ≥ privileged on `roc_auc` (0.641 vs 0.631) **and** downstream re-rank (0.373 vs 0.349); neither verifier beats majority vote. Teacher-level privilege is still validated — it just doesn't distill into the 1.5B student at this scale. Full numbers: [results/RESULTS.md](results/RESULTS.md). **This is an honest negative — do not report it as "privilege transfers."** Next = the diagnostics (below).

## ✅ Where we are now → next: **Phase B** (see `RUNBOOK_PHASE_B.md`)
Phase A diagnostics are **done** and the null is mechanistic: real +0.07 teacher gap, but it's **diffuse** (~31% of labels churn symmetrically) so it doesn't distill — the two students are statistically indistinguishable (paired McNemar p=0.14).

**Two things gate the paper now, both in `RUNBOOK_PHASE_B.md` (command-first, agent-ready):**
1. **Cheap-first, no new labeling:** re-run the paired BoN at N=1000 (A2 was N=200, p=0.14) **and** the strong-vs-weak-teacher **positive control** — these gate whether we even have a paper. (The N=1000 null is 0.641 vs 0.631 roc_auc on one seed — *not* "validated" until it has a CI and the student beats MV. Lead with the McNemar p.)
2. **Then Phase B — make the student beat majority vote** (it currently loses, 0.34/0.375 < 0.39): scale training data (5k/10k) + capacity sweep (1.5B → 3B → 7B via `STUDENT_MODEL=`). If those don't open the gap, the **distillation-method ablation (B4)** is the mechanism slot. Then re-ask: does privilege transfer into a *competent* verifier?

👉 **Go to `RUNBOOK_PHASE_B.md` and start at B0.** Push raw JSONs to a branch; don't hand-edit conclusions.

### 🖥️ No GPU? Train on Edward's Mac (remote, scoped)
While your GPU box is down, run the **1.5B cheap-first** cells on Edward's Mac. You SSH in as a **restricted `slfd` user** — no sudo, no access to his files — into a self-contained working copy. **The capacity sweep (3B/7B, B2) is NOT possible on the Mac (won't fit 48 GB) — that waits for your GPUs.**

**Connect (one-time setup):**
1. Send Edward your SSH **public** key (`cat ~/.ssh/id_ed25519.pub`) so he can add it to the `slfd` account.
2. `brew install cloudflared`, then add to your `~/.ssh/config`:
   ```
   Host edmac
     HostName ssh.elcl.systems
     User slfd
     ProxyCommand cloudflared access ssh --hostname %h
   ```
3. `ssh edmac` → a browser opens for Cloudflare Access; authenticate with your **terpmail email** (the only one allowed). You land in the restricted `slfd` account.

```bash
# once connected:
cd /Users/Shared/slfd/feedback-distillation
export HF_HOME=/Users/Shared/slfd/hf_cache HF_HUB_OFFLINE=1   # models are pre-cached; no download/network
git pull                                                       # get latest main
tmux new -s slfd                                               # so the run survives disconnects

# cheap-first: NO oMLX, NO teacher load — pure train/eval on the existing labels:
REUSE_LABELS=1 ABLATION=soft    N_EVAL=1000 EPOCHS=2 ./scripts/run_student_ablation.sh
REUSE_LABELS=1 ABLATION=verdict N_EVAL=1000 EPOCHS=2 ./scripts/run_student_ablation.sh
# logit_kd uses a local same-family teacher (also cached): add KD_TEACHER=Qwen/Qwen2.5-0.5B-Instruct
python -m experiments.transfer_ci \
  --priv results/ablation/priv_critique/per_step_scores.json \
  --nogt results/ablation/nogt_critique/per_step_scores.json --n_boot 10000

# push raw JSONs to a branch (this copy is yours; don't touch Edward's repo):
git checkout -b saksham/macrun-$(date +%m%d); git add results/ && git commit -m "..."; git push -u origin HEAD
```
Notes: this copy already has the real `data/labeled/math_{priv,nogt}.jsonl` + eval set, the venv, and the Qwen models cached — so cheap-first needs **no network and no oMLX**. Only *fresh-data* generation needs the oMLX key (`./scripts/run_single_box.sh`, key from Edward); prefer `REUSE_LABELS` to avoid loading the teacher.

(Historical context — the original reorientation + diagnostics — is retained below.)

_Goal: reproduce the privilege × difficulty result with an **official** Gemma checkpoint at scale. Phase 1 (the probe) is below; Phase 2 (the full student run + ablations — the paper result) is at the bottom, now also ready._

---

## ⚠️ READ FIRST — reorientation (2026-06-17, after your first end-to-end run)
> ⏩ **HISTORY — superseded.** This reorientation is complete: the metric was fixed, the re-score ran, and the diagnostics are done (the result is a verified null). **Current marching orders are the Phase B section at the top of this file → `RUNBOOK_PHASE_B.md`.** Kept below as the record of how we got here.

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
> 🔗 **This re-score is now on the paper's critical path.** Henry's draft has a §2.7 downstream-verifier section held as an honest stub — it can't claim a real result until your threshold-free re-score (ROC/PR-AUC) confirms privilege transfers to the student. So this is the one item unblocking that section; please push the JSONs as soon as it's run.
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
> ⚠️ **Superseded by the READ FIRST topology.** You do **not** need to serve the privileged teacher anymore — point `OMLX_URL` at Edward's live teacher (`https://teacher.elcl.systems/v1`). The steps below are only for: (a) running a *local* small model for the **generation** step (`GEN_OMLX_URL`), or (b) the single-endpoint fallback if you'd rather use `gemma-2-27b-it-bnb-4bit` locally as the teacher too.

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
