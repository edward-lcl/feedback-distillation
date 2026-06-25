# Step-Level Feedback Distillation (SLFD)

**Distilling step-level process-feedback generation from a privileged teacher into a small, ground-truth-free student.**

---

## Overview

Process reward models (PRMs) score the *intermediate steps* of a reasoning
trace, not just the final answer. They have become the dominant signal for
test-time search, re-ranking, and verifier-guided decoding. But in the
existing literature PRMs are used almost exclusively as **filters and
rankers** — a frozen scorer sitting on top of a generator. They are rarely
treated as a *distillation target*: a behavior that a smaller model can be
trained to reproduce.

**Step-Level Feedback Distillation (SLFD)** closes that gap. We take a large
teacher that, *with access to the ground-truth answer*, can produce reliable
per-step labels — a correctness score **and** a natural-language critique
explaining what is wrong — and we distill that step-level feedback behavior
into a small student that runs **without any ground-truth access at test
time**.

The student learns two coupled abilities:

1. **Step scoring** — predict a per-step correctness score (a lightweight
   scoring head on the step-boundary token).
2. **Step critique** — generate the natural-language feedback that explains
   *why* a step is wrong.

### The gap we target

| | PRM literature | SLFD (this work) |
|---|---|---|
| PRM role | filter / ranker / verifier | **distillation target** |
| Teacher signal | scalar step score | score **+** NL critique |
| GT at inference | n/a (frozen scorer) | **student is GT-free** |
| Output | a number | number **and** a written critique |

### Contribution

We distill **step-level feedback generation behavior** — the joint
(score + natural-language critique) signal — from a privileged teacher
(ground-truth access during labeling) into a GT-free small student, and
evaluate it on step-error detection (ProcessBench-style F1 and first-error
accuracy). This is a distillation framing of process feedback, not a new
filtering/ranking scheme.

> **Note:** SLFD does *not* claim to be the first application of knowledge
> distillation to feedback loops. The novelty is the *target*: distilling the
> step-level score **and** critique behavior of a privileged PRM-style teacher
> into a small GT-free student.

---

## Status & key finding (June 2026)

**Live dashboard:** https://feedback-distillation.exe.xyz · **Paper:** [Overleaf](https://www.overleaf.com/2555239245xpdcmsxkrzgx)

**Headline — privilege has a *tractability sweet spot* (at the teacher level).** A privileged
(answer-aware) teacher localizes step errors better only where it *needs* the reference
(can't self-verify) **and** can *use* it (the problem is tractable). And only *rich* privilege
works — a bare final answer is inert; the full worked solution carries the signal. (This is a
property of the teacher's labeling — whether it **distills** into the student is a separate
question, answered below: it doesn't.)

| Difficulty | Δ F1 (full solution − no-GT) | reading |
|---|---|---|
| GSM8K (easy) | ≈ 0 (−0.05) | teacher self-verifies; privilege redundant |
| **MATH (hard)** | **+0.05** (N=400, 95% CI [0.01, 0.09]) | the sweet spot — **significant** |
| OlympiadBench (hardest) | ≈ 0 (−0.03, verified) | teacher can't use the reference |

The MATH gap is significant at N=400 (95% CI excludes zero; the earlier N=150 +0.07 was
underpowered). Within MATH the sweet spot replicates: the gap peaks at the intermediate
levels (L3 +0.11) and collapses at L1 (−0.13, too easy) and the hardest tail.
**Mechanism:** the gain is a *rescue of self-verification failures* — privilege rescues
33% of the errors the no-GT teacher misses while breaking only 12% it already caught, and
the rescue rate falls with reference length (0.37→0.33→0.28). A bare *answer* flips ≈0
predictions everywhere. Cross-family **confirmed** (Qwen-27B teacher: +0.082 on MATH).
Teacher selected by bake-off (**Gemma-4-26b**, F1 0.91).

**Second finding — the teacher's advantage does NOT distill into the student (verified, N=1000).**
With the real Gemma-4 teacher labeling (+0.07 gap confirmed), the privileged and no-GT 1.5B
students are *statistically indistinguishable* verifiers: no-GT ≥ priv on threshold-free
`roc_auc` (0.641 vs 0.631) and on downstream best-of-N re-rank (0.375 vs 0.340, paired McNemar
p=0.14, n.s.); neither beats majority vote. The mechanism: privilege churns ~31% of step labels
but **symmetrically** (1001 vs 956) — the gain is real but *diffuse*, not a clean directional
signal a small student can latch onto. *(The earlier "F1 0.197 vs 0.037 transfers" was a
fixed-threshold artifact and does not reproduce.)*

**Gating caveat / what's next (Phase B).** The student PRM currently *loses to majority vote*
(0.34/0.375 < 0.39), so the null is presently "doesn't distill into a *weak* verifier." Phase B
makes the student competent (data scale + capacity sweep) and re-asks whether privilege transfers
into a verifier that actually works — see [`RUNBOOK_PHASE_B.md`](RUNBOOK_PHASE_B.md) and
[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).

### Pipeline (built &amp; smoke-tested)

| Phase | Command | Output |
|---|---|---|
| 1 — privilege probe | `scripts/run_privilege_probe.sh` | the privilege gap per difficulty tier |
| 2 — student ablation | `scripts/run_student_ablation.sh` | score-only vs score+critique · privileged vs no-GT |
| 3 — verifier (best-of-N) | `experiments/bon_rerank.py` | does the GT-free PRM beat majority vote at test time? |
| analysis | `experiments/evidence_pack.py` | bootstrap CIs · by-level breakdown · flip examples |

Teacher labeling runs against any OpenAI-compatible endpoint (oMLX locally, vLLM on GPU):
set `OMLX_URL` / `OMLX_MODEL` / `OMLX_API_KEY`.

**Paper:** the 4-page draft compiles — Related Work (§1.1–1.4) + 3-tier sweet-spot results
(Table 2). §2.7 (downstream verifier) is now a **verified negative** (no transfer; numbers in
[`results/RESULTS.md`](results/RESULTS.md)) and needs the reframe: lead with the teacher-level
sweet spot, present the student-transfer null + mechanism honestly. Remaining: §2.7 rewrite +
contribution reframe, the `+0.5`→`+0.05` typo, table consolidation, flip cases, OlympiadBench
cell — see [`HANDOFF_HENRY.md`](HANDOFF_HENRY.md). Compiled snapshot: [`paper/SLFD_draft.pdf`](paper/SLFD_draft.pdf).

### Runbooks
- **Next experiments:** [`RUNBOOK_PHASE_B.md`](RUNBOOK_PHASE_B.md) — make the student beat majority vote (data scale + capacity sweep) → re-test transfer. **Start here (B0).**
- **Plan:** [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) — the path from honest null to impactful result + reviewer-question map.
- **Saksham (GPU):** [`HANDOFF_SAKSHAM.md`](HANDOFF_SAKSHAM.md) — current status + pointer to Phase B.
- **Henry (paper):** [`HANDOFF_HENRY.md`](HANDOFF_HENRY.md) — §2.7 verified-negative reframe + remaining paper tasks.

---

## Architecture

```
                    OFFLINE LABELING (teacher, privileged)
  ┌──────────────────────────────────────────────────────────────┐
  │  (problem, solution, gt_answer)                                │
  │            │                                                   │
  │            ▼                                                   │
  │   segment_steps()              ← data/step_segmentation.py     │
  │            │                                                   │
  │            ▼                                                   │
  │   TeacherModel.label_solution()   ← Gemma-4-26b, FROZEN       │
  │     per step → {score, feedback, is_error}                     │
  │     (uses gt_answer — the privileged signal)                  │
  │            │                                                   │
  │            ▼                                                   │
  │   labeled JSONL                ← data/label_pipeline.py        │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    DISTILLATION (student, GT-free)
  ┌──────────────────────────────────────────────────────────────┐
  │   StudentModel.evaluate_step()    ← Qwen2.5-1.5B + LoRA        │
  │     │            │                  + score head              │
  │     │            └──► score_logit (grad) ─► L_score (MSE)      │
  │     └──► feedback tokens          ─► L_feedback_LM (CE)        │
  │                hidden states      ─► L_hidden (cosine)         │
  │            │                                                   │
  │            ▼                                                   │
  │   SLFDTrainer                  ← training/slfd_trainer.py      │
  │   AdaptiveWeightedKDPolicyEMA  ← training/threshold_policy.py  │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    EVALUATION (GT-free at test time)
  ┌──────────────────────────────────────────────────────────────┐
  │   evaluate_processbench()      ← evaluation/processbench.py    │
  │     step-error F1 / precision / recall                        │
  │     first-error-step accuracy                                 │
  └──────────────────────────────────────────────────────────────┘
```

The teacher sees the ground-truth answer **only during offline labeling**.
The student never does — at test time it judges steps from the problem and the
preceding steps alone.

---

## Models

| Role | Model | Params | Trained? | GT access |
|------|-------|--------|----------|-----------|
| **Teacher** | Gemma-4-26b class (bake-off winner; official ckpt for the reported run) | ~26B | **Frozen** | Yes (labeling only) |
| **Student** | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA + score head | 1.5B | **Trainable** | **No** |

> Teacher chosen by a with-GT bake-off (Gemma F1 0.91, zero parse failures, fastest);
> a Qwen-27B teacher reproduces the finding (cross-family). Older drafts named a
> Qwen2.5-7B teacher — superseded.

The teacher is loaded once for offline labeling and never updated. The student
is the only trained component: its base weights (via LoRA), a linear scoring
head, and a hidden-state alignment projection.

Defaults run on Apple Silicon (MPS), CUDA, or CPU with no access gates.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# No HuggingFace login needed for Qwen2.5 defaults
# Models download automatically on first run (~6 GB total: 7B teacher + 1.5B student)
```

### Hardware

- **Apple Silicon (M1–M5)**: runs natively via MPS. 16 GB+ recommended (the
  7B teacher dominates memory during labeling).
- **CUDA GPU**: 16 GB+ VRAM.
- **CPU-only**: works but slow.

---

## Pipeline

**One command for the whole loop** (teacher labeling runs on your local oMLX
server — no 72B download, no API credits):

```bash
export OMLX_API_KEY=...   # your oMLX server key; omit if the server is open
./run_experiment.sh       # download → label → train → save → eval
```

Or run the four stages by hand:

### 0. Download data

```bash
# Eval set — ProcessBench ships GOLD step-error labels (no teacher needed):
python -m scripts.download_data --processbench --config math \
    --output data/processbench_test.jsonl

# Training source — GSM8K problems for the teacher to label:
python -m scripts.download_data --train_source gsm8k --n 300 \
    --output data/raw/gsm8k_train.jsonl
```

### 1. Label steps with the teacher (offline)

```bash
python -m data.label_pipeline \
    --input data/raw/gsm8k_train.jsonl \
    --output data/labeled/gsm8k_labeled.jsonl \
    --max_samples 300 \
    --use_omlx              # call local oMLX instead of loading the 72B teacher
```

Produces labeled JSONL (see [`data/README.md`](data/README.md) for the format).

### 2. Train the student (distillation) + save a checkpoint

```bash
python -m experiments.train_slfd \
    --dataset data/labeled/gsm8k_labeled.jsonl \
    --checkpoint checkpoints/slfd_student.pt \
    --epochs 2 --batch_size 4
```

The labeled JSONL already carries the teacher's per-step score and critique, so
training runs **fully locally on the student alone** — the teacher is not
reloaded. The checkpoint bundles the base model, the trained score head, and the
alignment layer. (`train_slfd` flattens per-solution records into per-step
examples automatically; `data/flatten_labels.py` exposes this standalone.)

### 3. Evaluate on ProcessBench-style step-error detection

```bash
python -m experiments.run_processbench \
    --checkpoint checkpoints/slfd_student.pt \
    --dataset data/processbench_test.jsonl \
    --max_samples 500
```

Reports step-error **F1 / precision / recall** and **first-error-step
accuracy**.

### Local smoke test (Apple Silicon, tiny models)

```bash
./run_local.sh   # 10 synthetic samples, dev-mode 0.5B/1.5B models
```

---

## Repository structure

```
feedback-distillation/
├── models/
│   ├── teacher.py              # TeacherModel — frozen 7B, privileged labeler
│   ├── student.py              # StudentModel — 1.5B + LoRA + score head, GT-free
│   ├── expert_feedback.py      # (legacy) ExpertFeedbackModel
│   ├── amateur_feedback.py     # (legacy) AmateurFeedbackModel
│   └── parsing.py              # answer extraction
├── data/
│   ├── step_segmentation.py    # solution → steps
│   ├── label_pipeline.py       # offline teacher labeling → JSONL
│   └── README.md               # JSONL format + usage
├── training/
│   ├── slfd_trainer.py         # SLFD distillation loop
│   ├── kd_network.py           # (legacy) KD orchestration
│   ├── threshold_policy.py     # AdaptiveWeightedKDPolicyEMA gating
│   ├── losses.py               # L_LM, L_hidden, L_score, L_logit
│   └── loss_config.py          # loss enable/scale config
├── evaluation/
│   ├── processbench.py         # step-error F1 + first-error accuracy
│   └── metrics.py              # BERTScore, ROUGE, BLEU, etc.
├── experiments/
│   ├── run_processbench.py     # main SLFD evaluation
│   ├── run_gsm8k.py            # (legacy) output-level experiment
│   └── run_alpaca.py           # (legacy) reference
├── baselines/                  # CLEAR / CoT / CoD baselines
├── scripts/
│   └── run_all_experiments.py
└── results/                    # outputs (gitignored)
```

---

## Related work

SLFD sits at the intersection of process reward modeling and distillation:

- **CLEAR** — contrastive expert/amateur feedback loops for reasoning.
  ([arXiv:2504.07116](https://arxiv.org/abs/2504.07116))
- **LightReasoner** — small models extracting learning signal from larger ones.
  ([arXiv:2510.07962](https://arxiv.org/abs/2510.07962))
- **Math-Shepherd** — automatic step-level (process) supervision for math
  reasoning without human step annotations.
  ([arXiv:2312.08935](https://arxiv.org/abs/2312.08935))
- **Lightman et al., "Let's Verify Step by Step"** — process supervision beats
  outcome supervision for PRMs.
  ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050))
- **GenPRM / ThinkPRM** — generative process reward models that *reason about*
  step correctness rather than emitting a bare scalar.
- **Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet"** —
  motivates a privileged (GT-aware) teacher signal rather than relying on the
  model's own self-correction.
  ([arXiv:2310.01798](https://arxiv.org/abs/2310.01798))

Where prior PRM work uses step scores to *filter or rank* candidate traces,
SLFD treats the teacher's step-level score-plus-critique behavior as a
**distillation target** for a small, GT-free student.

---

## Citation

```bibtex
@misc{slfd2026,
  title={Step-Level Feedback Distillation: Distilling Process-Feedback Generation into Small Ground-Truth-Free Students},
  author={...},
  year={2026}
}
```
