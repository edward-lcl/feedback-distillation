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
  │   TeacherModel.label_solution()   ← Qwen2.5-7B, FROZEN         │
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
| **Teacher** | `Qwen/Qwen2.5-7B-Instruct` | 7B | **Frozen** | Yes (labeling only) |
| **Student** | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA + score head | 1.5B | **Trainable** | **No** |

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

### 1. Label steps with the teacher (offline)

```bash
python -m data.label_pipeline \
    --input data/raw/math_shepherd_sample.jsonl \
    --output data/labeled/math_shepherd_labeled.jsonl \
    --max_samples 500
```

Produces labeled JSONL (see [`data/README.md`](data/README.md) for the format).

### 2. Train the student (distillation)

```python
from models.student import StudentModel
from models.teacher import TeacherModel
from training.slfd_trainer import SLFDTrainer

student = StudentModel()
teacher = TeacherModel()
trainer = SLFDTrainer(student, teacher, dataset)   # dataset = list of per-step dicts
trainer.train(epochs=2, batch_size=4)
```

### 3. Evaluate on ProcessBench-style step-error detection

```bash
python -m experiments.run_processbench \
    --student_model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint checkpoints/slfd_student.pt \
    --dataset data/processbench_test.jsonl \
    --max_samples 500
```

Reports step-error **F1 / precision / recall** and **first-error-step
accuracy**.

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
