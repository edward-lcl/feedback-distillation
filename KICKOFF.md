# Kickoff — Where We Are and How We Got Here

A plain-English summary for the team meeting.

## The big idea (one line)

Teach a **small, cheap model** to give the kind of feedback a **big, expensive model**
gives — so we get expert-quality feedback without paying for the expert every time.

## How we got here

1. **Started on model "unlearning"** (making models forget things). Too hard for our
   timeline, so we pivoted.
2. **Moved to CLEAR** — a paper where a strong "expert" and a weak "amateur" model
   both critique an answer. Our plan: use the expert to **train** the amateur, so we
   eventually only need the small model.
3. **It kept breaking**, in two ways:
   - *The code was fragile* — the different training signals fought each other,
     results were unstable, lots of "it doesn't work."
   - *We couldn't prove it worked* — we graded "good feedback" by asking another AI
     to score it. **Reviewers rejected this**: "an AI scoring text is unreliable."
     The easy benchmarks were also maxed out (everyone scored 98%+).
4. That combination stalled the project and we missed several deadlines.

## What changed (the reframe → SLFD)

We kept the good idea (small model learns the expert's feedback) but changed
**what the feedback is about**:

| Old approach | New approach (SLFD) |
|---|---|
| Critique the **whole answer** | Critique **each reasoning step** |
| Grade with an **AI judge** (rejected) | Grade with **real math benchmarks** (an answer key exists) |
| Looked like a tweak on CLEAR | New framing: turn a step-by-step "referee" into a small trainable model |

**The key trick:** while the big teacher *builds the training data*, we give it the
**answer key** so its step-by-step labels are trustworthy. The small student is
trained on those labels but **never sees the answer key** at test time — like a
student who learns from worked solutions, then takes the exam cold.

Because math has a known right answer, we can finally measure success with **real
numbers** (did it catch the first wrong step?) instead of asking an AI "is this good?"
That fixes the exact thing reviewers kept rejecting.

## Two repos now exist — and they don't overlap

The original direction is being continued in a second repo. They are **complementary**,
not competing:

- **MAAH-revive** (original direction): *"Can a small model **fix** its own answer?"*
  — self-critique and rewrite, graded on final-answer accuracy (GSM8K).
- **SLFD** (ours): *"Can a small model **find the broken step**?"*
  — step-level error detection, graded on ProcessBench F1 / first-error accuracy.

Same tool (knowledge distillation of feedback), pointed at **two different stages**:
*refinement* vs. *verification*.

## The one-line pitch

> We took the team's teacher→student feedback idea and pointed it at **individual
> reasoning steps** in math instead of whole answers. That gives us an **objective
> benchmark to prove it works** — fixing the thing reviewers kept rejecting — and a
> cleaner novelty: we're the first to **distill a step-level referee** into a small
> model that needs no answer key.
