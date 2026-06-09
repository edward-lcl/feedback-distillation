# Data — Step-Level Feedback Distillation

This directory holds the offline teacher-labeling pipeline and the segmented
step-level datasets used to train and evaluate the SLFD student.

## Labeled JSONL format

Each line is one problem with its solution segmented into steps. Every step
carries a teacher-assigned score, natural-language feedback, and an `is_error`
flag (the privileged step-correctness label the teacher derives using the
ground-truth answer):

```json
{"problem": "...", "solution": "...", "gt_answer": "...", "steps": [{"text": "...", "score": 0.9, "feedback": "...", "is_error": false}]}
```

Field semantics:

| Field | Type | Meaning |
|-------|------|---------|
| `problem` | str | The math problem statement |
| `solution` | str | The full multi-step solution (pre-segmentation) |
| `gt_answer` | str | Ground-truth final answer (teacher-only signal) |
| `steps[].text` | str | One reasoning step |
| `steps[].score` | float | Teacher correctness score, `-1.0` (wrong) → `1.0` (correct) |
| `steps[].feedback` | str | NL critique explaining the error (empty if correct) |
| `steps[].is_error` | bool | `true` when the step is likely wrong (`score < 0`) |

## Raw input format

`label_pipeline.py` accepts JSONL with flexible keys (Math-Shepherd /
GSM8K-style). It looks for:

- `problem` or `prompt`
- `solution` or `original_answer`
- `answer` or `gt_answer`

## Running the labeling pipeline

```bash
python -m data.label_pipeline \
    --input data/raw/math_shepherd_sample.jsonl \
    --output data/labeled/math_shepherd_labeled.jsonl \
    --max_samples 500
```

The pipeline:

1. Loads the frozen `TeacherModel` (Qwen2.5-7B-Instruct).
2. Segments each solution into steps via `step_segmentation.segment_steps`.
3. Labels every step with `(score, feedback, is_error)` using the teacher's
   privileged GT access.
4. Writes the labeled JSONL.

## Step segmentation

`step_segmentation.segment_steps` handles three solution styles, in order:

1. Numbered steps (`Step 1:`, `1.`, …)
2. Double-newline paragraphs
3. Single-newline lines

It always returns a non-empty list of step strings.
