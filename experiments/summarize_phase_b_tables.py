"""Generate paper-ready Phase B tables from local result JSON artifacts."""
import argparse
import json
import os
import statistics as stats
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class RunGroup:
    label: str
    paths: list[str]


@dataclass(frozen=True)
class BudgetRun:
    label: str
    max_steps: str
    path: str


MAIN_GROUPS = [
    RunGroup(
        "GSM8K ProcessBench gold -> MATH1000",
        [
            f"results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/processbench_results.json"
            for seed in range(4)
        ],
    ),
    RunGroup(
        "OmniMath ProcessBench gold -> MATH1000",
        [
            f"results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/processbench_results.json"
            for seed in range(4)
        ],
    ),
    RunGroup(
        "Qwen2.5-Math-7B-PRM800K public baseline",
        ["results/diagnostics/qwen_prm800k_math1000/processbench_results.json"],
    ),
    RunGroup(
        "Generated privileged teacher labels, BCE",
        ["results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0/processbench_results.json"],
    ),
    RunGroup(
        "Generated no-GT teacher labels, rank loss",
        ["results/diagnostics/generated_rank_nogt_to_math1000_qwen3b_seed0/processbench_results.json"],
    ),
]

BOUNDARY_GROUPS = [
    RunGroup(
        "OlympiadBench ProcessBench gold -> MATH1000",
        [
            f"results/diagnostics/processbench_olympiadbench_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/processbench_results.json"
            for seed in range(2)
        ],
    ),
]

ENSEMBLE_GROUPS = [
    RunGroup(
        "GSM8K 4-seed mean score",
        [
            f"results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/per_step_scores.json"
            for seed in range(4)
        ],
    ),
    RunGroup(
        "OmniMath 4-seed mean score",
        [
            f"results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/per_step_scores.json"
            for seed in range(4)
        ],
    ),
    RunGroup(
        "GSM8K+OmniMath 8-seed mean score",
        [
            *[
                f"results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/per_step_scores.json"
                for seed in range(4)
            ],
            *[
                f"results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}/per_step_scores.json"
                for seed in range(4)
            ],
        ],
    ),
    RunGroup(
        "GSM8K 1500-step 3-seed mean score",
        [
            f"results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{seed}_ms1500/per_step_scores.json"
            for seed in range(3)
        ],
    ),
]

CI_COMPARISONS = [
    (
        "GSM8K+OmniMath 8-seed mean score",
        "Generated privileged teacher labels, BCE",
        "results/diagnostics/gsm8k_omnimath_8seed_mean_vs_teacher_bce_priv_sequence_ci.json",
    ),
    (
        "GSM8K+OmniMath 8-seed mean score",
        "Generated no-GT teacher labels, rank loss",
        "results/diagnostics/gsm8k_omnimath_8seed_mean_vs_best_generated_rank_nogt_sequence_ci.json",
    ),
    (
        "GSM8K+OmniMath 8-seed mean score",
        "OmniMath gold seed 3",
        "results/diagnostics/gsm8k_omnimath_8seed_mean_vs_omnimath_seed3_sequence_ci.json",
    ),
    (
        "Qwen2.5-Math-7B-PRM800K",
        "GSM8K+OmniMath 8-seed mean score",
        "results/diagnostics/qwen_prm800k_vs_gsm8k_omnimath_8seed_mean_sequence_ci.json",
    ),
    (
        "GSM8K 1500-step 3-seed mean score",
        "Generated no-GT teacher labels, rank loss",
        "results/diagnostics/gsm8k_ms1500_3seed_mean_vs_best_generated_rank_nogt_sequence_ci.json",
    ),
    (
        "Qwen2.5-Math-7B-PRM800K",
        "GSM8K 1500-step 3-seed mean score",
        "results/diagnostics/qwen_prm800k_vs_gsm8k_ms1500_3seed_mean_sequence_ci.json",
    ),
]

CALIBRATED_ENSEMBLE_PATH = (
    "results/diagnostics/math1000_calibrated_threshold_metrics_cal200_eval800_ensembles.json"
)

TRAINING_BUDGET_RUNS = [
    BudgetRun(
        "GSM8K gold seed 0",
        "500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json",
    ),
    BudgetRun(
        "GSM8K gold seed 0",
        "1500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0_ms1500/processbench_results.json",
    ),
    BudgetRun(
        "GSM8K gold seed 1",
        "500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json",
    ),
    BudgetRun(
        "GSM8K gold seed 1",
        "1500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1_ms1500/processbench_results.json",
    ),
    BudgetRun(
        "GSM8K gold seed 2",
        "500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed2/processbench_results.json",
    ),
    BudgetRun(
        "GSM8K gold seed 2",
        "1500",
        "results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed2_ms1500/processbench_results.json",
    ),
    BudgetRun(
        "OmniMath gold seed 0",
        "500",
        "results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json",
    ),
    BudgetRun(
        "OmniMath gold seed 0",
        "1500",
        "results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0_ms1500/processbench_results.json",
    ),
]

TRAINING_BUDGET_CI_COMPARISONS = [
    (
        "GSM8K gold seed 0, 1500 steps",
        "GSM8K gold seed 0, 500 steps",
        "results/diagnostics/gsm8k_seed0_ms1500_vs_ms500_sequence_ci.json",
    ),
    (
        "OmniMath gold seed 0, 1500 steps",
        "OmniMath gold seed 0, 500 steps",
        "results/diagnostics/omnimath_seed0_ms1500_vs_ms500_sequence_ci.json",
    ),
    (
        "GSM8K gold seed 1, 1500 steps",
        "GSM8K gold seed 1, 500 steps",
        "results/diagnostics/gsm8k_seed1_ms1500_vs_ms500_sequence_ci.json",
    ),
    (
        "GSM8K gold seed 2, 1500 steps",
        "GSM8K gold seed 2, 500 steps",
        "results/diagnostics/gsm8k_seed2_ms1500_vs_ms500_sequence_ci.json",
    ),
    (
        "GSM8K gold seed 0, 1500 steps",
        "Generated no-GT teacher labels, rank loss",
        "results/diagnostics/gsm8k_seed0_ms1500_vs_best_generated_rank_nogt_sequence_ci.json",
    ),
    (
        "Qwen2.5-Math-7B-PRM800K",
        "GSM8K gold seed 0, 1500 steps",
        "results/diagnostics/qwen_prm800k_vs_gsm8k_seed0_ms1500_sequence_ci.json",
    ),
    (
        "GSM8K gold seed 0, 1500 steps",
        "OmniMath gold seed 3, 500 steps",
        "results/diagnostics/gsm8k_seed0_ms1500_vs_omnimath_seed3_sequence_ci.json",
    ),
]

TRAINING_BUDGET_CALIBRATED_PATH = (
    "results/diagnostics/math1000_calibrated_threshold_metrics_cal200_eval800_gsm_ms1500_budget.json"
)

METRICS = [
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
    ("Best F1", "best_f1"),
    ("Fixed F1", "f1"),
    ("Pred error rate", "pred_error_rate"),
]


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(vals: list[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{stats.mean(vals):.4f} ({min(vals):.4f}-{max(vals):.4f})"


def seeds_for(group: RunGroup) -> str:
    if len(group.paths) == 1:
        return "0"
    if len(group.paths) == 2:
        return "0,1"
    return f"0-{len(group.paths) - 1}"


def summarize_group(group: RunGroup) -> list[str]:
    rows = [load_json(path) for path in group.paths]
    return [group.label, seeds_for(group)] + [
        fmt([float(row[key]) for row in rows]) for _, key in METRICS
    ]


def markdown_table(groups: list[RunGroup]) -> str:
    headers = ["Training source", "Seeds"] + [label for label, _ in METRICS]
    aligns = ["---", "---"] + ["---:"] * len(METRICS)
    rows = [summarize_group(group) for group in groups]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def budget_markdown_table(runs: list[BudgetRun]) -> str:
    headers = ["Run", "Max steps"] + [label for label, _ in METRICS]
    aligns = ["---", "---:"] + ["---:"] * len(METRICS)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for run in runs:
        row = load_json(run.path)
        vals = [f"{float(row[key]):.4f}" for _, key in METRICS]
        lines.append("| " + " | ".join([run.label, run.max_steps, *vals]) + " |")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def latex_table(groups: list[RunGroup], caption: str, label: str) -> str:
    headers = ["Training source", "Seeds"] + [name for name, _ in METRICS]
    col_spec = "ll" + "r" * len(METRICS)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in [summarize_group(group) for group in groups]:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def budget_latex_table(runs: list[BudgetRun], caption: str, label: str) -> str:
    headers = ["Run", "Max steps"] + [name for name, _ in METRICS]
    col_spec = "lr" + "r" * len(METRICS)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for run in runs:
        row = load_json(run.path)
        vals = [f"{float(row[key]):.4f}" for _, key in METRICS]
        lines.append(
            " & ".join(latex_escape(cell) for cell in [run.label, run.max_steps, *vals])
            + " \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def load_scores(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    return np.asarray(data["y_true"], dtype=np.int64), np.asarray(data["y_score"], dtype=np.float64)


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1s = []
    for precision, recall in zip(precisions, recalls):
        denom = precision + recall
        f1s.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return float(max(f1s))


def summarize_ensemble(group: RunGroup) -> list[str]:
    y_arrays = []
    score_arrays = []
    for path in group.paths:
        y_true, y_score = load_scores(path)
        y_arrays.append(y_true)
        score_arrays.append(y_score)
    if not all(np.array_equal(y_arrays[0], y) for y in y_arrays):
        raise AssertionError(f"label mismatch in ensemble group {group.label}")
    y_true = y_arrays[0]
    y_score = np.mean(score_arrays, axis=0)
    return [
        group.label,
        str(len(group.paths)),
        f"{roc_auc_score(y_true, y_score):.4f}",
        f"{average_precision_score(y_true, y_score):.4f}",
        f"{best_f1(y_true, y_score):.4f}",
    ]


def ensemble_markdown_table(groups: list[RunGroup]) -> str:
    headers = ["Ensemble", "Members", "ROC-AUC", "PR-AUC", "Best F1"]
    aligns = ["---", "---:"] + ["---:"] * 3
    rows = [summarize_ensemble(group) for group in groups]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ensemble_latex_table(groups: list[RunGroup], caption: str, label: str) -> str:
    headers = ["Ensemble", "Members", "ROC-AUC", "PR-AUC", "Best F1"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in [summarize_ensemble(group) for group in groups]:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def summarize_ci(label_a: str, label_b: str, path: str) -> list[str]:
    data = load_json(path)
    lower, upper = data["ci95"]
    return [
        label_a,
        label_b,
        f"{float(data['gap_model_a_minus_model_b']):+.4f}",
        f"[{float(lower):.4f}, {float(upper):.4f}]",
        f"{float(data['p_two_sided']):.4f}",
    ]


def ci_markdown_table(comparisons: list[tuple[str, str, str]]) -> str:
    headers = ["Model A", "Model B", "ROC-AUC gap", "95% CI", "p"]
    aligns = ["---", "---", "---:", "---", "---:"]
    rows = [summarize_ci(*comparison) for comparison in comparisons]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ci_latex_table(
    comparisons: list[tuple[str, str, str]], caption: str, label: str
) -> str:
    headers = ["Model A", "Model B", "ROC-AUC gap", "95% CI", "p"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in [summarize_ci(*comparison) for comparison in comparisons]:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def calibrated_rows(path: str) -> list[list[str]]:
    payload = load_json(path)
    rows = []
    for row in payload["results"]:
        ev = row["evaluation"]
        rows.append([
            row["name"],
            f"{float(ev['f1']):.4f}",
            f"{float(ev['roc_auc']):.4f}",
            f"{float(ev['pr_auc']):.4f}",
            f"{float(ev['precision']):.4f}",
            f"{float(ev['recall']):.4f}",
            f"{float(ev['pred_error_rate']):.4f}",
            f"{float(ev['first_error_acc']):.4f}",
        ])
    return rows


def calibrated_markdown_table(path: str) -> str:
    headers = [
        "Model",
        "Calibrated F1",
        "Eval ROC-AUC",
        "Eval PR-AUC",
        "Precision",
        "Recall",
        "Pred error rate",
        "First-error acc",
    ]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in calibrated_rows(path))
    return "\n".join(lines)


def calibrated_latex_table(path: str, caption: str, label: str) -> str:
    headers = [
        "Model",
        "Calibrated F1",
        "Eval ROC-AUC",
        "Eval PR-AUC",
        "Precision",
        "Recall",
        "Pred error rate",
        "First-error acc",
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in calibrated_rows(path):
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="paper/generated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "phase_b_main_table.md": markdown_table(MAIN_GROUPS),
        "phase_b_boundary_table.md": markdown_table(BOUNDARY_GROUPS),
        "phase_b_main_table.tex": latex_table(
            MAIN_GROUPS,
            "Full ProcessBench-MATH transfer results. Parentheses show seed ranges.",
            "tab:phase-b-main-transfer",
        ),
        "phase_b_boundary_table.tex": latex_table(
            BOUNDARY_GROUPS,
            "OlympiadBench boundary diagnostic. Parentheses show seed ranges.",
            "tab:phase-b-olympiad-boundary",
        ),
        "phase_b_ensemble_table.md": ensemble_markdown_table(ENSEMBLE_GROUPS),
        "phase_b_ensemble_table.tex": ensemble_latex_table(
            ENSEMBLE_GROUPS,
            "Optional score-averaging diagnostic over saved gold-source verifier scores.",
            "tab:phase-b-score-ensembles",
        ),
        "phase_b_ensemble_ci_table.md": ci_markdown_table(CI_COMPARISONS),
        "phase_b_ensemble_ci_table.tex": ci_latex_table(
            CI_COMPARISONS,
            "Sequence-cluster bootstrap comparisons for the score-averaging diagnostic.",
            "tab:phase-b-score-ensemble-ci",
        ),
        "phase_b_ensemble_calibrated_table.md": calibrated_markdown_table(
            CALIBRATED_ENSEMBLE_PATH
        ),
        "phase_b_ensemble_calibrated_table.tex": calibrated_latex_table(
            CALIBRATED_ENSEMBLE_PATH,
            "Held-out threshold calibration for the score-averaging diagnostic.",
            "tab:phase-b-score-ensemble-calibrated",
        ),
        "phase_b_training_budget_table.md": budget_markdown_table(TRAINING_BUDGET_RUNS),
        "phase_b_training_budget_table.tex": budget_latex_table(
            TRAINING_BUDGET_RUNS,
            "Seed-0 training-budget diagnostic for source-specific gold supervision.",
            "tab:phase-b-training-budget",
        ),
        "phase_b_training_budget_ci_table.md": ci_markdown_table(
            TRAINING_BUDGET_CI_COMPARISONS
        ),
        "phase_b_training_budget_ci_table.tex": ci_latex_table(
            TRAINING_BUDGET_CI_COMPARISONS,
            "Sequence-cluster bootstrap comparisons for the seed-0 training-budget diagnostic.",
            "tab:phase-b-training-budget-ci",
        ),
        "phase_b_training_budget_calibrated_table.md": calibrated_markdown_table(
            TRAINING_BUDGET_CALIBRATED_PATH
        ),
        "phase_b_training_budget_calibrated_table.tex": calibrated_latex_table(
            TRAINING_BUDGET_CALIBRATED_PATH,
            "Held-out threshold calibration for the seed-0 training-budget diagnostic.",
            "tab:phase-b-training-budget-calibrated",
        ),
    }

    for name, content in outputs.items():
        path = out_dir / name
        path.write_text(content + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
