"""Generate a self-contained HTML status dashboard from results/*.json.

Built so a NEWCOMER or returning member can drop in cold and get oriented in
under a minute: what we're building, why it was revived, where we are, the
headline result, decisions, the task board, how we work, and what's next.
Pure stdlib; embeds data into one HTML file (serve statically, regenerate as
runs land).

    python3 -m scripts.build_dashboard --out dashboard/index.html
"""
import os
import json
import glob
import html
import argparse
import datetime

RESULTS = "results"

# ---- editable narrative state -------------------------------------------------
TAGLINE = "Teaching a small, answer-blind model to catch reasoning mistakes step-by-step — by distilling a stronger teacher that's allowed to peek at the answer."

OVERVIEW = [
    ("What we're building",
     "A small <b>ground-truth-free student</b> that reads a math solution and flags <i>which step</i> the reasoning first goes wrong — not just whether the final answer is right. It learns this from a stronger <b>privileged teacher</b> that sees the reference solution while labeling."),
    ("Why it was revived",
     "The earlier version stalled (CLEAR + TinyLlama; weak scorers, gibberish on math). Revived <b>June 2026</b> with a shift to <b>step-level reasoning evaluation</b> over final-answer accuracy. Target: a publishable result for the <b>COLM workshop (~early July)</b> — flexible; quality over the date."),
    ("Where we are today",
     "<b>SUBMISSION SPRINT — COLM 2026 Efficient Reasoning workshop, deadline July 19 AoE.</b> The merged draft's three claims are all data-complete: teacher sweet spot (§4), verified transfer null with positive control (§5), supervision-mismatch diagnosis (§6). The <b>same-source GSM8K cell is DONE</b> (PR #30): holding the gold rows' exact 400 source problems fixed, generated labels stay weak — gold 0.7515 vs generated no-GT 0.6183 / priv 0.5494 ROC-AUC (4 seeds) — so source distribution alone is ruled out. Bonus: the §5 null reproduces <i>inside</i> the controlled cell (better teacher labels → significantly worse student, p=0.0004). Now in flight: the <b>format-rerender cell</b> (data ready, PR #31) to split rendering from label content, and a <b>BoN regrade</b> after a math_verify grading regression surfaced (pre-Jul-6 BoN numbers superseded, incl. the paper's §5 downstream diagnostic)."),
]

RESEARCH_Q = ("The research question",
    "Does distilling a teacher's <b>step-level score + natural-language critique</b> — and its <b>privileged</b> (answer-aware) judgment — into a small answer-blind student make it better at catching reasoning-step errors? <b>Finding:</b> privilege has a <b>tractability sweet spot</b> — it helps only where the teacher both <i>needs</i> the reference (can't self-verify) and can <i>use</i> it; and only <b>rich</b> privilege works (a worked solution, not a bare answer).")

PHASE = "SUBMISSION SPRINT → COLM 2026 Efficient Reasoning workshop, deadline JULY 19 AoE. All three claims data-complete; same-source cell DONE (PR #30). In flight: format-rerender cell (data ready, PR #31 → Saksham's cluster), BoN regrade (math_verify grading regression — paper's downstream numbers get replaced), Overleaf number port, COLM template port, length trim."

RUNS = [  # (label, state) state in {running, done, queued}
    ("Teacher bake-off (5 models, N=30)", "done"),
    ("Privilege × difficulty (GSM8K · MATH · OlympiadBench)", "done"),
    ("Cross-teacher replication (Qwen-27B · MATH N=150)", "done"),
    ("Threshold-free re-score (ROC/PR-AUC) — no-GT ≥ priv", "done"),
    ("Scaled run N=1000 — Gemma-4 labeling, BoN re-rank (no transfer)", "done"),
    ("Diagnostics: Gemma-4 probe (+0.07) · label agreement (~31% churn) · paired BoN (p=0.14)", "done"),
    ("Positive control — weak-vs-strong teacher, +0.050 ROC-AUC [0.021, 0.078], pipeline IS sensitive", "done"),
    ("Phase B mismatch diagnosis — gold 0.75–0.77 vs generated 0.48–0.68 under identical 3B path", "done"),
    ("4-seed generated-label sweep (priv BCE 0.5475 · no-GT rank 0.6062) — merged PR #28", "done"),
    ("Same-source GSM8K cell — gold 0.7515 vs generated no-GT 0.6183 / priv 0.5494 (4 seeds, PR #30)", "done"),
    ("1.5B paired BoN regrade — math_verify grading fix, cached pool + BoN-vs-N curve (Edward's Mac)", "running"),
    ("Format-rerender cell — generated labels in exact gold convention (data ready, PR #31 → cluster)", "queued"),
    ("Gold-3B downstream best-of-N vs majority vote (tooling ready; checkpoints live on Saksham's cluster)", "queued"),
]

DECISIONS = [
    ("Teacher", "Gemma-4-26b (official Gemma-class)",
     "F1 0.91, zero parse failures, ~12× faster than reasoning models, cross-family"),
    ("Dataset", "ProcessBench MATH",
     "GSM8K is saturated — the privilege gap is ≈0 there"),
    ("Privilege signal", "Full worked reference solution",
     "+0.05 F1 on MATH (N=400, 95% CI [0.01, 0.09], significant). A bare answer flips ZERO predictions — the teacher needs a reference reasoning trace, not a number."),
    ("Core claim (teacher)", "VALIDATED — privilege has a tractability sweet spot",
     "gap_solution: GSM8K ≈0 · MATH +0.05 (95% CI [0.01, 0.09], significant) · OlympiadBench ≈0 (verified) — helps only mid-difficulty; bare answer inert everywhere."),
    ("PRM eval metric", "Threshold-free (ROC-AUC / PR-AUC) — not F1 at a fixed cutoff",
     "F1/first_error_acc move with score-head calibration: a silent cell collapses F1 to ~0 while banking the ~0.44 error-free base rate on first_error_acc. Compare cells on AUC + the split (error_recall / pred_error_rate)."),
    ("Student transfer (Phase 2/3)", "NEGATIVE + diagnosed — real teacher gap is diffuse, doesn't distill",
     "no-GT ≥ priv on roc_auc (0.641 vs 0.631) AND paired downstream (n.s.); neither beats majority vote. Now reproduced INSIDE the same-source cell: teacher first-error match 0.800 (priv) vs 0.752 (no-GT) on the same 400 problems, yet the priv-trained student is significantly worse (no-GT +0.0242, p=0.0004). CAVEAT: all downstream BoN numbers pre-2026-07-06 were graded without math_verify (silent fallback) — regrade running; don't quote the old ones."),
]

TASKS = [  # who, track, status (active|blocked|queued|done), next action
    ("Edward", "Integration + submission", "active",
     "Merge PR #30 (same-source results, verified) and PR #31 (format-rerender data + BoN tooling — stacked on #30). BoN regrade running on the Mac (results/bon_paired_regrade). Then: review Overleaf ports, length trim to COLM 10pp, supplementary zip (anonymized artifacts promised in Appendix A/C), submit by ~Jul 18."),
    ("Saksham", "GPU pipeline", "active",
     "Two cluster jobs, recipes in HANDOFF_SAKSHAM.md: (1) format-rerender cells — run_gold_scorehead_gate.sh on data/labeled/math_{priv,nogt}_gsm8k400_pbformat_steps.jsonl, seeds 0–3, RUN_TAG=..._pbformat_to_math1000...; (2) gold-3B downstream BoN — the gold checkpoints only exist on the cluster; pip install math-verify FIRST. Still owed: the one-sentence seed-0 divergence explanation (variance footnote). Lit review: route cites through PAPER_FRAMING.md's must-cite table."),
    ("Henry", "Paper", "active",
     "Overleaf ports (Saksham pre-drafted most text in paper/main.tex — copy-adapt): 4-seed aggregates (0.5475/0.6062), same-source control table + paragraph, OPSD differentiation paragraph (drop-in in PAPER_FRAMING.md), efficiency sentence, §5 in-cell null sentence, GSM8K privilege reconciliation footnote (drafts in HANDOFF_HENRY.md). HOLD the §5 downstream BoN numbers — regrade in flight replaces them. Then: COLM template port, tighten title/abstract, Table 2 OlympiadBench dash, reference cleanup."),
]

HOW_WE_WORK = [
    ("Cadence", "Weekly <b>Sunday</b> syncs · ~10 hrs/week each · Slack between meetings"),
    ("Compute", "Two-phase: quick runs on <b>Apple Silicon</b> (M-series, 48 GB) → full runs on <b>2×3090 (48 GB VRAM)</b> via SSH"),
    ("Code", "<b>GitHub only</b> — single repo <code>edward-lcl/feedback-distillation</code>, frequent commits, no Colab sprawl"),
    ("Method", "Structured (Pydantic) step feedback · chain-of-thought evaluation over final-answer accuracy · Co-Scientist-style hypothesis iteration"),
]

BULLETPROOFING = [
    ("done", "Cross-teacher (Qwen-27B) confirms the pattern — solution gap +0.082, not a Gemma artifact."),
    ("done", "OlympiadBench (OE-only, verified) — privilege ≈0 there → sweet spot, NOT monotonic with difficulty."),
    ("done", "with_answer ≡ no_gt confirmed real (bare answer inert), not a wiring bug."),
    ("done", "PRM eval made threshold-free (ROC/PR-AUC + error_recall/pred_error_rate split) — fixed-cutoff F1 hid a silent-collapse artifact in Phase 2."),
    ("done", "N=1000 run verified to label through the real Gemma-4 teacher (~32k served requests) — the negative is not a weak-teacher artifact."),
    ("done", "Gemma-4 privilege probe: +0.07 solution gap — the actual labeling teacher's labels genuinely differ (null is not 'identical labels')."),
    ("done", "Label agreement: priv vs no-GT differ on ~31% of steps but symmetrically (1001 vs 956) — the teacher gain is diffuse, not directional."),
    ("done", "Same-pool paired BoN: students statistically indistinguishable (McNemar p=0.14) — but N=200 is underpowered; re-run @ N=1000."),
    ("done", "Positive control: weak-teacher student 0.575 vs 0.625 (+0.050 [0.021, 0.078], p=0.001) — pipeline detects teacher quality; the privilege null is not insensitivity."),
    ("done", "Transfer null now has a CI: priv−no-GT = −0.010 ROC-AUC, 95% CI [−0.025, 0.005] (paired bootstrap, 10k resamples)."),
    ("done", "Generated-label seed asymmetry closed: 4-seed sweep (priv BCE 0.5475 [0.50–0.59] · no-GT rank 0.6062 [0.54–0.68]); gap vs gold survives EVERY seed (best generated 0.676 < worst gold 0.726)."),
    ("done", "Same-source GSM8K cell: gold 0.7515 vs generated 0.6183/0.5494 with source problems held fixed (all 8 cells healthy, no collapse; weakest gold seed beats best generated seed +0.0642 [0.0445, 0.0831], p=0.0004) — source distribution ruled out as sole explanation (PR #30)."),
    ("done", "Transfer null reproduced in-cell: on the same 400 problems the priv teacher is better (first-error 0.800 vs 0.752) but its student is significantly worse (no-GT +0.0242 [0.0135, 0.0344], p=0.0004) — cleanest §5 evidence yet."),
    ("todo", "Format-rerender cell (data ready, PR #31): generated labels in the exact gold convention — splits label rendering from label content; Saksham trains."),
    ("todo", "BoN grading regression: math_verify was never installed, answers_match silently string-matched symbolic MATH answers — ALL pre-Jul-6 BoN numbers superseded (incl. the paper's §5 downstream diagnostic); 1.5B regrade running, gold-3B BoN queued on cluster."),
    ("todo", "Seed-0 variance footnote: rerun diverged 0.477 vs canonical 0.550 (both degenerate, pred_error_rate=0) — disclose, don't bury; it evidences §6.4 BCE instability."),
    ("idea", "Does a stronger teacher extend the sweet spot upward on OlympiadBench? (post-submission)"),
]

MILESTONES = [
    ("done", "Teacher/dataset/privilege locked; sweet spot verified (GSM8K·MATH·OlympiadBench)"),
    ("done", "Verified N=1000 null + diagnosed (diffuse gap) + positive control (pipeline IS sensitive)"),
    ("done", "Phase B mismatch diagnosis: gold labels train a competent 3B verifier, generated labels don't (4 seeds)"),
    ("done", "Papers merged into one COLM draft (paper/slfd_colm_er.pdf, 2026-07-04)"),
    ("done", "Same-source GSM8K cell: gap survives with source problems held fixed (PR #30)"),
    ("now", "Format-rerender cell + BoN regrade/gold-3B + Overleaf number port + COLM template + trim"),
    ("then", "Submit COLM 2026 Efficient Reasoning — DEADLINE JULY 19 AoE (notification Jul 31)"),
]

# Path to submission — granular, owner-tagged, so everyone (esp. Saksham) has line of sight.
PATH_TO_SUBMISSION = [
    ("done", "Team", "Teacher / dataset / privilege locked; sweet spot verified across 3 difficulty tiers + cross-family."),
    ("done", "Edward", "Full pipeline built &amp; smoke-tested (probe → student ablation → best-of-N verifier); runbooks written."),
    ("done", "Saksham", "N=1000 run complete &amp; verified (Gemma-4 labeling): privilege does NOT transfer to the 1.5B student — no-GT ≥ priv on roc_auc + downstream; neither beats majority vote."),
    ("done", "Saksham", "Phase A diagnostics: Gemma-4 probe (+0.07) · label agreement (~31% churn, symmetric) · paired BoN (p=0.14, n.s.). math_verify wired into answer-matching."),
    ("done", "Saksham", "Phase B complete: positive control (B3) + gold-vs-generated mismatch diagnosis + 4-seed generated-label sweep (PR #28, merged). Standalone draft already on the COLM style."),
    ("done", "Team", "Two papers combined into one 12-page draft (paper/slfd_colm_er.pdf) — sweet spot §4 + verified null §5 + mismatch diagnosis §6."),
    ("done", "Team", "Same-source GSM8K control complete (labels: Edward's Mac; training: Saksham's cluster, seeds 0–3 both conditions, all cells healthy): gold 0.7515 vs generated no-GT 0.6183 / priv 0.5494 — source distribution ruled out as sole explanation (PR #30)."),
    ("now", "Edward", "Merge PR #30 + #31; babysit the BoN regrade run; then supplementary zip (anonymized artifacts, full per-seed table), final review, submit."),
    ("now", "Saksham", "Cluster: format-rerender cells (pbformat data, gold-source recipe, seeds 0–3) + gold-3B downstream BoN (install math-verify first — grading regression). Seed-0 divergence sentence for the footnote. Recipes in HANDOFF_SAKSHAM.md."),
    ("now", "Henry", "Overleaf ports (4-seed + same-source + OPSD paragraph + efficiency + reconciliation footnote — drafts in HANDOFF_HENRY.md, much pre-written in paper/main.tex); HOLD §5 BoN numbers for the regrade; COLM template port; trim to 10pp; Table 2 dash fix; refs cleanup."),
    ("then", "Team", "Submit COLM 2026 Efficient Reasoning — DEADLINE JULY 19 AoE (extended from Jul 12; notification Jul 31; non-archival, dual-submission OK)."),
]

OVERLEAF_URL = "https://www.overleaf.com/7588472223rycxtnntqhqm#f07fa2"  # combined paper (Slack 2026-07-01)
LINKS = [
    ("Paper (Overleaf)", OVERLEAF_URL),
    ("Framing doctrine (PAPER_FRAMING.md)", "https://github.com/edward-lcl/feedback-distillation/blob/main/PAPER_FRAMING.md"),
    ("Research roadmap", "https://github.com/edward-lcl/feedback-distillation/blob/main/RESEARCH_ROADMAP.md"),
    ("Phase B runbook", "https://github.com/edward-lcl/feedback-distillation/blob/main/RUNBOOK_PHASE_B.md"),
    ("Saksham runbook", "https://github.com/edward-lcl/feedback-distillation/blob/main/HANDOFF_SAKSHAM.md"),
    ("Henry runbook", "https://github.com/edward-lcl/feedback-distillation/blob/main/HANDOFF_HENRY.md"),
    ("Repo", "https://github.com/edward-lcl/feedback-distillation"),
]

# Frontier reading list — mirrors PAPER_FRAMING.md (doctrine lives there; this
# renders it). action in {must-cite, nice, watch}. Updated 2026-07-05 scan.
FRONTIER = [
    ("must-cite", "OPSD — On-Policy Self-Distillation (arXiv:2601.18734)",
     "2026's 'privileged teacher' means THIS (successful, on-policy, policy-level). One paragraph must differentiate: we distill privileged step-error LABELS, off-policy, into an answer-blind VERIFIER — our null confirms, not contradicts, their off-policy-mismatch premise."),
    ("must-cite", "Position-Weighted OPSD (arXiv:2605.21606)",
     "Nearest 2026 relative of 'where are privileged-teacher signals unreliable' — token-level where ours is step-label-level."),
    ("must-cite", "Noise-aware PRM training (arXiv:2601.12748)",
     "Reviewers will ask 'would label denoising rescue the student?' Pre-empt: our gold-matched control isolates distribution/format, not noise."),
    ("nice", "Rethinking On-Policy Distillation (arXiv:2604.13016)",
     "Off-policy/distribution-mismatched supervision fails where on-policy succeeds — our null, predicted from the policy side."),
    ("nice", "Strong Teacher Not Needed (arXiv:2605.23857)",
     "Teacher strength ≠ student gains; supports the positive-control logic."),
    ("nice", "Adaptive Generate-Rank-Verify (arXiv:2605.17609) · Trust-but-Verify survey (arXiv:2508.16665)",
     "Costly-verification framing: the sweet spot as a verification-budget allocation rule."),
    ("nice", "uPRM (arXiv:2605.10158) · Scan (arXiv:2509.16548)",
     "Unsupervised/denoised PRM alternatives — the 'fix the labels' line our diagnosis is differentiated from."),
    ("watch", "PRMBench (arXiv:2501.03124)",
     "ProcessBench-only eval reads narrow by mid-2026; companion numbers are the first cut under time pressure."),
]
# ------------------------------------------------------------------------------


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _pill(state):
    return f"<span class='pill {state}'>{state}</span>"


def _cond_row(name, d):
    if not d:
        return f"<tr><td>{name}</td><td colspan=4 class=muted>pending</td></tr>"
    return (f"<tr><td>{name}</td><td><b>{_fmt(d.get('f1'))}</b></td>"
            f"<td>{_fmt(d.get('correct_acc'))}</td><td>{_fmt(d.get('error_acc'))}</td>"
            f"<td>{_fmt(d.get('parse_failure_rate'))}</td></tr>")


def _gap_badge(g):
    if g is None:
        return "<span class=muted>—</span>"
    cls = "pos" if g > 0.02 else ("neg" if g < -0.02 else "zero")
    return f"<span class='badge {cls}'>{g:+.3f}</span>"


def student_results_html():
    """Render the verified N=1000 student-transfer results (Phase 2 roc_auc + Phase 3 BoN)."""
    cells = [("priv_critique", "Privileged + critique"),
             ("priv_scoreonly", "Privileged, score-only"),
             ("nogt_critique", "No-GT + critique")]
    p2 = ""
    for key, label in cells:
        d = _load(f"{RESULTS}/ablation/{key}/processbench_results.json") or {}
        hi = " style='font-weight:700;color:var(--accent)'" if key == "nogt_critique" else ""
        p2 += (f"<tr><td>{label}</td><td{hi}>{_fmt(d.get('roc_auc'))}</td>"
               f"<td>{_fmt(d.get('pr_auc'))}</td><td>{_fmt(d.get('f1'))}</td>"
               f"<td>{_fmt(d.get('error_recall'))}</td></tr>")
    bp = _load(f"{RESULTS}/bon_priv/bon_results.json") or {}
    bn = _load(f"{RESULTS}/bon_nogt/bon_results.json") or {}

    def bonrow(label, d, hi=False):
        s = " style='font-weight:700;color:var(--accent)'" if hi else ""
        return (f"<tr><td>{label}</td><td>{_fmt(d.get('pass@1'))}</td>"
                f"<td{s}>{_fmt(d.get('prm_rerank'))}</td><td>{_fmt(d.get('majority_vote'))}</td>"
                f"<td>{_fmt(d.get('oracle_pass@N'))}</td></tr>")
    p3 = bonrow("No-GT verifier", bn, hi=True) + bonrow("Privileged verifier", bp)
    return f"""
<div class="section-label">Student transfer — verified N=1000 (privilege does NOT transfer)</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <div style="padding:10px 16px 4px;font-size:12px;color:var(--ink3)">Phase 2 — step-level (compare on <b>roc_auc</b>, not F1). No-GT is highest.</div>
  <table class="data-table">
    <tr><th>Student (teacher labels)</th><th>roc_auc</th><th>pr_auc</th><th>F1</th><th>error_recall</th></tr>
    {p2}
  </table>
  <div style="padding:10px 16px 4px;font-size:12px;color:var(--ink3);border-top:1px solid var(--border)">Phase 3 — Best-of-N re-rank, N=1000. No-GT verifier re-ranks better; neither beats majority vote.</div>
  <table class="data-table">
    <tr><th>Verifier</th><th>pass@1</th><th>prm_rerank</th><th>majority_vote</th><th>oracle</th></tr>
    {p3}
  </table>
  <div style="padding:10px 16px 14px;font-size:12px;color:var(--ink3);border-top:1px solid var(--border)">Labeling confirmed through the served Gemma-4 teacher (~32k requests). <b>Diagnostics:</b> Gemma-4 teacher gap +0.07 (real); priv vs no-GT labels differ on ~31% of steps but symmetrically (1001 vs 956) — diffuse, not directional; paired re-rank students statistically indistinguishable (McNemar p=0.14). Mechanism: a real teacher gap that's too diffuse to distill. Gating next step: the student must first beat majority vote (Phase B).</div>
</div>
"""


def privilege_card(probe, title, sub, running=False):
    if not probe:
        body = "<p class=muted>running…</p>" if running else "<p class=muted>pending</p>"
        return f"<div class=card><h3>{title}</h3><p class='muted small'>{sub}</p>{body}</div>"
    rows = "".join(_cond_row(n, probe.get(k)) for n, k in
                   [("no-GT", "no_gt"), ("+ answer", "with_answer"), ("+ full solution", "with_solution")])
    ga, gs = probe.get("gap_answer_f1"), probe.get("gap_solution_f1")
    return (f"<div class=card><h3>{title}</h3><p class='muted small'>{sub}</p>"
            f"<table><tr><th>condition</th><th>F1</th><th>correct</th><th>error_acc</th><th>parse</th></tr>{rows}</table>"
            f"<p class=small>gap vs no-GT — answer {_gap_badge(ga)} &nbsp; solution {_gap_badge(gs)}</p></div>")


def bakeoff_rows():
    rows = []
    for path in glob.glob(f"{RESULTS}/teacher_bakeoff/*/teacher_eval.json"):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("_"):
            continue
        w = (_load(path) or {}).get("with_gt", {})
        rows.append((w.get("f1", 0), name, w))
    if not rows:
        return "<tr><td colspan=5 class=muted>no results yet</td></tr>"
    rows.sort(reverse=True)
    return "".join(
        f"<tr><td>{'⭐ ' if i == 0 else ''}{html.escape(n)}</td><td><b>{_fmt(w.get('f1'))}</b></td>"
        f"<td>{_fmt(w.get('correct_acc'))}</td><td>{_fmt(w.get('error_acc'))}</td>"
        f"<td>{_fmt(w.get('parse_failure_rate'))}</td></tr>"
        for i, (_, n, w) in enumerate(rows))


def bakeoff_rows_dark():
    rows = []
    for path in glob.glob(f"{RESULTS}/teacher_bakeoff/*/teacher_eval.json"):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("_"):
            continue
        d = (_load(path) or {})
        w = d.get("with_gt") or d.get("no_gt") or {}
        rows.append((w.get("f1", 0), name, w))
    if not rows:
        return "<tr><td colspan=5 style='color:var(--ink3);padding:16px'>No results yet</td></tr>"
    rows.sort(reverse=True)
    out = []
    for i, (_, n, w) in enumerate(rows):
        star = "<span class='b b-star'>★</span>&nbsp; " if i == 0 else ""
        f1_cls = "highlight" if i == 0 else "mono"
        pf = w.get("parse_failure_rate", 0)
        pf_cls = "warn-val" if pf and pf > 0.1 else ("mono" if not pf or pf == 0 else "mono" )
        pf_style = ' style="color:var(--yellow)"' if pf and 0 < pf <= 0.1 else ""
        out.append(
            f"<tr><td>{star}{html.escape(n)}</td>"
            f"<td class='{f1_cls}'>{_fmt(w.get('f1'))}</td>"
            f"<td class='mono'>{_fmt(w.get('correct_acc'))}</td>"
            f"<td class='mono'>{_fmt(w.get('error_acc'))}</td>"
            f"<td class='{pf_cls}'{pf_style}>{_fmt(pf)}</td></tr>"
        )
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    stamp = args.stamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Verified numbers embedded as a fallback so the cards ALWAYS render — results/
    # is gitignored, so CI builds and fresh clones would otherwise show "Pending".
    def _c(f1, ea, nc, ne):
        return {"f1": f1, "error_acc": ea, "correct_acc": None,
                "n_correct_samples": nc, "n_error_samples": ne, "parse_failure_rate": 0.0}
    EMBED = {
        "gsm8k": {"no_gt": _c(0.889, 0.800, 25, 25), "with_answer": _c(0.856, 0.800, 25, 25),
                  "with_solution": _c(0.837, 0.720, 25, 25), "gap_answer_f1": -0.033, "gap_solution_f1": -0.052},
        # MATH headline = definitive N=400 evidence run: solution gap +0.051, 95% CI
        # [0.010, 0.093] (significant). no-GT vs +solution were the two conditions run at
        # N=400; the "+answer flips zero" result is the stable N=150 finding (kept inert).
        "math_main": {"no_gt": _c(0.726, 0.608, 173, 227), "with_answer": _c(0.726, 0.608, 173, 227),
                      "with_solution": _c(0.777, 0.661, 173, 227), "gap_answer_f1": 0.000, "gap_solution_f1": 0.051},
        "math_early": {"no_gt": _c(0.649, 0.480, 25, 25), "with_answer": _c(0.750, 0.600, 25, 25),
                       "with_solution": _c(0.781, 0.640, 25, 25), "gap_answer_f1": 0.101, "gap_solution_f1": 0.132},
        "qwen": {"no_gt": _c(0.653, 0.526, 72, 78), "with_answer": _c(0.690, 0.564, 72, 78),
                 "with_solution": _c(0.735, 0.667, 72, 78), "gap_answer_f1": 0.037, "gap_solution_f1": 0.082},
    }
    gsm = _load(f"{RESULTS}/teacher_eval/privilege_probe.json") or EMBED["gsm8k"]
    math = _load(f"{RESULTS}/evidence_pack_n400/privilege_probe.json") or EMBED["math_main"]
    math_early = _load(f"{RESULTS}/teacher_eval_math/privilege_probe.json") or EMBED["math_early"]
    qwen = _load(f"{RESULTS}/teacher_eval_math_qwen27b/privilege_probe.json") or EMBED["qwen"]

    overview = "".join(f"<div class=ocard><h3>{t}</h3><p class=small>{b}</p></div>" for t, b in OVERVIEW)
    runs = "".join(f"<li>{_pill(s)} {html.escape(l)}</li>" for l, s in RUNS)
    dec = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td><td class='muted small'>{html.escape(w)}</td></tr>"
                  for k, v, w in DECISIONS)
    tasks = "".join(
        f"<tr><td><b>{who}</b></td><td>{track}</td><td>{_pill(st)}</td><td class=small>{html.escape(nx)}</td></tr>"
        for who, track, st, nx in TASKS)
    how = "".join(f"<tr><td><b>{k}</b></td><td class=small>{v}</td></tr>" for k, v in HOW_WE_WORK)
    bullets = "".join(f"<li>{_pill(s)} {html.escape(t)}</li>" for s, t in BULLETPROOFING)
    miles = "".join(f"<div class='mile {s}'>{_pill(s)}<span>{html.escape(t)}</span></div>" for s, t in MILESTONES)

    head = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLFD — project dashboard</title>
<!-- System fonts only — no external dependency, so text always renders instantly
     (the Google Fonts version could blank out on a slow/blocked network). Inter is
     used if installed locally; otherwise the OS sans-serif. -->
<style>:root{}</style>
<style>
:root{--bg:#0a0c10;--bg2:#0d1117;--surface:#111620;--surface2:#161c28;--surface3:#1c2436;--border:#1f2a3e;--border2:#2a374f;--ink:#e8edf5;--ink2:#a8b8cc;--ink3:#5e7490;--accent:#4a7cf5;--accent-dim:rgba(74,124,245,.10);--accent-glow:rgba(74,124,245,.04);--green:#2ea86a;--green-dim:rgba(46,168,106,.10);--red:#d95f5f;--red-dim:rgba(217,95,95,.10);--yellow:#c4972a;--yellow-dim:rgba(196,151,42,.10);--purple:#8b74d4;--purple-dim:rgba(139,116,212,.10);--r:8px;--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--mono:'JetBrains Mono','SF Mono','Cascadia Code',Consolas,monospace}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--font);font-size:14px;line-height:1.65;color:var(--ink2);background:var(--bg);-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
nav{position:sticky;top:0;z-index:100;height:52px;display:flex;align-items:center;gap:4px;padding:0 28px;background:rgba(10,12,16,.85);backdrop-filter:blur(20px) saturate(180%);border-bottom:1px solid var(--border)}
.nav-brand{font-weight:700;font-size:15px;color:var(--ink);letter-spacing:-.3px;margin-right:20px;white-space:nowrap;display:flex;align-items:center;gap:8px}
.nav-dot{width:7px;height:7px;border-radius:50%;background:var(--green);opacity:.85}
.nav-links{display:flex;gap:2px}
.nav-links a{color:var(--ink3);font-size:13px;text-decoration:none;padding:5px 10px;border-radius:6px;transition:color .15s,background .15s}
.nav-links a:hover{color:var(--ink);background:var(--surface2)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.nav-meta{font-size:12px;color:var(--ink3)}
.nav-gh{font-size:12.5px;color:var(--ink3);text-decoration:none;border:1px solid var(--border2);padding:4px 12px;border-radius:6px;transition:all .15s;font-weight:500}
.nav-gh:hover{color:var(--ink);border-color:var(--accent)}
.hero{position:relative;overflow:hidden;padding:44px 36px 36px;border-bottom:1px solid var(--border);background:var(--bg2)}
.hero-inner{max-width:1120px;margin:0 auto}
.hero-eyebrow{font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:var(--ink3);margin-bottom:12px}
.hero h1{font-size:26px;font-weight:700;color:var(--ink);letter-spacing:-.3px;line-height:1.25;margin-bottom:10px}
.hero-sub{font-size:15px;color:var(--ink2);max-width:640px;line-height:1.7;margin-bottom:24px}
.hero-chips{display:flex;flex-wrap:wrap;gap:8px}
[title]{cursor:help}
.cond-table td[title]:first-child,.chip[title]{text-decoration:underline dotted rgba(168,184,204,.4);text-underline-offset:3px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.2px;border:1px solid transparent}
.chip-green{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.chip-blue{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.chip-yellow{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.chip-gray{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 80px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--ink3);display:flex;align-items:center;gap:12px;margin:44px 0 18px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px}
.card-title{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:6px}
.ov-card{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:var(--r);padding:20px 22px}
.ov-card h3{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:10px}
.ov-card p{font-size:13px;color:var(--ink2);line-height:1.65}
.ov-card strong{color:var(--ink);font-weight:600}
.callout{background:var(--accent-glow);border:1px solid rgba(74,124,245,.2);border-left:3px solid var(--accent);border-radius:var(--r);padding:18px 22px}
.callout h3{font-size:13px;font-weight:600;color:var(--accent);margin-bottom:8px}
.callout p{font-size:13.5px;color:var(--ink2);line-height:1.7}
.callout strong{color:var(--ink)}
.insight{background:rgba(46,168,106,.05);border:1px solid rgba(46,168,106,.15);border-left:3px solid var(--green);border-radius:var(--r);padding:16px 20px;font-size:13px;color:var(--ink2);line-height:1.7}
.insight strong{color:var(--green)}
.warn{background:var(--red-dim);border:1px solid rgba(217,95,95,.2);border-left:3px solid var(--red);border-radius:8px;padding:12px 16px;font-size:12.5px;color:#f4a0a0;line-height:1.6;margin-top:14px}
.warn strong{color:var(--red)}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;display:flex;flex-direction:column}
.result-card.primary{border-color:rgba(74,124,245,.4);box-shadow:0 0 0 1px rgba(74,124,245,.1),0 8px 32px rgba(74,124,245,.06)}
.rc-head{padding:16px 20px 14px;border-bottom:1px solid var(--border);background:var(--surface2)}
.rc-head h3{font-size:13.5px;font-weight:600;color:var(--ink)}
.rc-head .rc-meta{font-size:12px;color:var(--ink3);margin-top:2px}
.result-card.primary .rc-head{background:linear-gradient(135deg,rgba(74,124,245,.08) 0%,var(--surface2) 100%)}
.cond-table{width:100%}
.cond-table tr{border-bottom:1px solid var(--border)}
.cond-table tr:last-child{border-bottom:none}
.cond-table td{padding:10px 18px;font-size:13px;color:var(--ink2)}
.cond-table td:first-child{color:var(--ink3);font-size:12.5px}
.cond-table td.val{font-family:var(--mono);font-size:13.5px;font-weight:500;color:var(--ink);text-align:right}
.cond-table tr.winner td{background:rgba(46,168,106,.05)}
.cond-table tr.winner td.val{color:var(--green)}
.cond-table tr.delta td{padding-top:8px;padding-bottom:10px;background:rgba(255,255,255,.02);border-top:1px dashed var(--border);font-size:12px;color:var(--ink3)}
.cond-table tr.delta td.val{font-size:12px;color:var(--ink3)}
.metric-row{display:grid;grid-template-columns:repeat(2,1fr);gap:0;border-top:1px solid var(--border)}
.metric-cell{padding:14px 18px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:3px}
.metric-cell:last-child{border-right:none}
.metric-label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);font-weight:600}
.metric-value{font-family:var(--mono);font-size:22px;font-weight:500;color:var(--ink);line-height:1.1}
.metric-value.good{color:var(--green)}
.metric-sub{font-family:var(--mono);font-size:11.5px;color:var(--ink3)}
.rc-note{padding:10px 18px 14px;font-size:12px;color:var(--ink3);line-height:1.55;border-top:1px solid var(--border)}
.b{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;font-family:var(--mono)}
.b-pos{background:var(--green-dim);color:var(--green)}
.b-neg{background:var(--red-dim);color:var(--red)}
.b-zero{background:var(--surface2);color:var(--ink3)}
.b-star{background:rgba(196,151,42,.15);color:var(--yellow)}
.data-table{width:100%;border-collapse:collapse;font-size:13px}
.data-table th{text-align:left;padding:10px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2)}
.data-table td{padding:11px 16px;color:var(--ink2);border-bottom:1px solid var(--border);vertical-align:middle}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:rgba(255,255,255,.025)}
.data-table td.mono{font-family:var(--mono);font-size:13px}
.data-table td.highlight{color:var(--green);font-weight:600;font-family:var(--mono)}
.data-table td.warn-val{color:var(--red);font-family:var(--mono)}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;white-space:nowrap;border:1px solid transparent}
.p-done{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.p-running{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.p-todo,.p-next,.p-then{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.p-idea{background:var(--purple-dim);color:var(--purple);border-color:rgba(139,116,212,.25)}
.p-active{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.p-blocked{background:var(--red-dim);color:var(--red);border-color:rgba(217,95,95,.25)}
.p-crashed{background:var(--red-dim);color:var(--red);border-color:rgba(217,95,95,.25)}
.task-table{width:100%;border-collapse:collapse}
.task-table th{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);text-align:left;padding:10px 18px;border-bottom:1px solid var(--border);background:var(--surface2)}
.task-table td{padding:14px 18px;border-bottom:1px solid var(--border);vertical-align:top}
.task-table tr:last-child td{border-bottom:none}
.task-table tr:hover td{background:rgba(255,255,255,.018)}
.task-who{font-weight:600;color:var(--ink);white-space:nowrap;font-size:13.5px}
.task-track{font-size:12.5px;color:var(--ink3)}
.task-action{font-size:13px;color:var(--ink2);line-height:1.6}
.milestones{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start}
.mile{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:6px}
.path{list-style:none;padding:0;margin:0}
.path li{display:flex;align-items:flex-start;gap:12px;padding:9px 0;border-bottom:1px solid var(--border)}
.path li:last-child{border-bottom:none}
.path .ptext{font-size:13.5px;color:var(--ink2);line-height:1.5}
.path .pwho{font-weight:600;color:var(--ink)}
.links{display:flex;flex-wrap:wrap;gap:10px;margin-top:4px}
.links a{display:inline-flex;align-items:center;gap:6px;font-size:13px;border:1px solid var(--border2);padding:8px 14px;border-radius:8px;color:var(--ink);background:var(--surface2)}
.links a:hover{border-color:var(--accent);text-decoration:none}
.mile.done{border-color:rgba(46,168,106,.25)}
.mile.now{border-color:rgba(196,151,42,.25);background:rgba(196,151,42,.04)}
.mile-text{font-size:12.5px;color:var(--ink2);line-height:1.5}
.status-list{list-style:none}
.status-list li{display:flex;align-items:flex-start;gap:10px;padding:6px 0;font-size:13.5px;color:var(--ink2)}
.status-list li:not(:last-child){border-bottom:1px solid var(--border)}
.kv-table{width:100%;border-collapse:collapse}
.kv-table td{padding:12px 0;border-bottom:1px solid var(--border);font-size:13.5px;vertical-align:top}
.kv-table tr:last-child td{border-bottom:none}
.kv-table td:first-child{color:var(--ink);font-weight:600;width:110px;white-space:nowrap;padding-right:24px}
code{font-family:var(--mono);font-size:12.5px;background:var(--surface3);padding:2px 6px;border-radius:4px;color:var(--accent)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.dec-table{width:100%;border-collapse:collapse}
.dec-table th{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);text-align:left;padding:10px 16px;border-bottom:1px solid var(--border);background:var(--surface2)}
.dec-table td{padding:13px 16px;border-bottom:1px solid var(--border);font-size:13px;color:var(--ink2);vertical-align:middle}
.dec-table tr:last-child td{border-bottom:none}
.dec-table td:first-child{color:var(--ink);font-weight:600;white-space:nowrap}
.dec-table td.dim{color:var(--ink3);font-size:12.5px;line-height:1.55}
.dec-table td.validated{color:var(--green);font-weight:600}
@media(max-width:960px){.g3{grid-template-columns:1fr 1fr}}
@media(max-width:640px){.g2,.g3{grid-template-columns:1fr}}
</style>
</head>"""

    def _gpill(state):
        cls = {"done":"p-done","running":"p-running","crashed":"p-crashed",
               "active":"p-active","blocked":"p-blocked","todo":"p-todo",
               "now":"p-running","next":"p-next","then":"p-then","idea":"p-idea"}.get(state, "p-todo")
        return f"<span class='pill {cls}'>{state}</span>"

    _COND_TIP = {
        "no-GT": "Teacher sees only the problem + candidate steps (answer-blind) — the unprivileged baseline.",
        "+ answer": "Teacher additionally sees the final reference answer (thin privilege).",
        "+ full solution": "Teacher sees the full worked reference solution (richest privilege).",
    }

    def _gcond_row(name, d, winner=False):
        tip = _COND_TIP.get(name, "")
        if not d:
            return f"<tr><td title=\"{tip}\">{name}</td><td class='val' colspan=2 style='color:var(--ink3)'>pending</td></tr>"
        cls = " class='winner'" if winner else ""
        return (f"<tr{cls}><td title=\"{tip}\">{name}</td>"
                f"<td class='val' title=\"first-error F1: harmonic mean of correct-acc and error-acc\">{_fmt(d.get('f1'))}</td>"
                f"<td class='val' title=\"error recall: fraction of erroneous solutions whose first wrong step is correctly localized\">{_fmt(d.get('error_acc'))}</td></tr>")

    def _gcond_delta(ga, gs):
        def badge(v):
            if v is None: return "—"
            cls = "b-pos" if v > 0.02 else ("b-neg" if v < -0.02 else "b-zero")
            return f"<span class='b {cls}'>{v:+.3f}</span>"
        return f"<tr class='delta'><td>Δ vs no-GT</td><td class='val'>{badge(ga)}</td><td class='val'>{badge(gs)}</td></tr>"

    def _gpriv_card(probe, title, sub, winner_cond="with_solution", primary=False, running=False):
        card_cls = "result-card primary" if primary else "result-card"
        rc_head_content = f"<h3>{title}</h3><div class='rc-meta'>{sub}</div>"
        if not probe:
            body = "<div style='padding:16px 18px'>"
            if running:
                body += "<div class='warn'><strong>Crashed</strong> — oMLX ChunkedEncodingError mid-run. No data. Needs restart (~2-3 hrs).</div>"
            else:
                body += "<p style='color:var(--ink3);font-size:13px;padding:4px 0'>Pending</p>"
            body += "</div>"
            return f"<div class='{card_cls}'><div class='rc-head'>{rc_head_content}</div>{body}</div>"

        no_gt = probe.get("no_gt", {})
        with_ans = probe.get("with_answer", {})
        with_sol = probe.get("with_solution", {})
        ga = probe.get("gap_answer_f1")
        gs = probe.get("gap_solution_f1")

        rows = (_gcond_row("no-GT", no_gt) +
                _gcond_row("+ answer", with_ans) +
                _gcond_row("+ full solution", with_sol, winner=True) +
                _gcond_delta(ga, gs))

        sol_f1 = with_sol.get("f1", 0) if with_sol else 0
        no_f1  = no_gt.get("f1", 0) if no_gt else 0
        sol_err = with_sol.get("error_acc", 0) if with_sol else 0
        no_err  = no_gt.get("error_acc", 0) if no_gt else 0
        delta_f1 = sol_f1 - no_f1

        if primary:
            metric_block = f"""<div style='padding:0 18px 14px'>
  <div style='display:grid;grid-template-columns:1fr 1fr;border:1px solid rgba(74,124,245,.3);border-radius:8px;overflow:hidden;background:var(--accent-glow)'>
    <div class='metric-cell'><div class='metric-label'>solution Δ F1</div><div class='metric-value good' style='font-size:26px'>{delta_f1:+.2f}</div><div class='metric-sub'>N={no_gt.get("n_correct_samples",0)+no_gt.get("n_error_samples",0) if no_gt else "?"}</div></div>
    <div class='metric-cell'><div class='metric-label'>err_recall Δ</div><div class='metric-value good' style='font-size:26px'>{sol_err-no_err:+.2f}</div><div class='metric-sub'>{no_err:.3f} → {sol_err:.3f}</div></div>
  </div></div>"""
        else:
            verdict_color = "var(--green)" if delta_f1 > 0.02 else ("var(--red)" if delta_f1 < -0.02 else "var(--ink3)")
            verdict = "privilege helps" if delta_f1 > 0.02 else ("saturated" if delta_f1 < -0.02 else "no signal")
            metric_block = f"""<div style='padding:0 18px 14px'>
  <div style='display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--border);border-radius:8px;overflow:hidden'>
    <div class='metric-cell'><div class='metric-label'>solution Δ</div><div class='metric-value' style='font-size:20px;color:{verdict_color}'>{delta_f1:+.2f}</div><div class='metric-sub'>F1 gap</div></div>
    <div class='metric-cell'><div class='metric-label'>verdict</div><div class='metric-value' style='font-size:13px;color:{verdict_color};padding-top:4px'>{verdict}</div><div class='metric-sub'>vs no-GT</div></div>
  </div></div>"""

        return f"""<div class='{card_cls}'>
  <div class='rc-head'>{rc_head_content}</div>
  <table class='cond-table'><tr><th style='padding:8px 18px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2)'></th><th style='padding:8px 18px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2);text-align:right'>F1</th><th style='padding:8px 18px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2);text-align:right'>err_acc</th></tr>{rows}</table>
  {metric_block}
</div>"""

    overview_html = "".join(
        f"<div class='ov-card'><h3>{t}</h3><p>{b}</p></div>" for t, b in OVERVIEW)

    runs_html = "".join(f"<li>{_gpill(s)} {html.escape(l)}</li>" for l, s in RUNS)

    dec_html = "".join(
        "<tr><td>" + k + "</td><td>" +
        ("<span style='color:var(--green);font-weight:600'>" + v + "</span>" if k == "Core claim" else v) +
        "</td><td class='dim'>" + html.escape(w) + "</td></tr>"
        for k, v, w in DECISIONS)

    tasks_html = "".join(
        f"<tr><td><div class='task-who'>{who}</div><div class='task-track'>{track}</div></td><td></td><td>{_gpill(st)}</td><td class='task-action'>{nx}</td></tr>"
        for who, track, st, nx in TASKS)

    how_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in HOW_WE_WORK)

    bullets_html = "".join(f"<li>{_gpill(s)} {html.escape(t)}</li>" for s, t in BULLETPROOFING)

    miles_html = "".join(
        f"<div class='mile {'done' if s == 'done' else ('now' if s in ('now','running') else '')}'>"
        f"{_gpill(s)}<div class='mile-text'>{html.escape(t)}</div></div>"
        for s, t in MILESTONES)

    path_html = "".join(
        f"<li>{_gpill(s)}<div class='ptext'><span class='pwho'>{who}:</span> {t}</div></li>"
        for s, who, t in PATH_TO_SUBMISSION)
    links_html = "".join(f"<a href='{u}' target='_blank'>{html.escape(n)} ↗</a>" for n, u in LINKS)
    frontier_html = "".join(
        f"<tr><td><span class='pill {'done' if a == 'must-cite' else ('now' if a == 'nice' else 'queued')}'>{a}</span></td>"
        f"<td><b>{html.escape(t)}</b></td><td class=small>{html.escape(r)}</td></tr>"
        for a, t, r in FRONTIER)

    bakeoff_html = bakeoff_rows_dark()
    student_html = student_results_html()

    htmldoc = head + f"""
<body>
<nav>
  <span class="nav-brand"><span class="nav-dot"></span>SLFD</span>
<div class="nav-links">
    <a href="#overview">Overview</a>
    <a href="#results">Results</a>
    <a href="#decisions">Decisions</a>
    <a href="#tasks">Tasks</a>
    <a href="#path">Path</a>
    <a href="#next">Next</a>
    <a href="https://github.com/edward-lcl/feedback-distillation/blob/main/paper/SLFD_draft.pdf" target="_blank">Paper ↗</a>
  </div>
  <div class="nav-right">
    <span class="nav-meta">updated {stamp}</span>
    <a class="nav-gh" href="{OVERLEAF_URL}" target="_blank">Paper ↗</a>
    <a class="nav-gh" href="https://github.com/edward-lcl/feedback-distillation" target="_blank">GitHub ↗</a>
  </div>
</nav>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">Algoverse / JKJ · Active Research</div>
    <h1>Step-Level Feedback Distillation</h1>
    <p class="hero-sub">{html.escape(TAGLINE)}</p>
    <div class="hero-chips">
      <span class="chip chip-green" title="Label→train→eval + best-of-N verifier all built &amp; smoke-tested; runbooks written.">Pipeline built · handoff-ready</span>
      <span class="chip chip-green" title="Privilege helps on MATH but not GSM8K (too easy) or OlympiadBench (too hard) — verified, incl. cross-family.">Sweet spot verified</span>
      <span class="chip chip-blue" title="Teacher chosen by bake-off (Gemma, F1 0.91); result replicates with a Qwen-27B teacher.">Gemma teacher · Qwen cross-family</span>
      <span class="chip chip-yellow" title="Verified N=1000 (Gemma-4 labeling): teacher-level privilege validated, but it does NOT transfer to the 1.5B student (no-GT ≥ priv; neither beats majority vote). Now diagnosing why.">Honest null: no student transfer</span>
      <span class="chip chip-gray">edward-lcl/feedback-distillation</span>
    </div>
  </div>
</div>

<div class="wrap">

<div class="section-label" id="overview">Project overview</div>
<div class="g3">{overview_html}</div>

<div class="callout" style="margin-top:16px">
  <h3>{RESEARCH_Q[0]}</h3>
  <p>{RESEARCH_Q[1]}</p>
</div>

<div class="section-label" id="results">Results — privilege × difficulty</div>
<p style="font-size:13px;color:var(--ink3);margin-bottom:16px">Privilege buys signal only where the teacher can't self-verify, <em>and</em> only when given a full worked solution.</p>
<div class="g3">
  {_gpriv_card(gsm, "GSM8K — easy math", "Gemma · N=50")}
  {_gpriv_card(math_early, "MATH — hard (preliminary)", "Gemma · N=50 · early signal")}
  {_gpriv_card(math, "MATH — hard · N=400 (primary)", "Gemma · definitive · 95% CI [0.01, 0.09]", primary=True)}
</div>

<div class="insight" style="margin-top:16px">
  <strong>Reading across panels:</strong> The N=50 preliminary shows a stronger solution gap (+0.13) than the definitive N=400 run (<strong>+0.05, 95% CI [0.01, 0.09] — significant</strong>) — both point the same direction; N=400 is the paper headline. Within MATH the gap peaks at the intermediate levels (L3 +0.11) and collapses at L1 (−0.13, too easy) and the hardest tail — the sweet spot replicates inside the dataset. The bare-answer row is inert (+0.00, measured at N=150). Core claim is stable: <strong>rich privilege (full worked solution) helps on hard-but-tractable problems; a bare answer does not.</strong>
</div>

<div class="insight" style="margin-top:14px" title="Decomposed from the N=400 per-sample log (experiments/mechanism_analysis.py). Gate 1 isolates WHERE the gain comes from; Gate 2 shows it fades as the reference gets harder to follow.">
  <strong>Mechanism — why privilege helps (N=400):</strong> the gain is a <em>rescue of self-verification failures</em>, not a raw difficulty effect.
  <strong>Gate 1 (needs help):</strong> of 227 error problems, the no-GT teacher <strong>missed 89</strong> → +solution <strong>rescues 29 (33%)</strong>; of the 138 it already caught, +solution <strong>breaks only 17 (12%)</strong> — the benefit concentrates exactly where the teacher can't self-verify.
  <strong>Gate 2 (can use):</strong> rescue rate falls monotonically with reference length (short <strong>0.37</strong> → mid <strong>0.33</strong> → long <strong>0.28</strong>) — the reference only helps when the teacher can follow it. The moderator is <strong>self-verification-failure × tractability</strong>.
</div>

<div class="section-label">Cross-family replication — Qwen-27B</div>
<div class="g2">
  {_gpriv_card(qwen, "Qwen-27B · MATH N=150", "Cross-family confirmation · N=150", running=False)}
  <div class="card">
    <div class="card-title">Why this matters</div>
    <p style="font-size:13px;color:var(--ink2);line-height:1.7;margin-top:8px">If the privilege×difficulty pattern holds for Qwen-27B too, it's not a Gemma artifact — it's a property of step-level feedback under difficulty. Bakeoff F1 (0.873) shows Qwen-27B is a capable teacher. Unblocks Saksham once confirmed.</p>
  </div>
</div>

<div class="section-label" id="decisions">Locked decisions</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="dec-table">
    <tr><th>Item</th><th>Choice</th><th>Why</th></tr>
    {dec_html}
  </table>
</div>

<div class="section-label">Status &amp; roadmap</div>
<div class="g2">
  <div class="card">
    <div class="card-title" style="margin-bottom:14px">Experiment status</div>
    <ul class="status-list">{runs_html}</ul>
  </div>
  <div class="card">
    <div class="card-title" style="margin-bottom:14px"><a href="https://github.com/edward-lcl/feedback-distillation/blob/main/RESEARCH_ROADMAP.md" target="_blank" style="color:inherit;text-decoration:none">Roadmap <span style="color:var(--accent)">↗</span></a></div>
    <div class="milestones">{miles_html}</div>
  </div>
</div>

<div class="section-label" id="tasks">Task board</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="task-table">
    <tr><th>Who</th><th>Track</th><th>Status</th><th>Next action</th></tr>
    {tasks_html}
  </table>
</div>

<div class="section-label">Teacher evaluation — with-GT ceiling, N=30</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
  <table class="data-table">
    <tr><th>Teacher</th><th>F1</th><th>correct_acc</th><th>error_acc</th><th>parse_fail</th></tr>
    {bakeoff_html}
  </table>
  <div style="padding:10px 16px 14px;font-size:12px;color:var(--ink3);border-top:1px solid var(--border)">Official Gemma checkpoint used for the reported labeling pass. Gemma selected: best F1, zero parse failures, fastest throughput.</div>
</div>
{student_html}
<div class="section-label">How we work</div>
<div class="card">
  <table class="kv-table">{how_html}</table>
</div>

<div class="section-label" id="path">Path to submission</div>
<div class="card">
  <ul class="path">{path_html}</ul>
  <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border);font-size:12.5px;color:var(--ink3)">
    <strong style="color:var(--accent)">Saksham — you're up.</strong> The two GPU runs above are the critical path; everything they need is in the runbook. Pipeline is built and smoke-tested, so this is run-and-report, not build.
  </div>
</div>

<div class="section-label">Docs &amp; runbooks</div>
<div class="card"><div class="links">{links_html}</div></div>

<div class="section-label">Frontier — related papers &amp; positioning (mirrors PAPER_FRAMING.md)</div>
<div class="card"><table class="cond-table"><tr><th>action</th><th>paper</th><th>why it matters to US</th></tr>{frontier_html}</table></div>

<div class="section-label" id="next">Bulletproofing &amp; open items</div>
<div class="card"><ul class="status-list">{bullets_html}</ul></div>

</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(htmldoc)
    print(f"wrote {args.out} ({len(htmldoc)} bytes)")


if __name__ == "__main__":
    main()
