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
    ("🧩 What we're building",
     "A small <b>ground-truth-free student</b> that reads a math solution and flags <i>which step</i> the reasoning first goes wrong — not just whether the final answer is right. It learns this from a stronger <b>privileged teacher</b> that sees the reference solution while labeling."),
    ("🔄 Why it was revived",
     "The earlier version stalled (CLEAR + TinyLlama; weak scorers, gibberish on math). Revived <b>June 2026</b> with a shift to <b>step-level reasoning evaluation</b> over final-answer accuracy. Target: a publishable result for the <b>COLM workshop (~early July)</b> — flexible; quality over the date."),
    ("📍 Where we are today",
     "<b>Phase-0 teacher gate complete.</b> Teacher, dataset, and privilege form are all chosen <i>by data</i>, and the core claim is validated. Next: confirm it generalizes across model families, fix the student trainer, then hand labeling to the GPU box."),
]

RESEARCH_Q = ("The research question",
    "Does distilling a teacher's <b>step-level score + natural-language critique</b> — and its <b>privileged</b> (answer-aware) judgment — into a small answer-blind student make it better at catching reasoning-step errors? <b>Finding so far:</b> privilege helps, but only when the problem is <b>hard</b> and the privilege is <b>rich</b> (a worked solution, not a bare answer).")

PHASE = "Phase 0 (teacher gate) complete — research design validated by data"

RUNS = [  # (label, state) state in {running, done, queued}
    ("Teacher bake-off (5 models, N=30)", "done"),
    ("Privilege × difficulty (Gemma · GSM8K + MATH N=150)", "done"),
    ("Cross-teacher replication (Qwen-27B · MATH N=150)", "crashed"),
]

DECISIONS = [
    ("Teacher", "Gemma-4-26b (official Gemma-class)",
     "F1 0.91, zero parse failures, ~12× faster than reasoning models, cross-family"),
    ("Dataset", "ProcessBench MATH",
     "GSM8K is saturated — the privilege gap is ≈0 there"),
    ("Privilege signal", "Full worked reference solution",
     "+0.07 F1 on MATH (N=150). A bare answer flips ZERO predictions — the teacher needs a reference reasoning trace, not a number."),
    ("Core claim", "VALIDATED — rich privilege helps on hard problems",
     "≈0 on GSM8K and for bare answers everywhere; +0.07 for the full solution on MATH"),
]

TASKS = [  # who, track, status (active|blocked|queued|done), next action
    ("Edward", "Orchestration + trainer", "active",
     "Fix slfd_trainer: wire real LoRA (currently full-FT), score-head reads the boundary token. Then design the score-vs-critique student ablation."),
    ("Saksham", "GPU pipeline", "blocked",
     "Official Gemma via vLLM → label MATH train set with SOLUTION-privilege → train GT-free student → eval on ProcessBench MATH. Unblocks once cross-teacher replication confirms."),
    ("Henry", "Research / paper", "active",
     "Draft Related Work (PRM lit positioning) + results narrative around the GSM8K-vs-MATH table. See HANDOFF_HENRY.md on GitHub. Blocking on: by-level breakdown + flip examples from Edward."),
]

HOW_WE_WORK = [
    ("Cadence", "Weekly <b>Sunday</b> syncs · ~10 hrs/week each · Slack between meetings"),
    ("Compute", "Two-phase: quick runs on <b>Apple Silicon</b> (M-series, 48 GB) → full runs on <b>2×3090 (48 GB VRAM)</b> via SSH"),
    ("Code", "<b>GitHub only</b> — single repo <code>edward-lcl/feedback-distillation</code>, frequent commits, no Colab sprawl"),
    ("Method", "Structured (Pydantic) step feedback · chain-of-thought evaluation over final-answer accuracy · Co-Scientist-style hypothesis iteration"),
]

BULLETPROOFING = [
    ("todo", "Restart cross-teacher replication (Qwen-27B · MATH N=150) — oMLX ChunkedEncodingError, no data saved."),
    ("todo", "Spot-check why with_answer ≡ no_gt at N=150 (wiring confirmed; behavior is real)."),
    ("todo", "Use the OFFICIAL Gemma checkpoint (not the abliterated community quant) for the reported labeling pass."),
    ("idea", "A harder set (OlympiadBench) may widen the solution gap — optional extension."),
]

MILESTONES = [
    ("done", "Teacher + dataset + privilege form locked by data"),
    ("now", "Confirm generalization (cross-teacher) + fix student trainer"),
    ("next", "Hand MATH labeling to Saksham (GPU)"),
    ("then", "Score-vs-critique student ablation → paper draft"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    stamp = args.stamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    gsm = _load(f"{RESULTS}/teacher_eval/privilege_probe.json")
    math = _load(f"{RESULTS}/teacher_eval_math_n150/privilege_probe.json") or _load(f"{RESULTS}/teacher_eval_math/privilege_probe.json")
    qwen = _load(f"{RESULTS}/teacher_eval_math_qwen27b/privilege_probe.json")

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

    css = """
 :root{--bg:#0f1729;--panel:#fff;--line:#e6e8ee;--ink:#101522;--mut:#667085;--accent:#3b5bdb;--accent2:#0a7a2a}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:#eef1f6}
 .wrap{max-width:1080px;margin:0 auto;padding:0 18px 70px}
 nav{position:sticky;top:0;z-index:9;background:rgba(15,23,41,.97);backdrop-filter:blur(6px);color:#cdd5e6;display:flex;gap:18px;flex-wrap:wrap;align-items:center;padding:11px 18px;font-size:13.5px}
 nav b{color:#fff;font-size:15px;margin-right:6px} nav a{color:#aab4cc;text-decoration:none} nav a:hover{color:#fff}
 header{background:linear-gradient(135deg,#1e2a52,#0f1729);color:#fff;padding:30px 22px;border-radius:0 0 16px 16px;margin-bottom:18px}
 header h1{margin:0 0 6px;font-size:27px} header .tag{color:#c7d0e8;max-width:760px;font-size:15px}
 header .meta{margin-top:12px;font-size:13px;color:#9fb0d6}
 h2{font-size:18px;margin:34px 0 12px;display:flex;align-items:center;gap:8px}
 h3{margin:0 0 5px;font-size:15px}
 a{color:var(--accent)} .muted{color:var(--mut)} .small{font-size:13.3px}
 .statuschip{display:inline-block;background:#d8f5dd;color:#0a7a2a;font-weight:700;padding:4px 12px;border-radius:20px;font-size:13px;margin-top:10px}
 .ocards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
 .ocard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;border-top:3px solid var(--accent)}
 .qbox{background:#eef3ff;border:1px solid #c9d8ff;border-left:5px solid var(--accent);border-radius:10px;padding:14px 18px}
 .qbox h3{color:var(--accent)}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px 17px;box-shadow:0 1px 2px rgba(16,22,40,.04)}
 table{border-collapse:collapse;width:100%;margin:4px 0;font-size:13.4px}
 th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
 th{color:var(--mut);font-weight:600}
 .badge{padding:2px 9px;border-radius:11px;font-weight:700;font-size:12.5px}
 .pos{background:#d8f5dd;color:#0a7a2a}.neg{background:#fde0e0;color:#b32020}.zero{background:#ececec;color:#666}
 .pill{display:inline-block;padding:1px 9px;border-radius:11px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
 .running,.now{background:#fff3cd;color:#8a6d00}.done{background:#d8f5dd;color:#0a7a2a}
 .active{background:#d6e6ff;color:#1a4fb4}.blocked{background:#fde0e0;color:#b32020}
 .queued,.todo,.next,.then,.idea{background:#eceef2;color:#555}
 ul.clean{list-style:none;padding:0;margin:6px 0} ul.clean li{padding:3px 0}
 .miles{display:flex;gap:8px;flex-wrap:wrap}
 .mile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 12px;display:flex;flex-direction:column;gap:5px;flex:1;min-width:155px}
 .mile span{font-size:13px}
 @media(max-width:760px){.grid2,.ocards{grid-template-columns:1fr}}
"""

    htmldoc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>SLFD — project dashboard</title><style>{css}</style></head><body>
<nav><b>SLFD</b>
 <a href="#start">Start here</a><a href="#result">Result</a><a href="#decisions">Decisions</a>
 <a href="#tasks">Tasks</a><a href="#how">How we work</a><a href="#next">What's next</a>
 <a href="https://github.com/edward-lcl/feedback-distillation" target=_blank>GitHub ↗</a></nav>
<header>
 <h1>Step-Level Feedback Distillation</h1>
 <div class=tag>{html.escape(TAGLINE)}</div>
 <div class=statuschip>✅ {html.escape(PHASE)}</div>
 <div class=meta>team dashboard · updated {stamp} · repo <code>edward-lcl/feedback-distillation</code></div>
</header>
<div class=wrap>

<h2 id=start>🚀 Start here</h2>
<div class=ocards>{overview}</div>
<div class=qbox style="margin-top:14px"><h3>🎯 {RESEARCH_Q[0]}</h3><p class=small style="margin:0">{RESEARCH_Q[1]}</p></div>

<h2 id=result>★ Headline result — privilege × difficulty</h2>
<p class='muted small'>Read the gaps: a green badge = privilege helped. Privilege buys signal only where the teacher can't already self-verify.</p>
<div class=grid2>
 {privilege_card(gsm, "GSM8K — easy", "Gemma teacher · N=50")}
 {privilege_card(math, "MATH — hard competition", "Gemma teacher · N=150")}
</div>
<div class=grid2 style="margin-top:14px">
 {privilege_card(qwen, "MATH — cross-family check", "Qwen-27B teacher · N=150", running=True)}
 <div class=card><h3>How to read this</h3><p class=small>GSM8K: every privilege level ≈ no-GT → saturated, nothing to add.<br>MATH: a bare answer still ≈ no-GT, but the <b>full solution</b> lifts error recall (0.58→0.65). Privilege buys signal exactly where self-verification fails.</p></div>
</div>

<h2 id=decisions>✅ Locked decisions <span class='muted small'>(settled — don't relitigate)</span></h2>
<div class=card><table><tr><th>item</th><th>choice</th><th>why</th></tr>{dec}</table></div>

<h2>🗺️ Status &amp; roadmap</h2>
<div class=grid2>
 <div class=card><h3>Currently running</h3><ul class=clean>{runs}</ul></div>
 <div class=card><h3>Roadmap</h3><div class=miles>{miles}</div></div>
</div>

<h2 id=tasks>👥 Task board</h2>
<div class=card><table><tr><th>who</th><th>track</th><th>status</th><th>next action</th></tr>{tasks}</table></div>

<h2>🔬 Teacher bake-off <span class='muted small'>(with-GT ceiling, N=30)</span></h2>
<div class=card><table><tr><th>teacher</th><th>F1</th><th>correct</th><th>error_acc</th><th>parse</th></tr>{bakeoff_rows()}</table></div>

<h2 id=how>📋 How we work</h2>
<div class=card><table>{how}</table></div>

<h2 id=next>🧰 What's next &amp; handoff gates</h2>
<div class=card><ul class=clean>{bullets}</ul>
<p class='small muted' style='margin-top:8px'>Handoff: <b>labeling</b> ready for Saksham once cross-teacher confirms · <b>training</b> blocked on trainer fixes (Edward).</p>
<p class=small style='margin-top:6px'>📄 <a href="https://github.com/edward-lcl/feedback-distillation/blob/main/HANDOFF_HENRY.md" target="_blank">Henry handoff on GitHub</a></p></div>

</div></body></html>"""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(htmldoc)
    print(f"wrote {args.out} ({len(htmldoc)} bytes)")


if __name__ == "__main__":
    main()
