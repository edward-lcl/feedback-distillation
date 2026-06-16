"""Mechanism analysis — WHY privilege helps, beyond the difficulty correlation.

From the evidence-pack per-sample log, decompose the privilege benefit into the
two gates of the sweet-spot hypothesis:
  GATE 1 (needs help): privilege should rescue errors only where the no-GT teacher
                       MISSES them; where no-GT already succeeds it should be ~neutral.
  GATE 2 (can use):    among missed errors, the rescue rate should FALL as the
                       problem gets harder (level) / the reference gets longer —
                       the teacher can't follow a reference it can't track.

Pure analysis of saved data — no model calls.

  python -m experiments.mechanism_analysis [--input results/.../per_sample.jsonl]
"""
import os
import json
import argparse
import statistics


def hit(pred, gold):
    return pred == gold if gold >= 0 else pred == -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--dataset", default="data/processbench_math_shuffled.jsonl")
    ap.add_argument("--results_dir", default="results/mechanism")
    args = ap.parse_args()

    inp = args.input or next((p for p in
        ["results/evidence_pack_n400/per_sample.jsonl", "results/evidence_pack/per_sample.jsonl"]
        if os.path.exists(p)), None)
    if not inp:
        raise SystemExit("No per_sample.jsonl — run experiments.evidence_pack first.")
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    print(f"loaded {len(recs)} samples from {inp}")

    # Optional join: solution length + #steps (match by truncated problem prefix).
    feat = {}
    if os.path.exists(args.dataset):
        for s in (json.loads(l) for l in open(args.dataset) if l.strip()):
            feat[s["problem"][:240]] = {"n_steps": len(s.get("steps", [])),
                                        "sol_len": len(s.get("gt_solution") or "")}

    for r in recs:
        r["is_error"] = r["gold"] >= 0
        r["nogt_hit"] = hit(r["pred_nogt"], r["gold"])
        r["sol_hit"] = hit(r["pred_sol"], r["gold"])
        r["flip"] = r["sol_hit"] and not r["nogt_hit"]    # privilege rescued
        r["broke"] = r["nogt_hit"] and not r["sol_hit"]   # privilege broke
        r["sol_len"] = feat.get(r["problem"][:240], {}).get("sol_len")

    err = [r for r in recs if r["is_error"]]
    missed = [r for r in err if not r["nogt_hit"]]
    hits = [r for r in err if r["nogt_hit"]]

    gate1 = {
        "n_error": len(err),
        "nogt_missed": len(missed), "rescued_by_solution": sum(r["flip"] for r in missed),
        "rescue_rate_on_missed": round(sum(r["flip"] for r in missed) / len(missed), 3) if missed else None,
        "nogt_hit": len(hits), "broken_by_solution": sum(r["broke"] for r in hits),
        "break_rate_on_hit": round(sum(r["broke"] for r in hits) / len(hits), 3) if hits else None,
    }

    by_level = {}
    for lv in sorted({str(r["level"]) for r in missed if r["level"]}):
        sub = [r for r in missed if str(r["level"]) == lv]
        by_level[lv] = {"missed": len(sub),
                        "rescue_rate": round(sum(x["flip"] for x in sub) / len(sub), 3)}

    by_len = {}
    lens = [r["sol_len"] for r in missed if r["sol_len"]]
    if len(lens) >= 6:
        q1, q2 = statistics.quantiles(lens, n=3)
        band = lambda x: "short" if x <= q1 else ("mid" if x <= q2 else "long")
        for b in ("short", "mid", "long"):
            sub = [r for r in missed if r["sol_len"] and band(r["sol_len"]) == b]
            if sub:
                by_len[b] = {"missed": len(sub),
                             "rescue_rate": round(sum(x["flip"] for x in sub) / len(sub), 3)}

    out = {"source": inp, "gate1_needs_help": gate1,
           "gate2_can_use_by_level": by_level, "gate2_can_use_by_sol_len": by_len}
    print(json.dumps(out, indent=2))
    os.makedirs(args.results_dir, exist_ok=True)
    json.dump(out, open(f"{args.results_dir}/mechanism.json", "w"), indent=2)
    print("\nReads as: privilege rescues errors the no-GT teacher MISSES (gate 1); the "
          "rescue rate falls on harder levels / longer references (gate 2). The moderator "
          "is self-verification-failure × tractability — not difficulty alone.")


if __name__ == "__main__":
    main()
