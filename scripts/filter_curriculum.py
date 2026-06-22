import json
import argparse
import os

def flatten_labeled_records(records):
    flat = []
    for rec in records:
        problem = rec.get("problem", "")
        prefix = ""
        for step in rec.get("steps", []):
            text = step.get("text", "")
            if step.get("parse_failed") or step.get("score") is None:
                prefix += text + "\n"
                continue
            if not problem or not text:
                prefix += text + "\n"
                continue
            flat.append({
                "problem": problem,
                "solution_prefix": prefix,
                "step_text": text,
                "score": float(step.get("score", 0.0)),
                "feedback": step.get("feedback", ""),
                "is_error": bool(step.get("is_error", False)),
            })
            prefix += text + "\n"
    return flat

def filter_curriculum(priv_path, nogt_path, out_path, keep="priv"):
    print(f"Loading Privileged labels from: {priv_path}")
    print(f"Loading No-GT labels from: {nogt_path}")
    
    with open(priv_path, 'r') as f:
        priv_flat = flatten_labeled_records([json.loads(line) for line in f if line.strip()])
    with open(nogt_path, 'r') as f:
        nogt_flat = flatten_labeled_records([json.loads(line) for line in f if line.strip()])
        
    # Match by key
    def make_key(ex):
        return (ex["problem"], ex["solution_prefix"], ex["step_text"])
        
    priv_map = {make_key(ex): ex for ex in priv_flat}
    
    curriculum = []
    disagreements = 0
    agreements = 0
    
    for nogt_ex in nogt_flat:
        k = make_key(nogt_ex)
        if k in priv_map:
            priv_ex = priv_map[k]
            if priv_ex["score"] != nogt_ex["score"]:
                disagreements += 1
                curriculum.append(priv_ex if keep == "priv" else nogt_ex)
            else:
                agreements += 1
                
    with open(out_path, 'w') as out_f:
        for ex in curriculum:
            out_f.write(json.dumps(ex) + "\n")
            
    print(f"Total overlapping steps: {agreements + disagreements}")
    print(f"Disagreements (Curriculum size): {disagreements} ({(disagreements/(agreements+disagreements))*100:.1f}%)")
    print(f"Saved to: {out_path} (Using {keep} labels)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priv", default="data/labeled/math_priv.jsonl")
    parser.add_argument("--nogt", default="data/labeled/math_nogt.jsonl")
    parser.add_argument("--out", default="data/labeled/math_curriculum.jsonl")
    parser.add_argument("--keep", choices=["priv", "nogt"], default="priv", help="Which teacher's labels to use for the curriculum steps")
    args = parser.parse_args()
    
    filter_curriculum(args.priv, args.nogt, args.out, args.keep)
