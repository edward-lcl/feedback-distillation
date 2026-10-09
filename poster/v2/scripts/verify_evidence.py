"""Verify bundled source bytes and V2 numerical preservation without repository access."""
from pathlib import Path
import json,hashlib,statistics
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'data/figure_inputs/metrics.json').read_text())
checked=[]
for name,info in d['sources'].items():
 p=R/'data/figure_inputs'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],name
 checked.append(name)
means={}
for key,rows in d['cells'].items():
 for row in rows:
  m=json.loads((R/'data/figure_inputs'/row['source']).read_text())
  assert row['roc_auc']==m['roc_auc']
 means[key]=statistics.mean(row['roc_auc'] for row in rows)
expected={'raw_nogt':.6183,'pb_nogt':.7366,'raw_priv':.5494,'pb_priv':.6762,'gold':.7515}
assert {k:round(v,4) for k,v in means.items()}==expected
recovery=(means['pb_nogt']-means['raw_nogt'])/(means['gold']-means['raw_nogt'])
assert round(recovery*100,1)==88.8
m=d['teacher_math']; assert m['gap']==.0506 and m['gap_ci95']==[.0095,.0925]
original=json.loads((R/'provenance/independent-review/evidence-check.json').read_text())
for k,rows in d['cells'].items():assert [r['roc_auc'] for r in rows]==original['cells'][k]['auc']
report={'source_files_hash_verified':len(checked),'seed_values_identical_to_independent_review':20,'means':means,'descriptive_gap_fraction':recovery,'teacher_gap':m['gap'],'teacher_ci95':m['gap_ci95'],'no_new_training_or_inference':True}
(R/'provenance/numerical_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
