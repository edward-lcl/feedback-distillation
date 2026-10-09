"""Read-only independent checks against raw artifacts; no model runs or new inference."""
from pathlib import Path
import json,hashlib,subprocess,collections
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
snap=json.loads((R/'poster/data/figure_inputs/metrics.json').read_text()); report={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'cells':{},'checks':{}}
all_ref=None
for k,entries in snap['cells'].items():
 vals=[]
 for e in entries:
  p=R/e['source']; raw=p.read_bytes(); m=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==snap['sources'][e['source']]['sha256']
  assert subprocess.check_output(['git','show','HEAD:'+e['source']],cwd=R)==raw
  score_path=p.parent/'per_step_scores.json'
  assert subprocess.check_output(['git','show','HEAD:'+str(score_path.relative_to(R))],cwd=R)==score_path.read_bytes()
  s=json.loads(score_path.read_text())
  a=roc_auc_score(s['y_true'],s['y_score']); ap=average_precision_score(s['y_true'],s['y_score'])
  assert abs(a-m['roc_auc'])<1e-12 and abs(ap-m['pr_auc'])<1e-12
  ref=s['y_true']
  if all_ref is None: all_ref=ref
  assert ref==all_ref
  vals.append(a)
 report['cells'][k]={'auc':vals,'mean':float(np.mean(vals)),'range':[min(vals),max(vals)]}
report['evaluation']={'steps':len(all_ref),'errors':sum(all_ref),'sequences':'1000 per evaluation dataset; sequence IDs absent in some score artifacts'}
mean=lambda k:report['cells'][k]['mean']
report['gap_recovery']=(mean('pb_nogt')-mean('raw_nogt'))/(mean('gold')-mean('raw_nogt'))
loadlines=lambda p:[json.loads(x) for x in (R/p).read_text().splitlines()]
report['training']={}
gold=loadlines('results/diagnostics/processbench_gsm8k_gold_train400_steps.jsonl')
identity=lambda x:(x['problem'],x['solution_prefix'],x['step_text'])
for k in ['nogt','priv']:
 raw=loadlines(f'data/labeled/math_{k}_gsm8k400_steps.jsonl'); pb=loadlines(f'data/labeled/math_{k}_gsm8k400_pbformat_steps.jsonl')
 assert [identity(x) for x in raw]==[identity(x) for x in pb]
 gold_exact=collections.Counter(map(identity,raw))==collections.Counter(map(identity,gold))
 records=loadlines(f'data/labeled/math_{k}_gsm8k400.jsonl')
 report['training'][k]={'steps':len(raw),'solutions':len(records),'unique_problem_strings':len(set(x['problem'] for x in raw)),'raw_errors':sum(x['is_error'] for x in raw),'pb_errors':sum(x['is_error'] for x in pb),'changed_error_targets':sum(a['is_error']!=b['is_error'] for a,b in zip(raw,pb)),'raw_pb_traces_identical_order':True,'gold_traces_identical_multiset':gold_exact}
report['training']['gold']={'steps':len(gold),'errors':sum(x['is_error'] for x in gold)}
norm=lambda x:' '.join(x.split())
ident_norm=lambda x:tuple(norm(x[k]) for k in ['problem','solution_prefix','step_text'])
ca=collections.Counter(map(ident_norm,raw)); cb=collections.Counter(map(ident_norm,gold))
report['gold_input_discrepancy']={'normalized_unmatched_generated':sum((ca-cb).values()),'normalized_unmatched_gold':sum((cb-ca).values()),'affected_problem_texts':sorted(set(x[0] for x in ca-cb)|set(x[0] for x in cb-ca))}
eval_rows=loadlines('data/processbench_math_shuffled.jsonl')[:1000]
assert [int(s['is_error']) for row in eval_rows for s in row['steps']]==all_ref
report['evaluation']['sequences']=len(eval_rows)
report['evaluation']['unique_problem_strings']=len(set(x['problem'] for x in eval_rows))
report['evaluation']['exact_train_eval_problem_overlap']=len(set(x['problem'] for x in raw)&set(x['problem'] for x in eval_rows))
h=json.loads((R/'poster/data/historical/provenance.json').read_text()); report['historical']={}
for a in h['artifacts']:
 raw=subprocess.check_output(['git','show',h['git_commit']+':'+a['source']],cwd=R)
 assert raw==(R/a['bundled']).read_bytes()
 d=json.loads(raw); report['historical'][a['source']]={'roc_auc':d['roc_auc'],'n_steps':d.get('n_steps')}
report['manuscripts']=[]
for p in [Path.home()/'Downloads/171_Before_You_Scale_The_Cost_-1.pdf',Path.home()/'Downloads/171_Before_You_Scale_The_Cost_.pdf',Path.home()/'Downloads/Step_Level_Feedback_Distillation-1.pdf',R/'paper/slfd_colm_er_final.pdf']:
 report['manuscripts'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report['checks']={'all_20_metrics_recomputed':True,'auc_ap_match':True,'identical_evaluation_label_vectors':True,'metrics_committed_and_snapshot_hashes_match':True,'historical_git_bytes_match':True}
(O/'evidence-check.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
