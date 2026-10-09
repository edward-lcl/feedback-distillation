#!/usr/bin/env python3
"""Rebuild vector figures from a compact, provenance-linked artifact snapshot.
Run from any directory: python3 poster/scripts/figures.py
Use --refresh to rebuild snapshot from repository results (never synthesizes data).
"""
from pathlib import Path
import os, json, argparse, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/slfd-poster-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent; OUT=BASE/'assets'; INPUTS=BASE/'data'/'figure_inputs'
NAVY='#142D4E'; TEAL='#087F83'; GRAY='#62758A'; LIGHT='#DDE7EC'; RED='#B35143'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':NAVY,'axes.labelcolor':NAVY,'xtick.color':NAVY,'ytick.color':NAVY,'axes.edgecolor':LIGHT,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def snapshot():
 INPUTS.mkdir(parents=True,exist_ok=True)
 d={'schema':'poster-figure-data-v1','sources':{},'cells':{}}
 def read(path):
  p=ROOT/path; raw=p.read_bytes(); d['sources'][path]={'sha256':hashlib.sha256(raw).hexdigest()}; return json.loads(raw)
 for key,stem in [('raw_nogt','generated_nogt_gsm8k400_to_math1000_qwen3b_bce_bal_seed'),('pb_nogt','generated_nogt_gsm8k400_pbformat_to_math1000_qwen3b_bce_bal_seed'),('raw_priv','generated_priv_gsm8k400_to_math1000_qwen3b_bce_bal_seed'),('pb_priv','generated_priv_gsm8k400_pbformat_to_math1000_qwen3b_bce_bal_seed'),('gold','processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed')]:
  d['cells'][key]=[{'seed':i,'roc_auc':read(f'results/diagnostics/{stem}{i}/processbench_results.json')['roc_auc'],'source':f'results/diagnostics/{stem}{i}/processbench_results.json'} for i in range(4)]
 d['teacher_math']=read('results/evidence_pack_n400/evidence_pack.json'); d['teacher_math'].pop('flips',None)
 d['teacher_gsm']=read('results/teacher_eval/privilege_probe.json')
 d['teacher_olympiad']=read('results/teacher_eval_olympiadbench_oe/privilege_probe.json')
 d['cost']={'diagnostic_gpu_hours':.3,'sweep_gpu_hours':15,'status':'approximate manuscript-reported estimates; no runtime ledger located','source':'171_Before_You_Scale_The_Cost_-1.pdf, page 2; paper/slfd_colm_er_final.tex'}
 (INPUTS/'metrics.json').write_text(json.dumps(d,indent=2)+'\n'); return d

def save(fig,name):
 for ext in ['svg','pdf','png']: fig.savefig(OUT/f'figure_{name}.{ext}',dpi=140,facecolor='white')
 plt.close(fig)

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--refresh',action='store_true'); args=parser.parse_args()
 d=snapshot() if args.refresh or not (INPUTS/'metrics.json').exists() else json.loads((INPUTS/'metrics.json').read_text())
 cells={k:np.array([x['roc_auc'] for x in v]) for k,v in d['cells'].items()}; means={k:float(v.mean()) for k,v in cells.items()}
 expected={'raw_nogt':.6183,'pb_nogt':.7366,'raw_priv':.5494,'pb_priv':.6762,'gold':.7515}
 for k,v in expected.items(): assert round(means[k],4)==v,(k,means[k])
 fig,ax=plt.subplots(figsize=(12,5)); fig.subplots_adjust(left=.19,right=.98,bottom=.24,top=.86)
 ax.set_xlim(.45,.82); ax.set_ylim(-.49,1.5); ax.set_yticks([1,0],['No-GT\nteacher','Privileged\nteacher'],fontsize=17,fontweight='bold'); ax.tick_params(axis='y',length=0,pad=14)
 ax.set_xticks(np.arange(.45,.81,.05)); ax.set_xlabel('ProcessBench-MATH ROC-AUC  (higher is better)',labelpad=12,fontsize=17)
 ax.spines['left'].set_visible(False); ax.grid(axis='x',color=LIGHT,lw=.7,zorder=0)
 gold=means['gold']; ax.axvline(gold,color=TEAL,linestyle=(0,(4,3)),lw=1.8,alpha=.8)
 ax.text(gold+.004,.65,f'GSM8K gold\n{gold:.4f}',fontsize=15,color=TEAL,ha='left',va='top',fontweight='bold')
 for y,k in [(1,'nogt'),(0,'priv')]:
  a,b=means['raw_'+k],means['pb_'+k]
  ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops={'arrowstyle':'->','color':TEAL,'lw':3,'shrinkA':11,'shrinkB':11})
  for key,col in [('raw_'+k,GRAY),('pb_'+k,TEAL)]:
   value=means[key]; ax.scatter(value,y,s=155,color=col,zorder=4)
   ax.text(value,y+.14,f'{value:.4f}',ha='center',color=col,fontsize=20,fontweight='bold')
   ax.scatter(cells[key],np.full(4,y-.18),s=28,facecolor='white',edgecolor=col,linewidth=1.1,zorder=4)
  ax.text((a+b)/2,y-.02,f'+{b-a:.4f}',ha='center',va='top',fontsize=14,color=TEAL,bbox={'facecolor':'white','edgecolor':'none','pad':3})
 fig.text(.19,.94,'Raw labels',color=GRAY,fontsize=18,fontweight='bold'); fig.text(.39,.94,'→',color=TEAL,fontsize=18); fig.text(.43,.94,'ProcessBench first-error convention',color=TEAL,fontsize=18,fontweight='bold')
 fig.text(.19,.035,'Large dots: four-seed means. Small dots: individual seeds (not confidence intervals).',fontsize=13)
 save(fig,'rerender')
 # Discrete categories; no smooth inverted-U implying interpolated quantitative evidence.
 fig,ax=plt.subplots(figsize=(7,3)); fig.subplots_adjust(left=.15,right=.98,bottom=.25,top=.79)
 m=d['teacher_math']; gsm=d['teacher_gsm']; delta=[gsm['gap_solution_f1'],m['gap']]
 ax.axhline(0,color=GRAY,lw=1); ax.bar([0,1],delta,color=[GRAY,TEAL],width=.45,zorder=3)
 ax.set_xlim(-.6,1.6); ax.set_ylim(-.085,.115); ax.set_xticks([0,1],['GSM8K · N=50','MATH · N=400'],fontsize=13)
 ax.set_yticks([-.05,0,.05,.10],['−.05','0','+.05','+.10']); ax.set_ylabel('Δ first-error F1',fontsize=13)
 # Draw the complete interval above the bar; otherwise its lower half is hidden.
 ax.errorbar([1],[delta[1]],yerr=[[delta[1]-m['gap_ci95'][0]],[m['gap_ci95'][1]-delta[1]]],fmt='none',ecolor=NAVY,capsize=5,lw=1.5,zorder=5)
 ax.text(0,.012,'−0.052\nnot significant',ha='center',fontsize=12,color=GRAY); ax.text(1.27,.062,'+0.051',fontsize=14,fontweight='bold',color=TEAL)
 fig.text(.03,.98,'Privilege helps within a difficulty window',fontsize=17,fontweight='bold',va='top')
 fig.text(.03,.045,'OlympiadBench: Δ ≈ −0.03, n.s.; separate scoring protocol.\nMATH whisker: paired 95% bootstrap CI; none inferred for GSM8K.',fontsize=10.5)
 save(fig,'teacher')
 fig,ax=plt.subplots(figsize=(7,2.4)); fig.subplots_adjust(left=.30,right=.87,bottom=.29,top=.84)
 ax.barh([1,0],[.3,15],height=.44,color=[TEAL,NAVY]); ax.set_xlim(0,16)
 ax.set_yticks([1,0],['One diagnostic','Recipe sweep'],fontsize=13); ax.set_xticks([0,5,10,15]); ax.set_xlabel('GPU-hours',fontsize=12)
 ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
 for y,v in [(1,.3),(0,15)]: ax.text(v+.25,y,f'≈{v:g}',va='center',fontweight='bold',fontsize=16)
 fig.text(.03,.98,'Check supervision before sweeping recipes',fontsize=17,fontweight='bold',va='top')
 fig.text(.03,.02,'Paper-reported estimates in this setup; diagnostic = one 500-step run.',fontsize=10.5)
 save(fig,'cost')
 # Explicit conceptual example, implemented convention exactly matches rerender_record.
 fig,ax=plt.subplots(figsize=(10,3.2)); ax.set_xlim(0,10); ax.set_ylim(0,3.2); ax.axis('off'); fig.subplots_adjust(0,0,1,1)
 ax.text(.15,2.95,'Same trace. A different target convention.',fontsize=19,fontweight='bold')
 ax.text(.15,2.60,'Conceptual illustration · not an experimental example',fontsize=12,color=GRAY)
 headers=['Step','Trace (unchanged)','Raw score','PB score / feedback']
 xs=[.2,1.1,5.15,6.7]
 for x,h in zip(xs,headers): ax.text(x,2.16,h,fontsize=13,fontweight='bold')
 rows=[('1','2x = 6','+1','+1   Correct.'),('2','x = 2','−1','−1   Error.   ← first error'),('3','Thus x + 1 = 3','−0.5','+1   Correct.*')]
 for i,row in enumerate(rows):
  y=1.76-i*.42
  ax.add_patch(FancyBboxPatch((.12,y-.12),9.75,.36,boxstyle='round,pad=0.025',facecolor=('#EEF6F5' if i!=1 else '#F9EFEC'),edgecolor='none'))
  for x,txt in zip(xs,row): ax.text(x,y,txt,fontsize=14,color=RED if i==1 else NAVY)
 ax.text(.15,.34,'*Later steps become non-error targets by convention; “Correct.” is a label,',fontsize=12,fontweight='bold')
 ax.text(.15,.11,'  not a claim that the step is mathematically correct. Free-text feedback is standardized.',fontsize=12)
 save(fig,'convention')
 print(json.dumps({'means':means,'gap_recovery':(means['pb_nogt']-means['raw_nogt'])/(means['gold']-means['raw_nogt'])},indent=2))
if __name__=='__main__': main()
