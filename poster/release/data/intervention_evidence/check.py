"""Recompute target and input-identity checks from bundled hash-only records."""
from pathlib import Path
from collections import Counter
import json
D=Path(__file__).resolve().parent
load=lambda name: [json.loads(x) for x in (D/(name+'.jsonl')).read_text().splitlines()]
gold=load('gold'); report={'gold':{'steps':len(gold),'errors':sum(x['error'] for x in gold)},'generated':{}}
for condition in ['nogt','priv']:
 raw,pb=load('raw_'+condition),load('pb_'+condition)
 assert [x['input_sha256'] for x in raw]==[x['input_sha256'] for x in pb]
 a,b=Counter(x['normalized_input_sha256'] for x in raw),Counter(x['normalized_input_sha256'] for x in gold)
 report['generated'][condition]={'steps':len(raw),'raw_errors':sum(x['error'] for x in raw),'pb_errors':sum(x['error'] for x in pb),'changed_targets':sum(a['error']!=b['error'] for a,b in zip(raw,pb)),'raw_pb_inputs_identical_in_order':True,'normalized_unmatched_generated':sum((a-b).values()),'normalized_unmatched_gold':sum((b-a).values())}
print(json.dumps(report,indent=2))
