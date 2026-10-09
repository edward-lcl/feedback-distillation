"""Compare release with frozen reviewed source/PDF; no scientific edits allowed."""
from pathlib import Path
import hashlib,json
import fitz
import numpy as np
R=Path(__file__).resolve().parents[1];B=R/'provenance/reviewed-v2'
before=(B/'poster.typ').read_text();after=(R/'poster.typ').read_text()
expected=before.replace('Before You Scale — COLM poster V2 candidate','Before You Scale — COLM poster V2').replace("Scientifically qualified V2 poster candidate. Edward affiliation: Xi'an Jiaotong-Liverpool University (XJTLU). Pending independent re-review.","COLM 2026 Workshop on Efficient Reasoning poster. Edward affiliation: Xi'an Jiaotong-Liverpool University (XJTLU).").replace('V2 candidate · pending focused independent verification.','COLM 2026 · V2.')
assert after==expected
changed=[i+1 for i,(a,b) in enumerate(zip(before.splitlines(),after.splitlines())) if a!=b]
assert changed==[6,142]
a=fitz.open(B/'final-print.pdf');b=fitz.open(R/'final-print.pdf')
at,bt=a[0].get_text(),b[0].get_text()
assert at.replace('V2 candidate · pending focused independent verification.','COLM 2026 · V2.')==bt
assert a.metadata['author']==b.metadata['author']
assert [(l['uri'],tuple(l['from'])) for l in a[0].get_links()]==[(l['uri'],tuple(l['from'])) for l in b[0].get_links()]
oldline=[x for x in a[0].get_text('blocks') if 'pending focused independent verification' in x[4]][0]
allowed=fitz.Rect(oldline[:4]);allowed.x0-=2;allowed.y0-=2;allowed.x1+=2;allowed.y1+=2
pixel=[]
for dpi in (72,150):
 pa=a[0].get_pixmap(dpi=dpi,alpha=False);pb=b[0].get_pixmap(dpi=dpi,alpha=False)
 aa=np.frombuffer(pa.samples,np.uint8).reshape(pa.height,pa.width,3);bb=np.frombuffer(pb.samples,np.uint8).reshape(pb.height,pb.width,3)
 yy,xx=np.where(np.any(aa!=bb,axis=2));scale=dpi/72
 bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
 assert allowed.contains(fitz.Rect([v/scale for v in bbox]))
 pixel.append({'dpi':dpi,'changed_pixel_bounds':bbox,'changes_confined_to_acknowledgement_line':True})
report={'changed_source_lines':changed,'source_matches_exact_status_only_replacements':True,'all_other_extracted_text_identical':True,'author_metadata_identical':True,'links_and_rectangles_identical':True,'pixel_checks':pixel,'reviewed_pdf_sha256':hashlib.sha256((B/'final-print.pdf').read_bytes()).hexdigest(),'release_pdf_sha256':hashlib.sha256((R/'final-print.pdf').read_bytes()).hexdigest()}
(R/'provenance/minimal_diff_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
