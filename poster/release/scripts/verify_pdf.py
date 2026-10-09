"""Structural, rendered QR and vector checks for the delivered poster."""
from pathlib import Path
import json, hashlib
import fitz
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT/'final-print.pdf'
doc = fitz.open(pdf)
assert len(doc) == 1
page = doc[0]
assert tuple(page.rect) == (0., 0., 2448., 2448.)
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
outside = [s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
assert not outside, outside
pdf_text=page.get_text()
assert 'CSU' not in pdf_text and 'CSU' not in json.dumps(doc.metadata)
assert 'XJTLU' in pdf_text and 'XJTLU' in json.dumps(doc.metadata)
for value in ['0.726','0.777','0.631','0.641','0.6183','0.7366','0.5494','0.6762','0.7515','88.8%']:
 assert value in pdf_text, value
teacher_spans=[s for s in spans if s['bbox'][0]>=1540 and s['bbox'][1]>=1735 and s['bbox'][3]<=2185]
assert min(s['size'] for s in teacher_spans)>=24

url = 'https://openreview.net/forum?id=gKNwyhDxPk'
links = [l for l in page.get_links() if l.get('uri') == url]
assert links
qr_crop = fitz.Rect(links[0]['from'])
results = []
for dpi in (72,150,300):
 pix = page.get_pixmap(clip=qr_crop, dpi=dpi, alpha=False)
 arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height,pix.width,3)
 decoded = cv2.QRCodeDetector().detectAndDecode(arr)[0]
 normalized = False
 if not decoded:
  arr = cv2.resize(arr,(400,400),interpolation=cv2.INTER_AREA)
  decoded = cv2.QRCodeDetector().detectAndDecode(arr)[0]
  normalized = True
 assert decoded == url, (dpi,decoded)
 results.append({'dpi':dpi,'decoded':decoded,'decoder_resized_to_400px':normalized})
fonts = []
for f in page.get_fonts(full=True):
 xref=f[0]; name,ext,kind,data = doc.extract_font(xref)
 assert data, name
 fonts.append({'name':name,'type':kind,'embedded_bytes':len(data)})
figures=[]
for p in sorted((ROOT/'assets').glob('figure_*.pdf')):
 d=fitz.open(p)
 assert all(len(pg.get_images()) == 0 for pg in d)
 figures.append(p.name)
images=[]
for item in page.get_images(full=True):
 rects=page.get_image_rects(item[0]); images.append({'width_px':item[2],'height_px':item[3],'effective_ppi_x':round(item[2]/(rects[0].width/72),1)})
report={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'teacher_min_font_pt':min(s['size'] for s in teacher_spans),'metadata':doc.metadata,'affiliation_author_approved_xjtlu_only':True,'page_inches':[34,34],'page_count':len(doc),'bytes':pdf.stat().st_size,'text_spans_outside_page':outside,'font_size_range_pt':[min(s['size'] for s in spans),max(s['size'] for s in spans)],'embedded_fonts':fonts,'raster_images':images,'vector_figure_pdfs':figures,'qr_checks':results,'pdf_links':[l.get('uri') for l in page.get_links()]}
(ROOT/'provenance/pdf_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
