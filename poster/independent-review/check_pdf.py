"""Independent PDF checks; writes only this review's receipts/previews."""
from pathlib import Path
import fitz,cv2,numpy as np,json,hashlib
O=Path(__file__).resolve().parent; R=O.parent
d=fitz.open(R/'final-print.pdf'); old=fitz.open(O/'original/final-print.pdf'); p=d[0]
assert len(d)==1 and tuple(p.rect)==(0,0,2448,2448)
assert p.get_text()==old[0].get_text(), 'Poster text changed'
spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
outside=[s['text'] for s in spans if not p.rect.contains(fitz.Rect(s['bbox']))];assert not outside
fonts=[]
for f in p.get_fonts(full=True):
 n,e,k,data=d.extract_font(f[0]);assert data;fonts.append({'name':n,'embedded':True})
url='https://openreview.net/forum?id=gKNwyhDxPk';link=[x for x in p.get_links() if x.get('uri')==url];assert len(link)==1
qr=[]
for dpi in [72,150,300]:
 pix=p.get_pixmap(clip=link[0]['from'],dpi=dpi,alpha=False); a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
 v=cv2.QRCodeDetector().detectAndDecode(a)[0];native=v==url
 if not native:v=cv2.QRCodeDetector().detectAndDecode(cv2.resize(a,(400,400),interpolation=cv2.INTER_AREA))[0]
 assert v==url;qr.append({'dpi':dpi,'native_decode':native,'decoded':v})
# Reopen both documents symmetrically after QR rasterization to avoid renderer cache effects.
old=fitz.open(O/'original/final-print.pdf'); fresh=fitz.open(R/'final-print.pdf')
a=old[0].get_pixmap(alpha=False);b=fresh[0].get_pixmap(alpha=False)
a=np.frombuffer(a.samples,np.uint8).reshape(a.height,a.width,3);b=np.frombuffer(b.samples,np.uint8).reshape(b.height,b.width,3)
y,x=np.where(np.any(a!=b,axis=2));box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
assert 1560<box[0] and box[2]<2400 and 1650<box[1] and box[3]<2160,box
p.get_pixmap(clip=fitz.Rect(1560,1690,2400,2160),matrix=fitz.Matrix(1.1,1.1)).save(O/'teacher-fixed-crop.png')
pix=p.get_pixmap(matrix=fitz.Matrix(.6,.6),colorspace=fitz.csGRAY);pix.save(O/'grayscale.png')
logo=[]
for im in p.get_images(full=True):
 rect=p.get_image_rects(im[0])[0];logo.append({'pixels':[im[2],im[3]],'ppi':im[2]*72/rect.width})
report={'page_inches':[34,34],'page_count':1,'text_unchanged':True,'text_outside_page':outside,'fonts':fonts,'font_pt_min':min(s['size'] for s in spans),'qr':qr,'raster_images':logo,'changed_pixel_bbox_at_72dpi':box,'sha256_before':hashlib.sha256((O/'original/final-print.pdf').read_bytes()).hexdigest(),'sha256_after':hashlib.sha256((R/'final-print.pdf').read_bytes()).hexdigest(),'physical_print_or_phone_scan_tested':False}
(O/'pdf-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
