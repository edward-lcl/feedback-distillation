"""Render complete V2 composition plus requested 150-dpi detail crops."""
from pathlib import Path
import fitz
from PIL import Image
R=Path(__file__).resolve().parents[1]
p=fitz.open(R/'final-print.pdf')[0]
p.get_pixmap(matrix=fitz.Matrix(2000/2448,2000/2448),alpha=False).save(R/'final-preview.png')
Image.open(R/'final-preview.png').convert('L').save(R/'grayscale-preview.png')
(R/'review-crops').mkdir(exist_ok=True)
for name,box in {'intervention':(1660,855,2398,1705),'central-chart':(50,855,1625,1705),'teacher':(1540,1735,2398,2185),'footnotes':(50,2215,1970,2405),'gold-reference':(50,1735,950,2185),'compute':(985,1735,1510,2185)}.items():
 p.get_pixmap(clip=fitz.Rect(box),dpi=150,alpha=False).save(R/'review-crops'/f'{name}.png')
print('Rendered composition, grayscale, and six close-up crops.')
