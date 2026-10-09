"""Generate a vector QR with quiet zone, then independently decode PNG preview."""
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://openreview.net/forum?id=gKNwyhDxPk'
params = cv2.QRCodeEncoder_Params()
params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_M
qr = cv2.QRCodeEncoder_create(params).encode(URL)
# Encoder includes two quiet modules; add four more for a generous border.
qr = np.pad(qr, 4, constant_values=255)
n = len(qr)
rects = ''.join(f'<rect x="{x}" y="{y}" width="1" height="1"/>' for y,x in zip(*np.where(qr == 0)))
(ROOT/'assets/qr-paper.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" shape-rendering="crispEdges"><rect width="{n}" height="{n}" fill="white"/><g fill="#142D4C">{rects}</g></svg>')
preview = cv2.resize(qr,None,fx=12,fy=12,interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT/'assets/qr-paper.png'),preview)
assert cv2.QRCodeDetector().detectAndDecode(preview)[0] == URL
print('QR generated and decoded:',URL)
