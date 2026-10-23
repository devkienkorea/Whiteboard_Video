"""Vẽ 1 ảnh doodle mẫu (nhà, cây, mặt trời, người que) để test khi chưa có model sinh ảnh."""
import sys

import cv2
import numpy as np

W, H = 1000, 1100
img = np.full((H, W, 3), 255, np.uint8)
K, T, AA = (20, 20, 20), 7, cv2.LINE_AA


def shape(pts, fill):
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], fill, AA)
    cv2.polylines(img, [pts], True, K, T, AA)


# mặt trời
cv2.circle(img, (800, 180), 85, (60, 200, 250), -1, AA)
cv2.circle(img, (800, 180), 85, K, T, AA)
for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    p0 = (int(800 + 115 * np.cos(a)), int(180 + 115 * np.sin(a)))
    p1 = (int(800 + 160 * np.cos(a)), int(180 + 160 * np.sin(a)))
    cv2.line(img, p0, p1, K, T, AA)
# đám mây
for cx, cy, r in ((180, 170, 60), (250, 140, 75), (330, 175, 58)):
    cv2.circle(img, (cx, cy), r, (250, 225, 200), -1, AA)
for cx, cy, r in ((180, 170, 60), (250, 140, 75), (330, 175, 58)):
    cv2.circle(img, (cx, cy), r, K, T, AA)
for cx, cy, r in ((180, 170, 60), (250, 140, 75), (330, 175, 58)):
    cv2.circle(img, (cx, cy), r - T, (250, 225, 200), -1, AA)
# ngôi nhà
shape([(120, 620), (420, 620), (420, 900), (120, 900)], (170, 215, 245))
shape([(90, 630), (270, 450), (450, 630)], (70, 80, 210))
shape([(225, 760), (310, 760), (310, 900), (225, 900)], (60, 110, 160))
shape([(145, 670), (205, 670), (205, 730), (145, 730)], (240, 220, 150))
shape([(335, 670), (395, 670), (395, 730), (335, 730)], (240, 220, 150))
# cái cây
shape([(700, 640), (750, 640), (760, 900), (690, 900)], (50, 90, 140))
cv2.circle(img, (725, 540), 135, (80, 180, 90), -1, AA)
cv2.circle(img, (725, 540), 135, K, T, AA)
# người que
cv2.circle(img, (560, 690), 38, K, T, AA)
cv2.line(img, (560, 728), (560, 830), K, T, AA)
cv2.line(img, (560, 760), (510, 800), K, T, AA)
cv2.line(img, (560, 760), (610, 790), K, T, AA)
cv2.line(img, (560, 830), (525, 900), K, T, AA)
cv2.line(img, (560, 830), (595, 900), K, T, AA)
# mặt đất
cv2.line(img, (40, 905), (960, 905), K, T, AA)

cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "sample.png", img)
