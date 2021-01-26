import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

FILENAME = "dota2items4.bmp"

img = cv.imread(FILENAME)
img = cv.GaussianBlur(img, (5, 5), 0)

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
edge = cv.Canny(gray, 100, 200)

plt.figure("rgb")
plt.imshow(rgb)
plt.figure("gray")
plt.imshow(gray, cmap = "gray")
plt.figure("thresh")
plt.imshow(thresh, cmap = "gray")
plt.figure("edge")
plt.imshow(edge, cmap = "gray")

img = rgb
edge_ = cv.cvtColor(edge, cv.COLOR_GRAY2RGB)
ells = []
recs = []
recs2 = []
cols = []
contours, nothing = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for index, contour in enumerate(contours):
    if contour.shape[0] > 50:
        ells.append(cv.fitEllipse(contour))
        recs2.append(cv.boundingRect(contour))
        recs.append(cv.minAreaRect(contour))

random.seed(10)
for i in range(len(ells)):
    cols.append([random.randint(0, 256) for j in range(3)])


for index, ell in enumerate(ells):
    cv.ellipse(img, ell, cols[index], 2)
    cv.ellipse(edge_, ell, cols[index], 2)
for index, rec in enumerate(recs):
    box = cv.boxPoints(rec) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv.drawContours(img,[box],0,cols[index],1)
    cv.drawContours(edge_,[box],0,cols[index],1)
for index, rec2 in enumerate(recs2):
    x,y,w,h = rec2
    cv.rectangle(img,(x,y),(x+w,y+h),cols[index],1)
    cv.rectangle(edge_,(x,y),(x+w,y+h),cols[index],1)

plt.figure("res")
plt.imshow(img)
plt.figure("res_edge")
plt.imshow(edge_)
plt.show()

img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
cv.imwrite('res.jpg', img)
edge_ = cv.cvtColor(edge_, cv.COLOR_RGB2BGR)
cv.imwrite('res_edge.jpg', edge_)
cv.imwrite('edge.jpg', edge)
