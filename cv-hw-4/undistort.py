import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

print('Reading Matrices...')
print('-----------------------------------------')
cv_file = cv2.FileStorage("./matrices.xml", cv2.FILE_STORAGE_READ)

rvecs = []
tvecs = []
mtx = cv_file.getNode("mtx").mat()
dist = cv_file.getNode("dist").mat()
rvecs.append(cv_file.getNode("rvecs").mat())
tvecs.append(cv_file.getNode("tvecs").mat())

print('Camera\'s Intrinsic Matrix:\n', mtx)
print('Distortion:\n', dist)
print('Rotate Matrix:\n', rvecs)
print('Translate Matrix:\n', tvecs)

images = glob.glob('*.bmp')

print('Reading image...')
print('----------------------------------------------------------------')
 
for fname in images:
    print('Processing '+fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print('Undistorting Image using remapping...')
print('-----------------------------------------')
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
print('mapx:\n', mapx)
print('mapy:\n', mapy)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print('Done')
cv2.imwrite('calibresult.png',dst)
cv2.imshow('calibresult.png',dst)
cv2.waitKey(1)

# crop the image
x,y,w,h = roi
cropped = dst[y:y+h, x:x+w]
cv2.imwrite('cropped.png',cropped)
cv2.imshow('cropped.png',cropped)
cv2.waitKey(1)

cv2.destroyAllWindows()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title('undistorted')
plt.imshow(dst)
plt.subplot(1,2,2), plt.title('cropped')
plt.imshow(cropped)
plt.show()
