import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

plt.figure(figsize=(10,5))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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

print('Reading image...')
print('----------------------------------------------------------------')
img = cv2.imread('cropped.png')
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
cv2.imshow('img', img)
cv2.waitKey(1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1), plt.title('cropped')
plt.imshow(img)
print('Done')

h,  w = img.shape[:2]





