import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

plt.figure(figsize=(10,5))

images = glob.glob('*.bmp')

print('Reading image...')
print('----------------------------------------------------------------')
 
for fname in images:
    print('Processing '+fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    plt.subplot(1,2,1), plt.title('original image')
    plt.imshow(img)
    print('Done\n')

    # Find the chess board corners
    print('Finding Chessboard...')
    print('-----------------------------------------')
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(1)

print('Calibrating...')
print('-----------------------------------------')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('Camera\'s Intrinsic Matrix:\n', mtx)
print('Distortion:\n', dist)
print('Rotate Matrix:\n', rvecs)
print('Translate Matrix:\n', tvecs)

# serialize the matrices
cv_file = cv2.FileStorage("./matrices.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("mtx", mtx)
cv_file.write("dist", dist)
cv_file.write("rvecs", rvecs[0])
cv_file.write("tvecs", tvecs[0])
cv_file.release()
print('Done')

cv2.destroyAllWindows()

# show the images with plt
plt.subplot(1,2,2), plt.title('image with pattern')
plt.imshow(img)
plt.show()
