import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

A = 12
B = 12

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((A*B,3), np.float32)
objp[:,:2] = np.mgrid[0:A,0:B].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

plt.figure(figsize=(10,8))

images = glob.glob('*.jpg')

print('Reading image...')
print('----------------------------------------------------------------')
 
for fname in images:
    print('Processing '+fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    b,g,r = cv2.split(img)           # get b, g, r
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    plt.subplot(2,2,1), plt.title('original image')
    plt.imshow(img)
    print('Done\n')

    # Find the chess board corners
    print('\nFinding Chessboard...')
    print('-----------------------------------------')
    ret, corners = cv2.findChessboardCorners(gray, (A,B),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img_ = img.copy()
        # Draw and display the corners
        # Use img_ here instead of img is for the later use of undistorted image
        img_ = cv2.drawChessboardCorners(img_, (A,B), corners2,ret)
        cv2.imshow('img',img_)
        cv2.waitKey(1)

    print('\nCalibrating...')
    print('-----------------------------------------')
    # Using the cv2.calibrateCamera function, it can generate the needed matrices for the camera pose and how the image is distorted
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print('Camera\'s Intrinsic Matrix:\n', mtx)
    print('Distortion:\n', dist)
    print('Rotate Matrix:\n', rvecs)
    print('Translate Matrix:\n', tvecs)
    
    corners = np.squeeze(corners)
    
    # If map the image to a B*8 chessboard, it will be really small, so make a scale here
    Scale = 10
    objpts = np.float32([[0, 0], [A-1, 0], [0, B-1], [A-1, B-1]])
    objpts = objpts * Scale
    objpts += 300
    print(objpts)
    imgpts = np.float32([corners[0],corners[A-1],corners[A*(B-1)],corners[A*B-1]])
    print(imgpts)
    
    cv2.destroyAllWindows()
    
    # find how it transformed
    H = cv2.getPerspectiveTransform(imgpts, objpts)
    
    print("\nPress 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit")
    
    Z = H[2, 2]
    while True:
        H[2, 2] = Z
        Perspective_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv2.imshow("Birdseye View", Perspective_img)
        KEY = cv2.waitKey() & 0xFF
    
        if KEY == ord('u'):
            Z += 0.05
        if KEY == ord('d'):
            Z -= 0.05
        if KEY == 27:
            cv2.destroyAllWindows()
            break
    
