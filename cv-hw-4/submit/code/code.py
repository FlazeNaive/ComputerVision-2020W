import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

A = 9
B = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((A*B,3), np.float32)
objp[:,:2] = np.mgrid[0:A,0:B].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

plt.figure(figsize=(10,8))

images = glob.glob('*.bmp')

print('Reading image...')
print('----------------------------------------------------------------')
 
for fname in images:
    print('Processing '+fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    plt.subplot(2,2,1), plt.title('original image')
    plt.imshow(img)
    print('Done\n')

    # Find the chess board corners
    print('\nFinding Chessboard...')
    print('-----------------------------------------')
    ret, corners = cv2.findChessboardCorners(gray, (A,B),None)

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
    
    # serialize the matrices 
    np.savez('coefs.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    cv_file = cv2.FileStorage("./save/matrices.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("mtx", mtx)
    cv_file.write("dist", dist)
    cv_file.write("rvecs", rvecs[0])
    cv_file.write("tvecs", tvecs[0])
    cv_file.release()
    print('Done')
    
    cv2.destroyAllWindows()
    
    plt.subplot(2,2,2), plt.title('image with pattern')
    plt.imshow(img)
    
    print('\nUndistorting Image using remapping...')
    print('-----------------------------------------')
    # undistort
    h,  w = img.shape[:2]
    # generate the new matrix of camera pose after undistortion
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # here we can get how the pixels are mapped to the new image
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    print('mapx:\n', mapx)
    print('mapy:\n', mapy)
    # and use it to regenerate the undistorted photo
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
    
    # show the fo images with plt
    plt.subplot(2,2,3), plt.title('undistorted')
    plt.imshow(dst)
    plt.subplot(2,2,4), plt.title('cropped')
    plt.imshow(cropped)
    plt.show()
    
    print("\nGenerating Birdview...")
    print('----------------------------------------------------------------')
    
    # use the undistorted image as the source, find the corners for generate the birdview image
    img = cropped
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (A,B), None)
    print('ret = ', ret)
    if ret == True:
        objpoints.append(objp)
    
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
    
        img_ = img.copy()
        # Draw and display the corners
        img_ = cv2.circle(img_, (corners[0][0][0], corners[0][0][1]), 63, (0, 0, 255), 5)
        img_ = cv2.circle(img_, (corners[A-1][0][0], corners[A-1][0][1]), 63, (0, 255, 0), 5)
        img_ = cv2.circle(img_, (corners[A*(B-1)][0][0], corners[A*(B-1)][0][1]), 63, (255, 255, 0), 5)
        img_ = cv2.circle(img_, (corners[A*B-1][0][0], corners[A*B-1][0][1]), 63, (255, 0, 255), 5)
        img_ = cv2.drawChessboardCorners(img_, (A,B), corners2,ret)
        cv2.imshow('img',img_)
        cv2.waitKey(1)
    
    corners = np.squeeze(corners)
    
    # If map the image to a B*8 chessboard, it will be really small, so make a scale here
    Scale = 50
    objpts = np.float32([[0, 0], [A-1, 0], [0, B-1], [A-1, B-1]])
    objpts = objpts * Scale
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
    
