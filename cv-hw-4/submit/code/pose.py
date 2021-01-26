import cv2.cv2 as cv2
import numpy as np
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def save_pointcloud(fname, board, rvecs, tvecs, rows, cols):
    filename = fname.split('.')[0]+'_pointcloud.ply'
    with open(filename, mode='w') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {}\n'.format(rows*cols+2+3))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        file.write('element edge {}\n'.format(9+3))
        file.write('property int vertex1\n')
        file.write('property int vertex2\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')

        file.write('end_header\n')

        rmat, _ = cv2.Rodrigues(rvecs)
        rmat = rmat.transpose()
        tvecs = -np.matmul(rmat, tvecs)
        print(rmat)
        print(tvecs)
        # print(rmat)
        # print(rmat[:, 0])

        vertex_list = []
        vertex_list.append(tvecs.reshape(1, 3)[0])
        vertex_list.append(np.float32([0,0,1]))
        vertex_list.append(tvecs.reshape(1, 3)[0]+rmat[:, 0])
        vertex_list.append(tvecs.reshape(1, 3)[0]+rmat[:, 1])
        vertex_list.append(tvecs.reshape(1, 3)[0]+rmat[:, 2])


        for i in range(board.shape[0]):
            v = board[i]
            file.write('{} {} {}\n'.format(v[0], v[1], v[2]))
        for v in vertex_list:
            file.write('{} {} {}\n'.format(v[0], v[1], v[2]))

        file.write('{} {} {} {} {}\n'.format(0, cols-1, 255, 0, 0))
        file.write('{} {} {} {} {}\n'.format(0, (rows-1)*cols, 0, 255, 0))
        file.write('{} {} {} {} {}\n'.format(rows*cols-1, cols-1, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format(rows*cols-1, (rows-1)*cols, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format(0, rows*cols, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format(cols-1, rows*cols, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format((rows-1)*cols, rows*cols, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format(rows*cols-1, rows*cols, 255, 255, 255))
        file.write('{} {} {} {} {}\n'.format(0, rows*cols+1, 0, 0, 255))
        
        file.write('{} {} {} {} {}\n'.format(rows*cols, rows*cols+2, 255, 0, 0)) # rotate x
        file.write('{} {} {} {} {}\n'.format(rows*cols, rows*cols+3, 0, 255, 0))
        file.write('{} {} {} {} {}\n'.format(rows*cols, rows*cols+4, 0, 0, 255))

rows = 6
cols = 9

# Load previously saved data
with np.load('coefs.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
# print(objp)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob.glob('a*.bmp'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cols,rows),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # print(corners)
        # print(corners2)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        print(rvecs)
        print(tvecs)
        # print(inliers)
        save_pointcloud(fname, objp, rvecs, tvecs, rows, cols)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)

        # img_resized = cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC) 
        # cv2.imshow('img',img_resized)
        # # cv2.imshow('img',img)
        # k = cv2.waitKey(0) & 0xff
        # if k == 's':
        #     cv2.imwrite(fname+'_pose.bmp', img)
        # if k == 'q':
        #     break

cv2.destroyAllWindows()