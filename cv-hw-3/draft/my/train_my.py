import cv2 as cv
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NUM_EIGEN_FACES = 10
TRAIN_PATH = 'BioFaceDatabase\BioID-FaceDatabase-V1.2'
TRAIN_PATH = input('input the path of images:\n')

TrainFiles = os.listdir(TRAIN_PATH)
cnt = len(TrainFiles)

paths = glob.glob(os.path.join(TRAIN_PATH, '*.pgm'))
paths.sort()
# get image sources

T = []
# T is a cnt*62500 matrix which records the original face data

for path in paths:
    img = cv.imread(path, -1)
    FACE = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    Scan = FACE.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)
    if len(Scan) == 1 :
        for (x,y,w,h) in Scan:
            cropped = img[y:y+h, x:x+w]
            # cropped is where face is
            cv.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)
            cropped = cv.equalizeHist(cropped)
            cropped = cv.resize(cropped, (250, 250))
            # change it to 250 * 250 (then will be 62500)
            cv.imshow("ori",img)
            cv.imshow("figure",cropped)
            cv.waitKey(1)
            print(path)
            # change it to 1-D and append it to T
            cropped = cropped.reshape(cropped.size, 1)
            T.append(cropped)

# Down here is my own implementation of PCA, since the cv.PCAcompute() function is quite faster, I'll use it instead of mine. 

T = np.array(T).astype("float32")
T = T.squeeze()
[n, d] = T.shape
Mean = T.mean(axis=0)
T -= Mean
C = T.dot(T.T)
vs, Vs = np.linalg.eig(C)
Vs = T.T.dot(Vs)
'''
for i in range(n):
    V = Vs[:, i]
    mn = min(V)
    mx = max(V)
    Vs[:, i] = [ (V[j] - mn) / (mx - mn) for j in range(d) ]
    # Vs[:, i] = Vs[:, i] / np.linalg.norm(Vs[:, i])
'''

idx = np.argsort(vs)
vs = vs[idx]
Vs = Vs[:,idx]

ENERGY = float(input("input the energy(float in 0~1):\n"))
Sum = 0
for i in range(vs.shape[0]):
    Sum += vs[i]/sum(vs)
    if Sum >= ENERGY:
        break
    NUM_EIGEN_FACES = max([i, NUM_EIGEN_FACES])
vs = vs[0:NUM_EIGEN_FACES].copy()
Vs = Vs[:, 0:NUM_EIGEN_FACES].copy()
print(Vs.shape)

eigenFaces = []

Vs = Vs.T
for V in Vs:
    eigenFace = V.reshape((250, 250))
    # here we will show the eigenface
    cv.imshow('figure', eigenFace)
    cv.waitKey(1)
    eigenFaces.append(eigenFace)

eigenFaces = np.array(eigenFaces)
eigenFace_mean = eigenFaces.mean(axis=0)

Mean = Mean.reshape((250, 250))
plt.figure()
plt.title('Mean')
plt.subplot(1, 2, 1)
plt.imshow(Mean, cmap = plt.get_cmap('gray'))
plt.subplot(1, 2, 2)
plt.imshow(eigenFace_mean, cmap = plt.get_cmap('gray'))
plt.show()

# save the mat value for recognization
np.save('save_database\\save_Mean_my', Mean)
np.save('save_database\\save_A_my', T)
np.save('save_database\\eigenface_my', eigenFaces)
