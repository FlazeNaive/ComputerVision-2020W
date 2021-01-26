import cv2 as cv
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NUM_EIGEN_FACES = 10
# TRAIN_PATH = 'BioFaceDatabase\BioID-FaceDatabase-V1.2'
TRAIN_PATH = input('Please input the path of images:\n')
NUM_EIGEN_FACES = input('Please input the desire NUMBER of EIGENFACES:\n')

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

print("Calculating PCA ", end="...")
Mean, Vs = cv.PCACompute(T, mean=None, maxComponents=NUM_EIGEN_FACES)
print ("DONE")

eigenFaces = []

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
try:
    # save the mat value for recognization
    np.save('save_database\\save_Mean', Mean)
    np.save('save_database\\save_A', T)
    np.save('save_database\\eigenface', eigenFaces)
except Exception as e:
    print (e)

