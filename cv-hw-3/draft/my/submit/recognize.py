import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# load the data from *.npy
Mean = np.load('save_database\\save_Mean.npy')
A = np.load('save_database\\save_A.npy')
eigenfaces = np.load('save_database\\eigenface.npy')

# get the number of eigenfaces
cnt_eig = eigenfaces.shape[0]
# get the projection of images
Vectors = []
for i in range(cnt_eig):
    Vectors.append(np.mat(eigenfaces[i]).flatten())
Vectors = np.array(Vectors)
Vectors.squeeze()
Vectors = np.mat(Vectors).T
PRO = A.dot(Vectors)

TEST_PATH = input('Input the path of photo you want to recognize:\n')
# TEST_PATH = 'BioFaceDatabase\BioID-FaceDatabase-V1.2\\testset\\3.pgm'
testImg= cv.imread(TEST_PATH, -1)

# get the face area (same as in get.py)
FACE = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
Scan = FACE.detectMultiScale(testImg,scaleFactor=1.1,minNeighbors=3)
if len(Scan) == 1 :
    for (x,y,w,h) in Scan:
        cropped = testImg[y:y+h, x:x+w]
        cv.rectangle(testImg, (x,y), (x+w, y+h), (255, 255, 255), 1)
        cropped = cv.equalizeHist(cropped)
        cropped = cv.resize(cropped, (250, 250))
        cv.imshow("figure",cropped)
        cv.waitKey(1)

# to 1-D
Array = cropped.reshape(cropped.size,1)
Array = np.mat(np.array(Array)).squeeze()
# get the difference and project it to the eiganface space
meanVector = Mean.flatten()
meanVector = meanVector.squeeze()
diff = Array - meanVector
diff = diff.squeeze()
pro = diff.dot(Vectors)
print(pro.shape)


# load the train-data to find the most similar photo
TRAIN_PATH = 'BioFaceDatabase\BioID-FaceDatabase-V1.2'
TrainFiles = os.listdir(TRAIN_PATH)
paths = glob.glob(os.path.join(TRAIN_PATH, '*.pgm'))
paths.sort()

# find the minimum distance..
distance = []
for i in range(0, A.shape[0]):
    cur = PRO[i,:]
    temp = np.linalg.norm(pro - cur)
    distance.append(temp)
    print('No.' + str(i) + ' = ' + str(distance[i]))

minDistance = min(distance)
index = distance.index(minDistance) - 1
print('index = ' + str(index))
result = cv.imread(paths[index], -1)
testImg = cv.putText(testImg, "Most Similar: " + paths[index], (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
cv.imshow("ori",testImg)
cv.imshow("recognize result",result)
cv.waitKey(0)

