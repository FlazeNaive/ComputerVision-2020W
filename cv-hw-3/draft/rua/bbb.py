#coding:utf-8
from numpy import *
from numpy import linalg as la
import cv2
import os

def loadImageSet(add):
    FaceMat = mat(zeros((15,98*116)))
    j =0
    for i in os.listdir(add):
        if i.split('.')[1] == 'normal':
            try:
                img = cv2.imread(add+i,0)
            except:
                print 'load %s failed'%i
            FaceMat[j,:] = mat(img).flatten()
            j += 1
    return FaceMat

def ReconginitionVector(selecthr = 0.8):
    # step1: load the face image data ,get the matrix consists of all image
    FaceMat = loadImageSet('D:\python/face recongnition\YALE\YALE\unpadded/').T
    # step2: average the FaceMat
    avgImg = mean(FaceMat,1)
    # step3: calculate the difference of avgimg and all image data(FaceMat)
    diffTrain = FaceMat-avgImg
    #step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)
    eigvals,eigVects = linalg.eig(mat(diffTrain.T*diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in xrange(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex] # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg,covVects,diffTrain

def judgeFace(judgeImg,FaceVector,avgImg,diffTrain):
    diff = judgeImg.T - avgImg
    weiVec = FaceVector.T* diff
    res = 0
    resVal = inf
    for i in range(15):
        TrainVec = FaceVector.T*diffTrain[:,i]
        if  (array(weiVec-TrainVec)**2).sum() < resVal:
            res =  i
            resVal = (array(weiVec-TrainVec)**2).sum()
    return res+1

if __name__ == '__main__':

    avgImg,FaceVector,diffTrain = ReconginitionVector(selecthr = 0.9)
    nameList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
    characteristic = ['centerlight','glasses','happy','leftlight','noglasses','rightlight','sad','sleepy','surprised','wink']

    for c in characteristic:

        count = 0
        for i in range(len(nameList)):

            # 这里的loadname就是我们要识别的未知人脸图，我们通过15张未知人脸找出的对应训练人脸进行对比来求出正确率
            loadname = 'D:\python/face recongnition\YALE\YALE\unpadded\subject'+nameList[i]+'.'+c+'.pgm'
            judgeImg = cv2.imread(loadname,0)
            if judgeFace(mat(judgeImg).flatten(),FaceVector,avgImg,diffTrain) == int(nameList[i]):
                count += 1
        print 'accuracy of %s is %f'%(c, float(count)/len(nameList))  # 求出正确率
