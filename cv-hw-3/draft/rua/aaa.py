import os
import natsort
import cv2
import numpy as np
import PIL.Image as Image
import sys
imglist = os.listdir('data/face')
imglist = natsort.natsorted(imglist)

def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
    return mat

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu

def project(W, X, mu=None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)
def pca(X, y, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
   
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]

def lda(X, y, num_components=0):
    y = np.asarray(y)
    [n, d] = X.shape
    c = np.unique(y)
    if (num_components <= 0) or (num_components > (len(c) - 1)):
        num_components = (len(c) - 1)
    meanTotal = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in c:
        Xi = X[np.where(y == i)[0], :]
        meanClass = Xi.mean(axis=0)
        Sw = Sw + np.dot((Xi - meanClass).T, (Xi - meanClass))
        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(eigenvectors[0:, 0:num_components].real, dtype=np.float32, copy=True)
    return [eigenvalues, eigenvectors]

def fisherfaces(X, y, num_components=0):
    y = np.asarray(y)
    # print X.shape
    [n, d] = X.shape
    c = len(np.unique(y))
    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n - c))
    [eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)
    eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)
    return [eigenvalues_lda, eigenvectors, mu_pca]
   
def read_images(path):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c + 1
    return [X, y]

def FisherfacesModel(X,y,train=True):
    num_components = 0
    [D, W, mu] = fisherfaces(asRowMatrix(X), y, num_components)
    return W,mu

def dist_metric(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum(np.power((p - q), 2)))

def predict(W,mu,projections,y,X):
    minDist = np.finfo('float').max
    # print minDist
    minClass = -1
    Q = project(W, X.reshape(1, -1), mu)
    for i in xrange(len(projections)):
        dist = dist_metric(projections[i], Q)
        if dist < minDist:
            minDist = dist
            minClass = y[i]
    return minClass

if __name__ == '__main__':
    ##########################
    ###train
    [X, y] = read_images('face/train')
    # print y
    W,mu= FisherfacesModel(X, y)
    projections = []
    for xi in X:
        projections.append(project(W, xi.reshape(1, -1), mu))
    # projections = np.array(projections)
    # print projections.shape
    ########################
    ##test
    [Xtest, ytest] = read_images('face/test')
    accurancy = 0
    for i in range(len(Xtest)):
        pred = predict(W,mu,projections,y,Xtest[i])
        # print "expected =", ytest[i], "/", "predicted =", pred
        if pred == ytest[i]:
            accurancy += 1
    print accurancy
    print "accurancy:",accurancy*1.0/len(Xtest)


