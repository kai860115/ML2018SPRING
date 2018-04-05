import os
import sys
import numpy as np
from random import shuffle
from math import log, floor
import pandas as pd


def readTrain(XTrainDataPath, YTrainDataPath):
    X_train = pd.read_csv(XTrainDataPath, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(YTrainDataPath, sep=',', header=None)
    Y_train = np.array(Y_train.values)
    return X_train, Y_train


def readTest(xTestDataPath):
    X_test = pd.read_csv(xTestDataPath, sep=',', header=0)
    X_test = np.array(X_test.values)
    return X_test


def normalize(XTrain, XTest):
    XTrainAndTest = np.concatenate((XTrain, XTest))
    mu = np.mean(XTrainAndTest, axis=0)
    sigma = np.std(XTrainAndTest, axis=0)
    mu = np.tile(mu, (len(XTrainAndTest), 1))
    sigma = np.tile(sigma, (len(XTrainAndTest), 1))
    XStandardization = (XTrainAndTest - mu) / sigma

    XTrain = XStandardization[0:len(XTrain)]
    XTest = XStandardization[len(XTrain):]

    return XTrain, XTest


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


def getPred(X, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1-mu2), sigma_inverse)
    xt = X.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * \
        np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    z = np.dot(w, xt) + b
    y = sigmoid(z)
    y_ = np.around(y)
    return y_


def getScore(X, Y, mu1, mu2, shared_sigma, n1, n2):
    y_ = getPred(X, mu1, mu2, shared_sigma, n1, n2)
    s = 0
    for i in range(len(y_)):
        if(y_[i] == Y[i]):
            s += 1
    score = s/len(y_)
    return score


def generative(X_train, Y_train):
    featureSize = len(X_train[0])
    dataSize = len(X_train)
    n1 = 0
    n2 = 0

    mu1 = np.zeros(featureSize)
    mu2 = np.zeros(featureSize)

    for i in range(dataSize):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            n1 += 1
        else:
            mu2 += X_train[i]
            n2 += 1
    mu1 /= n1
    mu2 /= n2

    sigma1 = np.zeros((featureSize, featureSize))
    sigma2 = np.zeros((featureSize, featureSize))

    for i in range(dataSize):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]),
                             [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]),
                             [(X_train[i] - mu2)])

    sigma1 /= n1
    sigma2 /= n2

    shared_sigma = (float(n1) / (n1 + n2)) * sigma1 + \
        (float(n2) / (n1 + n2)) * sigma2

    np.save('n1_1.npy', n1)
    np.save('n2_1.npy', n2)
    np.save('mu1_1.npy', mu1)
    np.save('mu2_1.npy', mu2)
    np.save('shared_sigma_1.npy', shared_sigma)

    print('Accuracy: %f' %
          (getScore(X_train, Y_train, mu1, mu2, shared_sigma, n1, n2)))

    return


def writeAns(X, filename):
    mu1 = np.load('mu1_1.npy')
    mu2 = np.load('mu2_1.npy')
    shared_sigma = np.load('shared_sigma_1.npy')
    n1 = np.load('n1_1.npy')
    n2 = np.load('n2_1.npy')

    y_ = getPred(X, mu1, mu2, shared_sigma, n1, n2)
    file = open(filename, 'w')
    file.write('id,label\n')
    for i, v in enumerate(y_):
        file.write('%d,%d\n' % (i+1, v))
    return


XTrainDataPath = sys.argv[3]
YTrainDataPath = sys.argv[4]
xTestDataPath = sys.argv[5]
ansPath = sys.argv[6]

XTrainData, YTrain = readTrain(XTrainDataPath, YTrainDataPath)
XTestData = readTest(xTestDataPath)

XTrain, XTest = normalize(XTrainData, XTestData)

generative(XTrain, YTrain)
writeAns(XTest, ansPath)
