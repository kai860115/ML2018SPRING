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


def trainShuffle(XTrainData, YTrainData):
    trainData = np.concatenate((XTrainData, YTrainData), 1)
    np.random.shuffle(trainData)
    XTrainData = trainData[:, :len(XTrainData[0])]
    YTrainData = trainData[:, len(XTrainData[0]):]
    return XTrainData, YTrainData


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


def getPred(X, w, b):
    z = (np.dot(X, np.transpose(w)) + b)
    y = sigmoid(z)
    YPred = np.around(y)
    return YPred


def getScore(X, w, b):
    YPred = getPred(XTrain, w, b)
    s = 0.0
    for i in range(len(YPred)):
        if(YPred[i] == YTrain[i]):
            s += 1
    return s/len(YPred)


def logRegression(XTrain, YTrain):
    w = np.zeros(len(XTrain[0]))
    b = np.zeros(1)
    lr = 0.001
    batchSize = 32
    epoch = 1000
    dataSize = len(XTrain)
    batchStep = int(floor(dataSize/batchSize))

    loss = 0.0
    for idx in range(epoch):
        if(idx > 0):
            print('%.0f:epoch loss = %f    accuracy = %f' %
                  (idx, (loss/dataSize), getScore(XTrain, w, b)))
            loss = 0.0

        XTrain, YTrain = trainShuffle(XTrain, YTrain)

        for j in range(batchStep):
            X = XTrain[j*batchSize:(j+1)*batchSize]
            Y = YTrain[j*batchSize:(j+1)*batchSize]
            z = np.dot(X, w.T) + b
            y = sigmoid(z)

            cross_entropy = -1 * \
                (np.dot(np.squeeze(Y), np.log(y)) +
                 np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) -
                                       y).reshape((batchSize, 1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            w = w - lr * w_grad
            b = b - lr * b_grad

    np.save('w1.npy', w)
    np.save('b1.npy', b)

    return


def writeAns(XTest, fileName):
    w = np.load('w1.npy')
    b = np.load('b1.npy')

    YPred = getPred(XTest, w, b)

    file = open(fileName, 'w')
    file.write("id,label\n")
    for i, v in enumerate(YPred):
        file.write('%d,%d\n' % (i+1, v))
    file.close()


XTrainDataPath = sys.argv[3]
YTrainDataPath = sys.argv[4]
xTestDataPath = sys.argv[5]
ansPath = sys.argv[6]

XTrainData, YTrain = readTrain(XTrainDataPath, YTrainDataPath)
XTestData = readTest(xTestDataPath)

X_add = np.log(1+XTrainData[:, :1])
X_test_add = np.log(1+XTestData[:, :1])

X_add = np.concatenate((np.log(1+XTrainData[:, 78:81]), X_add), axis=1)
X_test_add = np.concatenate(
    (np.log(1+XTestData[:, 78:81]), X_test_add), axis=1)

XTrain = np.concatenate((XTrainData, X_add), axis=1)
XTest = np.concatenate((XTestData, X_test_add), axis=1)

XTrain, XTest = normalize(XTrain, XTest)

print(XTrain.shape)
print(XTest.shape)

logRegression(XTrain, YTrain)
writeAns(XTest, ansPath)
