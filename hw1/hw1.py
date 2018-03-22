
# coding: utf-8

# In[1]:


import sys
import csv
import math
import numpy as np
import pandas as pd


# In[2]:


def readTrain(fileName):
    data = []

    for i in range(18):
        data.append([])

    n_row = 0
    text = open(fileName, 'r', encoding = 'big5')
    row = csv.reader(text, delimiter = ',')
    for r in row:
        if(n_row != 0):
            for i in range(3, 27):
                if(r[i] != 'NR'):
                    data[(n_row - 1) % 18].append(float(r[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row = n_row + 1
    text.close()
    x = []
    y = []

    for i in range(12):
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s] )
            y.append(data[9][480*i+j+9])
    x = np.array(x)
    y = np.array(y)
    
    #print(x[:,81:90])
    #x = np.concatenate((x,x[:,81:90]**2), axis=1)

    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    rmidx = []
    zeroitem = 0
    singularityitem = 0
    for i in range(len(y)):
        if(y[i] <= 0):
            rmidx.append(i)
            zeroitem = zeroitem + 1
            for j in range(10):
                if(i + j < len(y)):
                    rmidx.append(i+j)
        if(y[i] - y[i-1] > 50 and i % 471 != 0):
            rmidx.append(i)
            singularityitem = singularityitem + 1
            for j in range(10):
                if((i + j) % 471 == 0):
                    break
                else:
                    rmidx.append(i+j)
        if(y[i] >= 500):
            rmidx.append(i)
            singularityitem = singularityitem + 1
            for j in range(10):
                if(i + j < len(y)):
                    rmidx.append(i+j)
        
    y = np.delete(y, rmidx)
    x = np.delete(x, rmidx,0)
    return x, y


# In[3]:


def RowAverage(X):
    n = 0
    s = 0
    for i in range(len(X)):
        if X[i] > 0:
            s = s + X[i]
            n = n + 1
    if n == 0:
        return 0
    return s/n


# In[4]:


def readTest(fileName):
    df = pd.read_csv(fileName, header = None)
    testdata = df.values
    testdata = testdata[:,2:]
    testdata[testdata == 'NR'] = 0
    testdata = testdata.astype(float)

    testdata = np.vsplit(testdata, len(testdata)/18)
    for i in range(len(testdata)):
        for j in range(9):
            if(testdata[i][9][j] <= 0):
                testdata[i][9][j] = RowAverage(testdata[i][9])
            if(testdata[i][8][j] <= 0):
                testdata[i][8][j] = RowAverage(testdata[i][8])
    Xtest = []
    for i in range(len(testdata)):
        Xtest.append(testdata[i].flatten())
    Xtest = np.array(Xtest)
    #print(Xtest[:,81:90])
    #Xtest = np.concatenate((Xtest,Xtest[:,81:90]**2), axis=1)
    Xtest = np.concatenate((np.ones((Xtest.shape[0],1)),Xtest), axis=1)
    return Xtest


# In[5]:


def LinearRegression(x,y,lr,iter):
    w = np.zeros(len(x[0]))         # initial weight vector
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))
    #costs = []

    for i in range(iter):
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        gra = np.dot(x_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - lr * gra/ada
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
        #costs.append(cost_a)
    return w


# In[6]:


def LRWithRegularization(x,y,lr,iter,r):
    w = np.zeros(len(x[0]))         # initial weight vector
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))
    #costs = []

    for i in range(iter):
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        gra = np.dot(x_t,loss)+2*r*w
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - lr * gra/ada
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
        #costs.append(cost_a)
    return w


# In[7]:


def RMSE(y, ypred):
    loss = ypred - y
    cost = np.sum(loss**2) / len(y)
    return math.sqrt(cost)


# In[8]:


def writeAns(fileName, w, xtest):
    yPred = np.dot(xtest,w)
    file = open(fileName, 'w')
    file.write("id,value\n")
    for i in range(len(yPred)):
        file.write('id_'+str(i)+','+str(yPred[i])+'\n')
    file.close()

xtest = readTest(sys.argv[1])

w = np.load('model.npy')

writeAns(sys.argv[2], w, xtest)
