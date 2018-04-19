import numpy as np
import pandas as pd
import math
import sys
from keras.utils import np_utils

def readTrain(fileName):
    train_data = pd.read_csv(fileName)
    train_data = train_data.values
    y_train = train_data[:,0]
    y_train = np_utils.to_categorical(y_train,7)
    x_train = []
    for i in range(len(train_data)):
        x_train.append([int(j) for j in train_data[i][1].split()])
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(x_train),48,48,1).astype('float32')
    return x_train,y_train

def readTest(fileName):
    test_data = pd.read_csv(fileName)
    test_data = test_data.values
    x_test = []
    for i in range(len(test_data)):
        x_test.append([int(j) for j in test_data[i][1].split()])
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test),48,48,1).astype('float32')
    return x_test

def normalize(x):
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    x = (x-mean)/std
    return x

x_test = readTest(sys.argv[1])
print(x_test.shape)

x_test = normalize(x_test)
print(x_test.shape)

from keras.models import load_model

model = load_model('./best_model.h5')

y_pred = model.predict_classes(x_test)

file = open(sys.argv[2],'w')
file.write('id,label\n')
for i in range(len(y_pred)):
    file.write(str(i)+','+str(y_pred[i])+'\n')
file.close()

