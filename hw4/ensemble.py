import os, sys
import numpy as np
from random import shuffle
from math import log, floor
import pandas as pd

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=None)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

    return (X_train, Y_train, X_test)

def sample_with_replacement(n, X_all, Y_all):
    X_sample = np.zeros(X_all.shape)
    Y_sample = np.zeros(Y_all.shape)
    idx = np.random.choice(X_all.shape[0], n)
    for i in range(X_all.shape[0]):
        X_sample[i] = X_all[idx[i]]
        Y_sample[i] = Y_all[idx[i]]
    return X_sample, Y_sample

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return

def train(X_all, Y_all, save_dir):
    X_all, Y_all = sample_with_replacement(X_all.shape[0], X_all, Y_all)
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    w = np.zeros(len(X_train[0]))
    b = np.zeros((1,))
    l_rate = 0.001
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 10000
    save_param_iter = 50

    total_loss = 0.0
    for epoch in range(1, epoch_num):
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        X_train, Y_train = _shuffle(X_train, Y_train)

        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return

X_all, Y_all, X_test_all = load_data('train_X', 'train_Y', 'test_X')

print(X_all[:,:1])
X_add = np.log(1+X_all[:,:1])
X_test_add = np.log(1+X_test_all[:,:1])

print(X_all[:,78:81])
X_add = np.concatenate((np.log(1+X_all[:,78:81]),X_add), axis=1)
X_test_add = np.concatenate((np.log(1+X_test_all[:,78:81]),X_test_add), axis=1)

X = np.concatenate((X_all,X_add), axis=1)
X_test = np.concatenate((X_test_all,X_test_add), axis=1)

X, X_test = normalize(X, X_test)
print(X.shape)
print(X_test.shape)

n_bag = 10

for i in range(n_bag):
    dirname = 'model' + str(i)
    train(X,Y_all,dirname)

def get_most(arr):
    count_dict = {}
    for i in arr:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1
    n = 0
    key = 0
    for i in count_dict.keys():
        if(count_dict[i] > n):
            n = count_dict[i]
            key = i
    return key

def predict(X_test, save_dir):
    test_data_size = len(X_test)

    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    y_ = np.reshape(y_, (len(y_),1))
    return y_

def write_ans(y_, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    output_path = os.path.join(output_dir, 'ans.csv')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

y_ = []
for i in range(10):
    save_dir = 'model' + str(i)
    if i == 0:
        y_ = predict(X_test, save_dir)
    else:
        y_ = np.concatenate((y_, predict(X_test, save_dir)),axis = 1)
y_.shape

y_pred = []
for i in range(y_.shape[0]):
    y_pred.append(get_most(y_[i]))
y_pred = np.array(y_pred)


write_ans(y_pred, './')

