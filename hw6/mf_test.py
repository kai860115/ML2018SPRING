import numpy as np
import os, sys
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping

def nn_model(n_users, n_items, latent_dim=100):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dropout(0.25)(merge_vec)
    hidden = Dense(150)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU(0.05)(hidden)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(150)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU(0.05)(hidden)
    hidden = Dropout(0.35)(hidden)
    hidden = Dense(50)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = LeakyReLU(0.05)(hidden)
    hidden = Dropout(0.45)(hidden)
    output = Dense(1)(hidden)
    model = Model([user_input, item_input], output)
    model.compile(loss='mse', optimizer='adam')
    return model

def mf_model(n_users, n_items, latent_dim=100):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adam')
    return model
   
filepath = './mf_model.h5'
testpath = sys.argv[1]
outputpath = sys.argv[2]

TEST = pd.read_csv(testpath)

test_user_id = TEST['UserID'].values
test_movie_id = TEST['MovieID'].values

test_movie_id = test_movie_id.reshape(len(test_movie_id), 1)
test_user_id = test_user_id.reshape(len(test_user_id), 1)

print(test_movie_id.shape)
print(test_user_id.shape)

from keras.models import load_model

model = load_model(filepath)
model.summary()

pred = model.predict([test_user_id, test_movie_id])

pred = pred.reshape(len(pred))

std = 1.116897661146206
mean = 3.5817120860388076

pred = pred * std + mean
pred = np.clip(pred, 1, 5)

try:
    if not os.path.dirname(outputpath) == '':
        if not os.path.isdir(os.path.dirname(outputpath)):
            os.makedirs(os.path.dirname(outputpath))
except:
    print("makedir fail")

with open(outputpath,'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(len(pred)):
        f.write(str(i + 1) + ',' + str(pred[i]) + '\n')

from keras import backend as K
K.clear_session()