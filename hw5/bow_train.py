import os, sys
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle

train_path = sys.argv[1]
semi_path = sys.argv[2]

data = {}

def add_data(name, data_path, with_label):
    X, Y = [], []
    with open(data_path,'r') as f:
        for line in f:
            line = line.strip()
            if with_label:
                lines = line.split(' +++$+++ ')
                X.append(lines[1])
                Y.append(int(lines[0]))
            else:
                X.append(line)

    if with_label:
        data[name] = [X,Y]
    else:
        data[name] = [X]

add_data('train_data', train_path, True)
add_data('semi_data', semi_path, False)

vocab_size = 256

tokenizer = Tokenizer(vocab_size)
for key in data:
    tokenizer.fit_on_texts(data[key][0])
pickle.dump(tokenizer, open('tokenizer_256.pickle', 'wb'))

tokenizer = pickle.load(open('tokenizer_256.pickle', 'rb'))

for key in data:
    data[key][0] = tokenizer.texts_to_matrix(data[key][0], mode='tfidf')

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def DNN_Model():
    inputs = Input(shape=(vocab_size,))
    
    outputs = Dense(256, activation='relu')(inputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(128, activation='relu')(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(64, activation='relu')(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    
    return model
    
model = DNN_Model()

earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

filepath="./bow_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
callbacks_list = [checkpoint, earlystopping]

def split_valid_set(ratio):
    val_size = (int)(ratio * len(data['train_data'][0]))
    x_train = data['train_data'][0][val_size:]
    y_train = data['train_data'][1][val_size:]
    x_val = data['train_data'][0][:val_size]
    y_val = data['train_data'][1][:val_size]
    
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = split_valid_set(0.3)

print(x_train.shape)
print(np.array(y_train).shape)
print(x_val.shape)
print(np.array(y_val).shape)

history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=20, 
                    batch_size=128, 
                    callbacks=callbacks_list) 

from keras.models import load_model

model = load_model(filepath)

def get_semi_data(threshold):
    x_semi = data['semi_data'][0]
    y_semi = model.predict(x_semi)
    y_semi = np.squeeze(y_semi)
    index = (y_semi > 1 - threshold) + (y_semi < threshold)
    y_semi = np.greater(y_semi, 0.5).astype(np.int32)
    
    return x_semi[index,:], y_semi[index]

for i in range(10):
    x_semi, y_semi = get_semi_data(0.2)
    x_semi = np.concatenate((x_train, x_semi))
    y_semi = np.concatenate((y_train, y_semi))
    
    print ('-- iteration %d  semi_data size: %d' %(i+1,len(x_semi)))
    history = model.fit(x_semi, y_semi, 
                        validation_data=(x_val, y_val), 
                        epochs=3, 
                        batch_size=64, 
                        callbacks=callbacks_list)
    del x_semi
    del y_semi
    
    model = load_model(filepath)
