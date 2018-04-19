import sys
import os
import numpy as np
import pandas as pd
import math
from keras.utils import np_utils


def readTrain(fileName):
    train_data = pd.read_csv(fileName)
    train_data = train_data.values
    y_train = train_data[:, 0]
    y_train = np_utils.to_categorical(y_train, 7)
    x_train = []
    for i in range(len(train_data)):
        x_train.append([int(j) for j in train_data[i][1].split()])
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(x_train), 48, 48, 1).astype('float32')
    return x_train, y_train


def normalize(x):
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    x = (x-mean)/std
    return x


x_train, y_train = readTrain(sys.argv[1])
print(x_train.shape)

x_train = normalize(x_train)
print(x_train.shape)


def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(math.floor(all_data_size * percentage))

    X_all, Y_all = shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:-valid_data_size], Y_all[0:-valid_data_size]
    X_valid, Y_valid = X_all[-valid_data_size:], Y_all[-valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


x_train, y_train, x_valid, y_valid = split_valid_set(x_train, y_train, 0.1)

print(x_train.shape)
print(x_valid.shape)


if not os.path.isdir('./model'):
    os.mkdir('./model')
np.save('./model/x_train.npy', x_train)
np.save('./model/y_train.npy', y_train)
np.save('./model/x_valid.npy', x_valid)
np.save('./model/y_valid.npy', y_valid)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def buildModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5),
                     input_shape=(48, 48, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    return model


filepath = "./model/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=False)
callbacks_list = [checkpoint]

model = buildModel()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                              steps_per_epoch=int(
                                  np.ceil(5*x_train.shape[0] / 128.0)),
                              validation_data=(x_valid, y_valid),
                              epochs=300, verbose=2,
                              callbacks=callbacks_list)
