import os, sys
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle

test_path = sys.argv[1]
output_path = sys.argv[2]

vocab_size = 256

tokenizer = pickle.load(open('tokenizer_256.pickle', 'rb'))

def read_test(data_path):
    x = []
    with open(data_path,'r') as f:
        first_line = f.readline()
        for line in f:
            begin = 0
            for i in range(len(line)):
                if(line[i]==','):
                    begin = i + 1
                    break
            x.append(line[begin:].strip())
    x = tokenizer.texts_to_matrix(x)
    return x

from keras.models import load_model
model = load_model("./bow_model.h5")

x_test = read_test(test_path)
y_test = model.predict(x_test)


y_test = np.squeeze(y_test)
y_test = np.greater(y_test, 0.5).astype(np.int32)

try:
    if not os.path.dirname(output_path) == '':
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
except:
    print("makedir fail")

with open(output_path,'w') as f:
    f.write('id,label\n')
    for i in range(len(y_test)):
        f.write(str(i) + ',' + str(y_test[i]) + '\n')
