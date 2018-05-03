import numpy as np
import pandas as pd
import sys, os

imgPath = sys.argv[1]
testPath = sys.argv[2]
predictPath = sys.argv[3]

x = np.load(os.path.join(imgPath,'image.npy'))

x = x.astype('float32') / 255.

x = np.reshape(x,(len(x),-1))

valid_data_size = int(x.shape[0] * 0.1)
x_valid = x[:valid_data_size]
x_train = x[valid_data_size:]

from keras.models import load_model

encoder = load_model('bestEncoder.h5')
encoder.summary()

from sklearn.cluster import KMeans

encoded_imgs = encoder.predict(x)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

f = pd.read_csv(os.path.join(testPath,'test_case.csv'))
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

if not os.path.exists(predictPath):
    os.makedirs(predictPath)

o = open(os.path.join(predictPath,'prediction.csv'), 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else: 
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()

