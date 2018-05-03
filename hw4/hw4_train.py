
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


x = np.load('image.npy')


# In[3]:


x = x.astype('float32') / 255.


# In[5]:


x = np.reshape(x,(len(x),-1))


# In[7]:


valid_data_size = int(x.shape[0] * 0.1)
x_valid = x[:valid_data_size]
x_train = x[valid_data_size:]


# In[10]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


# In[11]:


input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

encoder = Model(input=input_img, output=encoded)

adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()


# In[12]:


autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_valid, x_valid))


# In[13]:


autoencoder.save('deepAutoencoder.h5')
encoder.save('deepEncoder.h5')


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


encoded_imgs = encoder.predict(x)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


# In[16]:


f = pd.read_csv('test_case.csv')
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])


# In[17]:


o = open('prediction.csv', 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1  # two images in same cluster
    else: 
        pred = 0  # two images not in same cluster
    o.write("{},{}\n".format(idx, pred))
o.close()

