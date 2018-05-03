from skimage import io

import os,glob,sys
import numpy as np

dirname = sys.argv[1]
filename = sys.argv[2]
filetype = '*.jpg'

filenames = glob.glob(os.path.join(dirname,filetype))

x=[]
for i in filenames:
    img = io.imread(i)
    x.append(img.flatten())
x = np.array(x)

x = x.transpose()

x = x.astype(np.float64)

x_mean = np.tile(x.mean(axis = 1).reshape(len(x),1),(1,415))

U, s, V = np.linalg.svd(x - x_mean, full_matrices=False)

y = io.imread(os.path.join(dirname,filename))
y = y.flatten()
y = y - x.mean(axis = 1)

n_component = 4

w = np.dot(y, U[:,:n_component])

M = np.zeros(len(y))
for i in range(n_component):
    M += w[i] * U[:,:n_component].transpose()[i]
M += x.mean(axis = 1)

M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)

M = M.reshape(600,600,3)

io.imsave("reconstruction.png",M)

