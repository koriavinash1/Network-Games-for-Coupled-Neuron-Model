import os, sys
import numpy as np
sys.path.append('..')
from nfm.helper.GaussianStatistics import *
from nfm.helper.configure import Config
from nfm.SOM import SOM
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Gstat  = GaussianStatistics()
config = Config()

data = []
for _data_ in x_test[:30]:
    data.append(_data_.flatten('F')/255.0)

data = 1.0*(np.array(data))

print (data.shape)

save_path = '../logs/SOM_weights_MNIST.npy'

# ##
som = SOM((28, 28), 25, learning_rate=1e-2, rad = 5, sig = 3)
som.fit(data)
# som.save_weights(save_path)

som.load_weights(save_path)
som.moveresp(data)


som.view_weights(path = '../logs/SOM_weights_MNIST.png')
