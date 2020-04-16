import os, sys
import numpy as np
sys.path.append('../../../Projects/nfm/')
from nfm.helper.GaussianStatistics import *
from nfm.helper.configure import Config
from nfm.SOM import SOM
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

index = np.arange(len(x_test))
np.random.shuffle(index)
x_test = x_test[index]
y_test = y_test[index]

_som_ = SOM((10, 10),(28, 28), 25, learning_rate=1e-2, rad = 5, sig = 3)
_som_.load_weights('../../../Projects/nfm/logs/SOM_weights_MNIST_noise_0.0.npy')

complex = lambda x: np.random.uniform(0, 1, size=x) + np.random.uniform(0, 1, size = x)*1j
star = lambda x: np.conjugate(x)    
norm = lambda x: x/(np.abs(x) + 1e-3)
minmax = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))


def som(digit):
    response = _som_.response(digit, _som_.weights)
    # response = minmax(response)
    real = response
    imag = 1. - response
    return real + imag*1j


# DP codes
def get_nbrs(i, j, Z):
    nbrs = []
    idx = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for ix, iy in idx:
        try:
            nbrs.append(Z[i+ix, j+iy])
        except:
            pass
    return nbrs
    
def couple(W, Z):
    coupling = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            nbrs = get_nbrs(i,j,Z)
            coupling[i, j] = W[i, j]*np.prod(nbrs)
    return coupling


def nbr(Z):
    nbr_matrix = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            nbrs = get_nbrs(i,j,Z)
            nbr_matrix[i, j] = np.prod(nbrs)
    return nbr_matrix


omega = np.pi/20.
mu = 5.0
T = 50
deltaT = 0.1
ita = 1e-2


digits = np.arange(10)
phase_information = []
magnitude_information = []
for digit in digits:
    Zs = []; Ws = []

    # initalizations
    Z = som(x_test[y_test == digit][10])
    W = complex(Z.shape)

    # plt.clf()
    # plt.ion()
    for _ in range(int(T/deltaT)):

        Zdot = (mu - np.abs(Z)**2)*Z + Z*omega*1j + couple(W, Z)
        Z = Z + deltaT*Zdot

        W = norm(W); Z = norm(Z)
        utility = np.zeros(Z.shape)
        utility[np.abs(Z) > 0.5] = 1.0       

        W = W + ita*((utility + 1)*Z*star(nbr(Z)) - W)
        Zs.append(Z); Ws.append(W)

        # plt.imshow(np.abs(Z))
        # plt.pause(0.01)

    Zs = np.array(Zs)
    mZs = np.mean(Zs, axis=0)
    idx = np.where(mZs == np.max(mZs))
    phase_information.append(np.unwrap(np.angle((Zs[:, 5, 5]))))
    magnitude_information.append(np.mean(Zs, axis=0))


for i in digits:
    plt.plot(phase_information[i])
plt.ylabel("Unwrapped phase for different digits")
plt.xlabel('t')
plt.grid()
plt.show()

"""
plt.subplot(1, 2, 1)
plt.plot(np.real(Z1s), np.imag(Z1s), 'b')
plt.plot(np.real(Z1s[0]), np.imag(Z1s[0]), '*g')
#plt.xlim(-10, 10)
#plt.ylim(-10, 10.0)
plt.grid()
plt.xlabel("(1) Neuron $Z_1$ stable state oscillator")

plt.subplot(1, 2, 2)
plt.plot(np.real(Z2s), np.imag(Z2s), 'g')
plt.plot(np.real(Z2s[0]), np.imag(Z2s[0]), '*r')
#plt.xlim(-10, 10)
#plt.ylim(-10, 10.0)
plt.grid()
plt.xlabel("(2) Neuron $Z_2$ stable state oscillator")
plt.show()


#####


plt.subplot(3, 1, 1)
plt.plot(np.real(Z1s))
plt.plot(np.real(Z2s))
plt.xlabel("(1) Real part of $Z_1$ and $Z_2$")

plt.grid()

plt.subplot(3, 1, 2)
plt.plot(np.imag(Z1s))
plt.plot(np.imag(Z2s))
plt.xlabel("(2) Imaginary part of $Z1$ and $Z2$")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(np.angle(Z1s))
plt.plot(np.angle(Z2s))
plt.xlabel("(3) Phase plot for both $Z_1$ and $Z2$")
plt.grid()
plt.show()

"""
