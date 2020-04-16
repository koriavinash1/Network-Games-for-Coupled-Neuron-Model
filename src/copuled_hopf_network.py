
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

complex = lambda x: np.random.uniform(0, 1, size=x) + np.random.uniform(0, 1, size = x)*1j
star = lambda x: np.conjugate(x)    
norm = lambda x: x/np.sum(np.abs(x))
minmax = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))

def som(digit):
    response = SOM(digit)
    response = minmax(response)
    real = response
    imag = 1-response
    return real + imag*1j


# DP codes
def couple(W, Z):
    pass

def nbr(Z):
    pass


digit = 0
omega = np.pi/6.
mu = 1.0
T = 50
deltaT = 0.01
ita = 1e-2
Zs = []; Ws = []


# initalizations
Z = som(x_test[y_test == digit][0])
W = complex()

for _ in range(int(T/deltaT)):
    W = norm(W); Z = norm(Z)
    Zs.append(Z); Ws.append(W)

    Zdot = (mu - np.abs(Z)**2)*Z + Z*omega*1j + couple(W, Z)
    Z = Z + deltaT*Zdot

    utility = np.zeros(Z.shape)
    utility[np.abs(Z) > 0.5] = 1.0
    
    W = W + ita*((utility + 1)*Z*star(nbr(Z)) - W)

Zs = np.array(Zs)


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
