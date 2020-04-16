
import numpy as np
import matplotlib.pyplot as plt


def hopf(mu, omega, Z):
    T = 50
    deltaT = 0.1
    Zs = []
    for _ in range(int(T/deltaT)):
        Zs.append(Z)
        deltaZ = (mu - np.abs(Z)**2)*Z + Z*omega*1j 
        Z = Z + deltaZ*deltaT
    return np.array(Zs)


Z = lambda: np.random.randn() + np.random.randn()*1j

plt.subplot(1, 3, 1)
Zs = hopf(0.01, np.pi/6., Z())
plt.plot(np.real(Zs), np.imag(Zs), 'b')
plt.plot(np.real(Zs[0]), np.imag(Zs[0]), '*g')
#plt.xlim(-10, 10)
#plt.ylim(-10, 10.0)
plt.grid()
plt.xlabel("a")

plt.subplot(1, 3, 2)
Zs = hopf(1.0, np.pi/6., Z())
plt.plot(np.real(Zs), np.imag(Zs), 'g')
plt.plot(np.real(Zs[0]), np.imag(Zs[0]), '*r')
#plt.xlim(-10, 10)
#plt.ylim(-10, 10.0)
plt.grid()
plt.xlabel("b")

plt.subplot(1, 3, 3)
Zs = hopf(20., np.pi/6., Z())
plt.plot(np.real(Zs), np.imag(Zs), 'r')
plt.plot(np.real(Zs[0]), np.imag(Zs[0]), '*b')
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.grid()
plt.xlabel("c")


plt.show()
