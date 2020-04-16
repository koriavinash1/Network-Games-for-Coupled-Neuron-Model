
import numpy as np
import matplotlib.pyplot as plt

complex = lambda: np.random.uniform(0, 1) + np.random.uniform(0, 1)*1j
star = lambda x: np.conjugate(x)
norm = lambda x: x/np.abs(x)

omega = np.pi/6.
mu = 1.0
T = 50
deltaT = 0.01
ita = 1e-2
Z1s = []; Z2s =[]
W1s = []; W2s = []
payoff = {'C-C': (1, 1), 'C-DC': (1, 0),'DC-C': (0, 1),'DC-DC': (0, 0)}


# initalizations
Z1 = complex(); Z2 = complex()
W1 = complex(); W2 = complex()

for _ in range(int(T/deltaT)):

    W1 = norm(W1); W2 = norm(W2)
    Z1 = norm(Z1); Z2 = norm(Z2)

    Z1s.append(Z1); Z2s.append(Z2)
    W1s.append(W1); W2s.append(W2)

    Z1dot = (mu - np.abs(Z1)**2)*Z1 + Z1*omega*1j + W1*Z2
    Z1 = Z1 + deltaT*Z1dot

    Z2dot = (mu - np.abs(Z2)**2)*Z2 + Z2*omega*1j + W2*Z1
    Z2 = Z2 + deltaT*Z2dot
    
    s1 = 'C' if np.abs(Z1) > 0.5 else 'DC'
    s2 = 'C' if np.abs(Z2) > 0.5 else 'DC'
    s = s1 + '-' + s2
    
    W1 = W1 + ita*((payoff[s][0] + 1)*Z1*star(Z2) - W1)
    W2 = W2 + ita*((payoff[s][1] + 1)*Z2*star(Z1) - W2)

Z1s = np.array(Z1s)
Z2s = np.array(Z2s)



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


