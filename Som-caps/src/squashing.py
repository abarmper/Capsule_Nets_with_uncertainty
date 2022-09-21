'''
Helps me decide which squashing function to use.
'''

import matplotlib.pyplot as plt
import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def squash(x):
    return x**2/((1+x**2)*abs(x))

x = np.arange(-10,10,0.05)
y_tanh = np.tanh(x)
y_sigmoid = sigmoid(x)
y_squashed = squash(x)


fig, ax = plt.subplots()
plt.grid()
ax.plot(x,y_tanh, label = 'tanh')
ax.plot(x,y_sigmoid, label = 'sigmoid')
ax.plot(x,y_squashed, label = 'squash')
ax.legend()
plt.title('Normalize Functions')
plt.savefig('squashing.png')