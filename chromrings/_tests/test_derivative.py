import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.misc import derivative

x1 = 46.7
s1 = 26.6
A1 = 0.64622

x2 = 81.58
s2 = 8
A2 = 0.3

B = 0

def gaussian(x):
    y1 = A1*np.exp(-((x-x1)**2/(2*s1**2)))
    y2 = A2*np.exp(-((x-x2)**2/(2*s2**2)))
    y = y1 + y2 +B
    return y

def gaussian_gradient(xx):
    grad = np.zeros(len(xx))
    for i, x in enumerate(xx):
        grad[i] = derivative(gaussian, x, n=1)
    return grad

def gaussian_second_derivative(xx):
    grad = np.zeros(len(xx))
    for i, x in enumerate(xx):
        grad[i] = derivative(gaussian, x, n=2)
    return grad

def roots_initial_guess(yy, xx):
    yy_sign = np.sign(yy)
    sign_change = ((np.roll(yy_sign, 1) - yy_sign) != 0).astype(bool)
    sign_change[0] = False
    roots = xx[sign_change]
    return roots

# print(gaussian_gradient(30.9))

xx = np.linspace(0, 120, 1000)
y_grad = gaussian_gradient(xx)

plt.plot(xx, y_grad)
plt.show()

print('Solving...')

inital_guess = roots_initial_guess(y_grad, xx)

first_solutions = fsolve(gaussian_gradient, inital_guess)

print(first_solutions)
print(gaussian_gradient(first_solutions))

y_second = gaussian_second_derivative(xx)

plt.plot(xx, y_second)
plt.show()

inital_guess = roots_initial_guess(y_second, xx)
second_solutions = fsolve(gaussian_second_derivative, inital_guess)

print(second_solutions)
print(gaussian_second_derivative(second_solutions))