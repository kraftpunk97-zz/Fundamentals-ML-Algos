import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

k = 3  # The number of classes
n = 8  # Number of features
m = 199 # Number of training examples
X = [] # Matrix of training examples
y = [] # vector specifying the class of the training example
mu = [] # nxk matrix containing the mean for the gaussian distribution for a particular class
phi = [] # 1xk vector containing the Bernoulli probabililty of each class appearing in the training examples

for x, y in zip(X, y):
    phi[y] += 1
    mu[y] += x


_mu = mu / phi
_phi = phi / m

sigma = []  # The covariance matrix initialized to a zero nxn matrix
i = 0
while i < m:
    error_vec = x[i] - _mu[y[i]]
    sigma = sigma + error_vec.T @ error_vec
    i += 1
sigma = sigma / m
sigma_inv  = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)

# Dropping the Bayes...
x = []
p_of_x_given_y = []
for i in range(k):
    error_vec = x - mu[i]
    p_of_x_given_y[i] = 1/((2*np.pi)**(n/2) * sigma_det**(1/2)) * np.exp(error_vec @ sigma_inv @ error_vec.T * (-1/2)) * phi[i]
