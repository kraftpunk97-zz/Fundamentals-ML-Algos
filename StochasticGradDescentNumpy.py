'''
Implementation of the stochastic gradient descent algorithm using lists using NumPy. Provides a good approximation
for a very large dataset.
'''

import numpy as np
import matplotlib.pyplot as plt

x = [2, 0, 1, 0, 9, 3, 2, 4, 9, 2, 5, 1, 3, 1, 3, 9, 4, 1, 9, 1]
y = [6, 3, 2, 7, 1, 6, 0, 3, 1, 1, 2, 3, 9, 7, 1, 4, 0, 3, 1, 5]

data_set = np.column_stack(([1]*len(x), x, y))

theta = np.array([0, 0], dtype=float)
alpha = 0.001
no_epochs = 1000

X = np.array([[1]*10, np.arange(0, 10, 1)])
Y = np.sum(np.multiply(X, theta.reshape((len(theta), 1))), axis=1)
plt.plot(X, Y, c='blue', label="Initial plot")

for epoch in range(no_epochs):
    sum_error = 0
    for train_ex in data_set:
        hypo = np.sum(np.multiply(train_ex[:-1], theta))
        error = train_ex[-1] - hypo
        sum_error += error ** 2
        grad = -1 * train_ex[:-1] * error
        theta = theta - grad * alpha
    print("Cost after {} epochs : {}".format(epoch, sum_error))

plt.scatter(x, y, c='green', label="Data Point")

X = np.array([[1]*10, np.arange(10)])

theta = theta.reshape((len(theta), 1))
Y = np.sum(np.multiply(X, theta), axis=0)
plt.plot(X[1], Y, c='red', label="Final plot")

plt.legend(loc="best")
plt.show()