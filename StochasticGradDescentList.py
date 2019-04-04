'''
Implementation of the stochastic gradient descent algorithm using lists. Provides a good approximation
for a very large dataset.
'''

import numpy as np
import matplotlib.pyplot as plt

x = [2, 0, 1, 0, 9, 3, 2, 4, 9, 2, 5, 1, 3, 1, 3, 9, 4, 1, 9, 1]
y = [6, 3, 2, 7, 1, 6, 0, 3, 1, 1, 2, 3, 9, 7, 1, 4, 0, 3, 1, 5]
#x = [1, 2, 4, 3, 5]
#y = [1, 3, 3, 2, 5]

x = [[1]* len(x), x]

theta = [0, 0]
alpha = 0.001
epochs = 137

X = np.arange(0, 10, 1)
Y = theta[0] + theta[1] * X
plt.plot(X, Y, c='blue', label="Initial plot")

for k in range(epochs):
    sum_error  = 0
    for i in range(len(y)):
        grad = [0.0, 0.0]
        error = y[i] - sum([theta[j] * x[j][i] for j in range(len(x))])
        sum_error += error**2
        grad = [-(error * x[j][i]) for j in range(len(x))]
        theta = [theta[j] - alpha * grad[j] for j in range(len(x))]
        print("theta after {} epochs is ".format(k+1), theta)

        #X = np.arange(0, 10, 1)
        #Y = theta[0] + theta[1] * X
        #plt.plot(X, Y, c='blue')

    print("Cost = {}".format(sum_error))

plt.scatter(x[1], y, c='green', label="Data Point")

X = np.arange(0, 10, 1)
Y = theta[0] + theta[1] * X
plt.plot(X, Y, c='red', label="Final plot")
plt.legend(loc='best')

plt.show()
