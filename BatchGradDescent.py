'''
Implementation of the batch gradient descent algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt

data_set = [ [2, 0, 1, 0, 9, 3, 2, 4, 9, 2, 5, 1, 3, 1, 3, 9, 4, 1, 9, 1],
            [6, 3, 2, 7, 1, 6, 0, 3, 1, 1, 2, 3, 9, 7, 1, 4, 0, 3, 1, 5] ]

'''
x = np.arange(10)
y = np.sin(x)
data_set = [x, y]
'''

initial_theta = [0, 0]

X = np.column_stack((np.ones(len(data_set[0]), dtype=int), data_set[0]))
Y = np.transpose(np.array(data_set[-1], dtype=int)[np.newaxis])
theta = np.transpose(np.array(initial_theta, dtype=float)[:][np.newaxis])

m = len(X)
alpha = 0.01
iterations = 1000

plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y for {} iterations and LR = {}'.format(iterations, alpha))

plt.scatter(data_set[0], data_set[-1], c='green', label='Data points')

for i in range(iterations):
    hypo = np.dot(X, theta)
    error = Y - hypo
    avg_cost = np.dot(np.transpose(error), error) / m
    print("Average cost for iteration {} = {}".format(i, avg_cost[0][0]))
    grad = np.transpose((np.sum(error * X, axis=0)/m)[np.newaxis])
    theta = theta + alpha * grad
    
    #x = np.arange(0, 10)
    #y = theta[0][0] + theta[1][0] * x
    #plt.plot(x, y, c='blue')

x = np.arange(0, 10)
y = theta[0][0] + theta[1][0] * x
plt.plot(x, y, c='red', label='Final result')

plt.legend(loc = 'best')
plt.show()