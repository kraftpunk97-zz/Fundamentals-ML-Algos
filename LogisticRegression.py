'''
Binary classification using Logistic Regression via Stochastic Gradient Descent Algorithm.
Data set obtained from the wikipedia page of Logistic Regression (https://en.wikipedia.org/wiki/Logistic_regression)

Logistic Function = f(x) = 1/(1 + e^(-x)) = hypothesis
'''

import numpy as np
import matplotlib.pyplot as plt

x = [ 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,    # x corresponds to hours of study
      2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50 ]
y = [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1 ]   # y corresponds to whether a student passed(1) or failed(0)   

theta = np.array([0, 0], dtype=float)  # Parameters
alpha = 0.01  # Learning rate
no_epochs = 1000  # No. of iterations

data_set = np.column_stack((np.ones(len(x)), x, y))

def log_likelihood(X, Y, theta):    # Using Log likehood to ease the computational load
    return -((1 - Y) * np.sum(np.multiply(theta, X)) +
                 np.log( 1 + np.exp(-1 * np.sum(np.multiply(theta, X)))))

hypothesis = lambda X, theta: 1/(1 + np.exp(-1 * np.sum(np.multiply(theta, X))))

for epoch in range(no_epochs):
    log_L = 0.0
    for training_eg in data_set:
        Y = training_eg[-1]
        X = training_eg[0:-1]
        log_L += log_likelihood(X, Y, theta)
        grad = (Y - hypothesis(X, theta)) * X
        theta = theta + alpha * grad
    print("Log likelihood after {} iterations = {}".format(epoch, log_L))


Xp = np.column_stack(([1] * 5, np.arange(1, 6)))
Yp = np.array([hypothesis(test_X, theta) for test_X in Xp])

plt.scatter(x, y, c="green", label="Training example")
plt.scatter(Xp[:, 1], Yp, c="red", label="Test data")

plt.xlabel("Hours of study")
plt.ylabel("Probability of passing")
plt.title("Prob. of passing for various hours of study (alpha={}, epochs={})".format(alpha, no_epochs))
plt.legend(loc="best")

plt.show()