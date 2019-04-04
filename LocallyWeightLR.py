import numpy as np
import matplotlib.pyplot as plt

#x = [2, 0, 1, 0, 9, 3, 2, 4, 9, 2, 5, 1, 3, 1, 3, 9, 4, 1, 9, 1]
#y = [6, 3, 2, 7, 1, 6, 0, 3, 1, 1, 2, 3, 9, 7, 1, 4, 0, 3, 1, 5]
x = np.arange(10)
y = np.sin(x)

data_set = np.column_stack(([1]*len(x), x, y))

theta = np.array([0, 0], dtype=float)  # Initializing parameters
m = len(data_set)  # No. of available training examples

tau = 0.5  # Bandwidth
alpha = 0.01  # Learning rate
no_iterations = 100  # No. of iterations
xp = 3.5  # x, whose y needs to be predicted

plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y, c='green', label="Training example")
# xp = int(input("Enter the value of x: "))
xp = np.array([1, xp], dtype=float)


weight = np.exp(np.square(np.subtract(data_set[:, 1], xp[1:]))/(-2 * tau**2)).reshape(m, 1)

# weight_mat = np.diag(weight)

for iteration in range(no_iterations):
    hypo = np.sum(np.multiply(data_set[:, 0:-1], theta), axis=1).reshape((m, 1))
    error = data_set[:, -1][np.newaxis].T - hypo
    cost = np.sum(np.square(error))/m
    print("Average cost for this iteration is : {}".format(cost))
    
    mid = (weight * error) * data_set[:, 0:-1]
    grad = np.sum(mid, axis=0)
    theta = theta + alpha * grad
    
    print("Updated theta", theta)

yp = np.sum(xp * theta)
print(" For x =", xp, ", y = {}".format(yp))
plt.scatter(xp[1], yp, c='red', label="Predicted value of y")

'''
X = np.arange(10)              # Uncomment if you want to see the line
Y = theta[0] + theta[1] * X    # by the final values of the 
plt.plot(X, Y, c='red')        # values of the parameters.
'''

plt.title("LOWESS for x={} (BD={}, alpha={}, {} cycles)".format(xp[1], tau, alpha, no_iterations))
plt.legend(loc="best")

plt.show()