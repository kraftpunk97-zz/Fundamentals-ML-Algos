'''
Creating my very own implementation of the K Nearest Neighbour classifier,
that agrees with scikit-learn's classifier interface paradigm.
Accuracy is over 90% for the Iris dataset.
'''
import time
import numpy as np
from collections import Counter

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, k=5):
        predictions = []
        for row in X_test:
            distance_vect = []  # A data structure to hold the calculated distance
                                # between testing example and training example
            for train_ex_data, train_ex_label in zip(self.X_train, self.y_train):
                dist = ScrappyKNN.distance(row, train_ex_data)
                distance_vect.append((dist, train_ex_label))
        
            distance_vect.sort(key=lambda x: x[0])
            k_nearest = distance_vect[:k]
            count = Counter([entry[1] for entry in k_nearest])  # Creates a dictionary with labels as keys
                                                                # and the count of their occurances in k_nearest as values
            
            predictions.append(max(count, key=lambda x: count[x]))  # The prediction is the label with most occurences
                                                                    # in the k-nearest training examples.
        return predictions
    
    @staticmethod
    def distance(point_a , point_b):
        '''
        Utility for calculating the "distance" between the two data points.
        '''
        error = np.subtract(point_a, point_b)
        return np.sqrt(np.sum(error * error))

# Loading the Iris Dataset
from sklearn import datasets
iris = datasets.load_iris()

# Restructing the dataset into the conventional X and y matrices
X = iris.data
y = iris.target

# Normalizing data
from sklearn.preprocessing import normalize
X = normalize(X, axis=0, norm='l1')

# Dividing the data for training and testing purposes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Building our classifier and training it.
#from sklearn.neighbors import KNeighborsClassifier
clf = ScrappyKNN()
clf.fit(X_train, y_train)


# Predictions for the testing data made by the classifier
start_time1 = time.time()
predictions = clf.predict(X_test)
end_time1 = time.time()

print("-------%s seconds for %d testing examples-------" % (end_time1-start_time1, len(y_test)))

# Checking the accuracy of the classifier.
from sklearn.metrics import accuracy_score

print("Features used are {} (values displayed are normalized values)\n".format(iris.feature_names))
for test, _y, pred in zip(X_test, y_test, predictions):
    if _y != pred:
        print("Mismatch for {}: predicted = {}, observed = {}".format(test,
                                                                      iris.target_names[pred],
                                                                      iris.target_names[_y]))

print("\nAccuracy = %f" % float(accuracy_score(y_test, predictions, normalize=True) * 100))