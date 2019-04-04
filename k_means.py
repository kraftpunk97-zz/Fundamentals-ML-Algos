#! /usr/local/bin/python3
'''Implementing the k-Means Clustering Algorithm from scratch'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def kmeans(dataset, num_clusters=3, max_iter=10):
    '''
    The k-means implemention
    '''
    m, n = dataset.shape
    cluster_arr = np.zeros(dtype=int, shape=(m,))

    # Initialize clusters
    cluster_centroids = np.array(dataset[np.random.randint(low=0,
                                                           high=m,
                                                           size=num_clusters
                                                          )])
    for i in range(max_iter):

        # To find which cluster a point belongs to, just find the cluster centroid
        # closest to that point.
        for i in range(m):
            cluster_arr[i] = np.argmin(np.linalg.norm(cluster_centroids - dataset[i], axis=1))

        # This is where we update the cluster centroids to the mean position of
        # the datapoints in that cluster
        for i in range(num_clusters):
            cluster_family = dataset[np.where(cluster_arr == i)]
            cluster_centroids[i] = np.mean(cluster_family, axis=0)

    return cluster_centroids, cluster_arr


def nice_plot(dataset, num_clusters, cluster_arr, cluster_centroids):
    '''A utility function that shows the distribution of data points and the cluster they belong to.'''
    color_list = ('red', 'blue', 'green', 'black', 'brown', 'turquoise')
    plt.figure()
    for i in range(num_clusters):
        idx = np.where(cluster_arr == i)
        plt.scatter(dataset[idx, 0], dataset[idx, 1], c=color_list[i], alpha=0.4)
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='white')
    plt.show()



data = pd.read_csv('speeding_data.csv', header=0, delimiter='\t')
data = data.drop('Driver_ID', axis=1)

dataset = np.array(data)

print("First, let's see how the points are distributed...")
data.plot(x='Distance_Feature', y='Speeding_Feature', kind='scatter', alpha=0.4,
          c='black')
plt.show()

print('Running the kmeans algorithm...')

num_clusters = 4
num_iterations = 8
cluster_centroids, cluster_arr = kmeans(dataset, num_clusters, num_iterations)

print('Alogrithm ran successfully. Plotting results')
nice_plot(dataset, num_clusters, cluster_arr, cluster_centroids)
