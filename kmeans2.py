"""
This file will determine centroids, centroid membership, and kmeans.

"""

import numpy as np


class k_means:


    def __init__(self, data, clusters):
        """K Means class constructor
        Inputs:
        data: data to be processed
        clusters: number of clusters for the data
        Returns:
        none
        """
        self.data = data
        self.clusters = clusters


    def create_centroids(data, clusters):
        """
        Creates how many centroids are wanted for the given daataset
        inputs:
        data: the given dataset
        num_clusters:  number of clusters you want for the dataset
        returns:
        random centroids
        """

        # Get number of training examples.
        n_data = data.shape[0]

        # Randomly reorder indices of training examples.
        random_ids = np.random.permutation(n_data)

        # Take the first K examples as centroids.
        centroids = data[random_ids[:clusters], :]

        # Return generated centroids.
        return centroids


    def nearest_centroid(data, centroids):
        """
        Finds the nearest centroid for each data point.
        inputs:
        data: the given dataset
        centroids: the centroids for the dataset
        returns:
        the closest centroid for each datapoint in the dataset
        """

        # Get number of training examples.
        n_data = data.shape[0]

        # Get number of centroids.
        n_centroids = centroids.shape[0]

        # We need to return the following variables correctly.
        nearest_centroid = np.zeros((n_data, 1))

        # find the closest centroid for every data point
        for data_point in range(n_data):
            distances = np.zeros((n_centroids, 1))
            for which_centroid in range(n_centroids):
                distance_difference = data[data_point, :] - centroids[which_centroid, :]
                distances[which_centroid] = np.sum(distance_difference ** 2)
            nearest_centroid[data_point] = np.argmin(distances)

        return nearest_centroid


    def place_centroids(data, nearest_centroid, clusters):
        """
        Places the centroids according to the means of the data points that were assigned to it.
        inputs:
        data: given dataset
        nearest_centroid: the centroid assigned to the data point
        clusters: specified clusters
        """

        # Get number of features.
        features = data.shape[1]

        # create starting points for all the centroid ids.
        centroids = np.zeros((clusters, features))

        # calculate the means for the data points associated with a centroid
        for centroid in range(clusters):
            associated_centroid = nearest_centroid == centroid
            centroids[centroid] = np.mean(data[associated_centroid.flatten(), :], axis=0)

        return centroids

    def train(self, iterations):
        """
        This function will use K-Means algorithm to cluster the data
        Inputs:
        iterations: number of iterations for training
        Returns:
        Centroids, and ID of closest Centroid
        """

        # Generate random centroids based on training set.
        centroids = k_means.create_centroids(self.data, self.clusters)

        # create an empty array for nearest centroids
        num_examples = self.data.shape[0]
        nearest_centroid = np.empty((num_examples, 1))

        # Run K-Means.
        for _ in range(iterations):
            # Find the closest centroids for training examples.
            nearest_centroid = k_means.nearest_centroid(self.data, centroids)

            # Compute means based on the closest centroids found in the previous part.
            centroids = k_means.place_centroids(
                self.data,
                nearest_centroid,
                self.clusters
            )

        return centroids, nearest_centroid