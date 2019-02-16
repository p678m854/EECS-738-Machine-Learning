"""
This file will determine centroids, centroid membership, and kmeans.

"""

import numpy as np
from equations import relSSECalc, CHtest, mdvar


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


    def create_centroids(self):
        """
        Creates how many centroids are wanted for the given daataset
        inputs:
        data: the given dataset
        num_clusters:  number of clusters you want for the dataset
        returns:
        random centroids
        """

        # Get number of training examples.
        n_data = self.data.shape[0]

        # Randomly reorder indices of training examples.
        random_ids = np.random.permutation(n_data)

        # Take the first K examples as centroids.
        centroids = self.data[random_ids[:self.clusters], :]

        # Return generated centroids.
        return centroids


    def nearest_centroid(self, centroids):
        """
        Finds the nearest centroid for each data point.
        inputs:
        data: the given dataset
        centroids: the centroids for the dataset
        returns:
        the closest centroid for each datapoint in the dataset
        """

        # Get number of training examples.
        n_data = self.data.shape[0]
        features = self.data.shape[1]

        # Get number of centroids.
        n_centroids = centroids.shape[0]

        # We need to return the following variables correctly.
        nearest_centroid = np.zeros((n_data, 1))

        # find the closest centroid for every data point
        for data_point in range(n_data):
            distances = np.zeros((n_centroids, 1))
            for which_centroid in range(n_centroids):
                for j in range(features):
                    distances[which_centroid] += (self.data[data_point, j] - centroids[which_centroid, j]) ** 2
            nearest_centroid[data_point] = np.argmin(distances)

        return nearest_centroid


    def place_centroids(self, nearest_centroid):
        """
        Places the centroids according to the means of the data points that were assigned to it.
        inputs:
        data: given dataset
        nearest_centroid: the centroid assigned to the data point
        clusters: specified clusters
        """

        # Get number of features.
        features = self.data.shape[1]

        # create starting points for all the centroid ids.
        centroids = np.zeros((self.clusters, features))

        # calculate the means for the data points associated with a centroid
        for centroid in range(self.clusters):
            associated_centroid = nearest_centroid == centroid
            centroids[centroid] = np.mean(self.data[associated_centroid.flatten(), :], axis=0)

        return centroids

    def train(self, iterations = 100):
        """
        This function will use K-Means algorithm to cluster the data
        Inputs:
        iterations: number of iterations for training
        Returns:
        Centroids, and ID of closest Centroid
        """

        # Generate random centroids based on training set.
        centroids = k_means.create_centroids(self)

        # create an empty array for nearest centroids
        num_examples = self.data.shape[0]
        nearest_centroid = np.empty((num_examples, 1))

        # Run K-Means.
        for _ in range(iterations):
            # Find the closest centroids for training examples.
            nearest_centroid = k_means.nearest_centroid(self.data, centroids)

            # Compute means based on the closest centroids found in the previous part.
            centroids = k_means.place_centroids(
                self,
                nearest_centroid,
            )

        return centroids, nearest_centroid

    """
    kmeansOpt: Optimized number of clusters for data set in K-means algorithm 
    Inputs: data       == data set of interest
            Ftestlim   ==  upperbounds of F testing (F test should converge to 1)
            kmeansIter == Iterations for K-means algorithm
    Outputs: kopt == Optimal K in K-means based on Calinski-Harabasz index
    """


    def kmeansOpt(self, SSELimit = 0.0001, innerIterLim = 100):
        # loop preallocations
        self.clusters = 1  # Clusterings start at 1 and increments at top of loop
        
        # Preallocating CH and F-test limits
        relativeSSE = 1  # Initialize Relative Sum of Squared Error test results
        CHList = []  # List of CH index results
        iterCount = 0

        # While loop
        while relativeSSE > SSELimit:
            self.clusters += 1
            iterCount += 1
            #k_means = k_means(data, clusters)  # initialize kmeans and post indent clusters
            (centroids, nearestCent) = k_means.train(self,innerIterLim)  # train K-means for kmeansIter iterations
            
            CHList.append(CHtest(self.data, centroids, nearestCent)) # update CHlist
            relativeSSE = relSSECalc(self.data, centroids, nearestCent)  # F test for loop conditions

            #Printing loop information
            print("Iteration: ",iterCount)
            print("Clusters: ",self.clusters)
            print("Relative unexplained variance results: ",relativeSSE)
            print("CH-index test: ",CHList[-1],"\n")
        # return opt K for K-means
        self.clusters = (CHList.index(max(CHList)) + 2)
        (centroids, nearestCent) = k_means.train(self, 100)
        return centroids, nearestCent