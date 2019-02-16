"""
Equations to find the mu, sigma, and gaussian
"""

import numpy as np
import pandas as pd
from math import sqrt, pi, e


def getmu(lst):
    return sum(lst)/len(lst)

def getsigma(lst):
    return sqrt(np.var(lst))

def gaussian(x, mu, sig):
    return 1/(sqrt(2*pi)*sig)*e**(-0.5*((x-mu)/sig)**2)


"""
Multidimensional Variance: Variance of norms
Inputs:
	data == list of data around mean
	mean == mean of data

Output:
	RSS == residuals sum of squares
"""

def mdvar(data, mean):
    RSS = 0
    N = data.shape[0]  # Number of data points

    for i in range(N):
        RSS += np.linalg.norm(data[i, :] - mean) ** 2
    return RSS


"""
CHtest: Calinski-Harabasz index test
        Used to evaluate optimal K-means clustering
Inputs: data        == data set of interest
        centroids   == list of centroids from K-means alg.
        nearestCent == list of nearest centroid index for data

Output: CHindex == Calinski-Harabasz index test result
Ref: http://ethen8181.github.io/machine-learning/clustering_old/clustering/clustering.html
"""

def CHtest(data, centroids, nearestCent):
    # Overall method characteristics
    d_mean = np.mean(data, axis=0)  # vector mean of data
    TSS = mdvar(data, d_mean)  # total sum of squares
    N = data.shape[0]  # number of data points
    K = centroids.shape[0]  # number of centroids

    # Loop and preallocation
    SSW = 0  # Sum of squares within clusters
    for cent in range(K):
        assoc_cent = nearestCent == cent
        SSW += mdvar(data[assoc_cent.flatten(), :], centroids[cent])

    # CH index calc
    SSB = TSS - SSW  # Sum of squares between data sets
    CHindex = (SSB / SSW) * (N - K) / (K - 1)


    return CHindex


"""
relSSECalc: In use for elbow method of kmeans

Inputs: data        == data set of interest
	centroids   == list of centroids
	nearestCent == list of nearst centroid index for data

Outputs: Test result which is (unexplained variance)/(Total variance of set)
"""

def relSSECalc(data, centroids, nearestCent):
    # Overall method parameters
    K = centroids.shape[0]  # Number of means/clusters
    N = data.shape[0]  # Total number of data points
    d_mean = np.mean(data, axis=0)  # vector mean of data
    # Loop preallocations
    #expVar = 0, trying to do fraction of unexplained variances
    initialVar = mdvar(data, d_mean)
    unexpVar = 0

    # For loop through centroids
    for cent in range(K):
        assoc_cent = nearestCent == cent
        #ni = data[assoc_cent.flatten(), :].shape[0]
        #expVar += ni * (np.linalg.norm(centroids[cent] - d_mean) ** 2) / (K - 1)
        unexpVar += mdvar(data[assoc_cent.flatten(), :], centroids[cent]) / (N - K)


     # Returning F test results
    return unexpVar / initialVar

