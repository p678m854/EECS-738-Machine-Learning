"""
CHtest: Calinski-Harabasz index test
        Used to evaluate optimal K-means clustering

Inputs: data        == data set of interest
        centroids   == list of centroids from K-means alg.
        nearestCent == list of nearest centroid index for data
				
Output: CHindex == Calinski-Harabasz index test result

Ref: http://ethen8181.github.io/machine-learning/clustering_old/clustering/clustering.html
"""

import numpy as np

def CHtest(data, centroids, nearestCent):
	#Overall method characteristics
	d_mean = np.mean(data, axis = 0) #vector mean of data
	TSS = mdvar(data, d_mean) #total sum of squares
	N = data.shape[0] #number of data points
	K = centroids.shape[0] #number of centroids
	
	#Loop and preallocation
	SSW = 0 #Sum of squares within clusters
	for cent in range(K):
		assoc_cent = nearestCent == cent
		SSW += mdvar(data[assoc_cent.flatten(), :], centroids[cent])
		
	#CH index calc
	SSB = TSS - SSW #Sum of squares between data sets
	CHindex = (SSB / SSW) * (N - K) / (K - 1)
	return CHindex
