"""
F_test: F statistic test, specificially ANOVA version
				In use for elbow method of kmeans
				
Inputs: data        == data set of interest
				centroids   == list of centroids
				nearestCent == list of nearst centroid index for data
				
Outputs: F test result which is (explained variance)/(unexplained variance)
"""

def F_test(data, centroids, nearestCent):
	# Overall method parameters
	K = centroids.shape[0] #Number of means/clusters
	N = data.shape[0] #     Total number of data points
	
	# Loop preallocations
	expVar = 0
	unexpVar = 0
	
	# For loop through centroids
	for cent in range(K)
		assoc_cent = nearestCent == cent
		ni = data[assoc_cent.flatten(), :].shape[0]
		expVar += ni * (np.linalg.norm(centroids[cent] - d_mean) ** 2) / (K - 1)
		unexpVar += mdvar(data[assoc_cent.flatten(), :], centroids[cent]) / (N - K)
		
	# Returning F test results
	return (expVar/unexpVar)
