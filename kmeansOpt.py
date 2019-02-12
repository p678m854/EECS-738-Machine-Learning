"""
kmeansOpt: Optimized number of clusters for data set in K-means algorithm 

Inputs: data       == data set of interest
        Ftestlim   ==  upperbounds of F testing (F test should converge to 1)
        kmeansIter == Iterations for K-means algorithm

Outputs: kopt == Optimal K in K-means based on Calinski-Harabasz index
"""

#import numpy as np

def kmeansOpt(data, Ftestlim = 0.95, kmeansIter = 100):
	#loop preallocations
	clusters = 2 #Clusterings start at 2
	Ftest = 0 #Initialize F test results
	CHList = np.zeros((1,1)) #List of CH index results
	
	#While loop
	while Ftest < Ftestlim:
		kmeans = kmeans(data, clusters++) #initialize kmeans and post indent clusters
		(centroids, nearestCent) = kmeans.train(kmeansIter) #train K-means for kmeansIter iterations
		Ftest = F_test(data, centroids, nearestCent) #F test for loop conditions
		
		# update CHlist
		if CHList[0] == 0:
			CHList[0] = CHtest(data, centroids, nearestCent)
		else:
			CHList.append(CHtest(data, centroids, nearestCent)
			
	#return opt K for K-means
	return (np.argmax(CHList) +2)
