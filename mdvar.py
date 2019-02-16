"""
Multidimensional Variance: Variance of norms

Inputs:
	data == list of data around mean
	mean == mean of data
	
Output:
	RSS == residuals sum of squares
"""
# necessary library for norm
import numpy as np

def mdvar(data, mean):
	RSS = 0
	N = data.shape[0] #Number of data points
	
	for i in range(N):
		RSS += (np.linalg.norm(data[i, :] - mean)) ** 2

	return RSS
