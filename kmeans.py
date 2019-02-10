
# Inputs:
#   nk is number of means to calculate
#   data is the data to be used
#   epsilon is relative error of mean change limits
#
# Outputs:
#   array of means

import numpy as np
import random as random

def kmeans( nk = 1, data = np.empty((1,1)), epsilon = 0.01, looplimit = 100):
    n_points = len(data) #number of data points
    dimension = len(data[0]) #dimensional size

    #finding minimum and maximum
    minarray = [0]*dimension
    maxarray = [0]*dimension

    #finding min and max
    for i in range(n_points):
        for j in range(dimension):
            if data[j] < minarray[j]:
                minarray[j] = data[j]
            if data[j] > maxarray[j]:
                maxarray[j] = data[j]
    
    #array for means
    kmeans = np.zeros((nk,dimension))
    #assigning a random start
    for i in range(nk):
        for j in range(dimension):
            kmeans[i][j] = random.uniform(minarray[j], maxarray[j])
    
    #memberhsip array for mean recalculations
    membership = np.empty((np,1))

    #kmeans looping
    while 1:
        #membership assignment
        nmembers = np.empty((nk,1))
        for npoint in range(n_points):
            costarray = [0] * nk
            nmembers = [0] * nk
            for i in range(nk):
                costarray[i] = np.linalg.norm(kmeans[i] - data[npoint])
            minindex = costarray.argmin()
            membership[npoint] = minindex
            nmembers[minindex] += 1

        #recalculation of means
        newkmeans = np.zeros((nk,dimension))
        for npoint in range(n_points):
            index = membership[npoint]
            newkmeans[index] += data[npoint] / nmembers[index]

        #finding k means shift
        dkmeans = numpy.empty((nk,1))
        for index in range(nk):
            dkmeans[index] = np.linalg.norm(kmeans[index] - newkmeans[index]) /\
                             np.linalg.norm(kmeans[index])
        #Epsilon condition break
        if np.amax(dkmeans) < epsilon:
            break

        #Loop break
        looplimit -= 1
        if looplimit == 0:
            #if loop is initially set as 0, then while loop
            #otherwise, it is essentially a for loop
            break
            
        #Updating k means
        kmeans = newkmeans
    #Outside of while loop
    return kmeans

