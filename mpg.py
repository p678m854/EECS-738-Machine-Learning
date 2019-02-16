"""
THis file uses definitions in kmeans.py and applies them to the auto-mpg.csv dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import equations as eq
#from kmeans2 import KMeans

#import CSV File
data = pd.read_csv('auto-mpg.csv', index_col='model year')

#take the mpg and weight columns
mpg = data['mpg']
weight = data['weight']

#take the data from 1976
data76 = data.loc[76]

#take the mpg and weight from the 1976 data
mpg76 = data76['mpg']
weight76 = data76['weight']
plt.plot(mpg76,weight76,'ro')

#take the mu and the sigma from the mpg76 data
mu = eq.getmu(mpg76)
sig = eq.getsigma(mpg76)
mu2 = eq.getmu(weight76)
sig2 = eq.getsigma(weight76)

#apply the gaussian and plot it mpg76
#plt.plot(mpg76, eq.gaussian(mpg76,mu,sig), 'go')
#plt.plot(weight76, eq.gaussian(weight76,mu2,sig2), 'bo')
#plt.show()
print(data.shape[1])
print(data)


