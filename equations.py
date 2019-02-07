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

