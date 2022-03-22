import numpy as np


# Calculates the softmax of vector x with given temperature
def softmax(x, temp):
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax
