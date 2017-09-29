import numpy as np

def sigmoide(z):
    return(1 / (1 + np.exp(-z)))
