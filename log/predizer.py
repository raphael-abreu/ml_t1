from sigmoide import sigmoide
import numpy as np

def predizer(theta, X):
    m = X.shape[0] 
    p = np.zeros((m, 1))

    sigValue = sigmoide( np.dot(X,theta) )
    p = sigValue >= 0.5

    return p