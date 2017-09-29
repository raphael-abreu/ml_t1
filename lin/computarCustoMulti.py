import numpy as np
def computarCustoMulti(X, y, theta):
    m = y.size
    J = 0   
    J = 1/(2*m)*np.sum(np.square(X.dot(theta)-y))
    return(J)