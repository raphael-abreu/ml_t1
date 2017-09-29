import numpy as np
def computarCusto(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0   
    J = 1/(2*m)*np.sum(np.square(X.dot(theta)-y))
    return(J)