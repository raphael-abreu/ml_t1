import numpy as np
from sigmoide import sigmoide
def funcaoCustoRegressaoLogistica(theta,X, y):
    m = y.size
    J = 0

    g_x = sigmoide(X.dot(theta))

    J = -1*(1/m)*( np.log(g_x).T.dot(y)  +  np.log(1-g_x).T.dot(1-y) )

    return(J[0])

def GD(theta, X, y):
    m = y.size
    h = sigmoide(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m)*X.T.dot(h-y)

    return(grad.flatten())
