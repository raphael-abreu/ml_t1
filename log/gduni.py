from computarCusto import computarCusto
import numpy as np


def GD(X, y, theta=[[0],[0]], alpha=0.01, epsilon=0.0001):
    m = y.size
    J_history = []
    
    while True:
        old_J = J_history[J_history.len()]
        theta = theta - alpha*(1/m)*np.dot(X.T,X.dot(theta)-y)
        J_history.append(computarCusto(X, y, theta))
        if old_J - J_history[J_history.len()] < epsilon:
            break
    return(theta, J_history)