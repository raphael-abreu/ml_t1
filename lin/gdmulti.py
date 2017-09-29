from computarCustoMulti import computarCustoMulti
import numpy as np


def GD(X, y, theta, alpha=0.01, epsilon=0.0001):
    m = y.size
    J_history = []
    
    while True:
        old_theta = theta
        theta = theta - alpha*(1/m)*np.dot(X.T,X.dot(theta)-y)
        J_history.append(computarCustoMulti(X, y, theta))
        if np.linalg.norm(theta) - np.linalg.norm(old_theta) < epsilon:
            break
    return(theta, J_history)