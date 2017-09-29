import numpy as np

def normalizarCaracteristica(X):
    
    X_norm = X
    valor_media    = np.zeros((1, X.shape[1]))
    desvio_padrao = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        # Evitar usar a primeira coluna
        if not i: continue
        valor_media[:,i] = np.mean(X[:,i])
        desvio_padrao[:,i] = np.std(X[:,i])
        X_norm[:,i] = (X[:,i] - float(valor_media[:,i]))/float(desvio_padrao[:,i])

    return X_norm, valor_media, desvio_padrao