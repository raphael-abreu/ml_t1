import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

datafile = '../dados/ex2data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size 
X = np.insert(X,0,1,axis=1)


pos = np.where(y==1)
neg = np.where(y==0)

p1 = plt.plot(X[pos,1],X[pos,2],'k+',label='Admitted')
p2 = plt.plot(X[neg,1],X[neg,2],'yo',label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((p1[0], p2[0]), ('Admitted', 'Not Admitted'), numpoints=1, handlelength=0)
plt.show()
plt.savefig('dataset.png')

theta = np.zeros((X.shape[1],1))

from funcaoCustoRegressaoLogistica import funcaoCustoRegressaoLogistica,GD
from sigmoide import sigmoide

cost = funcaoCustoRegressaoLogistica(theta,X,y)
grad = GD(theta, X, y)

print('Cust: \n', cost)
print('Grad: \n', grad)



from scipy.optimize import minimize

res = minimize(funcaoCustoRegressaoLogistica, theta, args=(X,y), method=None, jac=GD, options={'maxiter':400})

prob = sigmoide(np.array([1, 45, 85]).dot(res.x.T))
print('Um candidato com notas 45 e 85 será aprovado com probabilidade de ',prob)

from predizer import predizer
p = predizer(res.x, X) 
print('Porcentagem de acertos : {}%'.format(100*sum(p == y.ravel())/p.size))