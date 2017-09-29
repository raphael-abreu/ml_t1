# coding=UTF-8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

datafile = '../dados/ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True)

#Lendo na forma de matriz "X" e vetor "y"
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))

m = y.size

# Por convenção, a primeira coluna é preenchida de 1's
X = np.insert(X,0,1,axis=1)


## Plotar figura
plt.figure(figsize=(10,8))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='training data')
plt.tick_params(axis='both', direction='in', width=2, length=7,bottom='on', top='on', left='on', right='on')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.xticks(np.arange(4, 24, 2))
plt.ylim(-5, 25)
plt.xlim(4, 24)



## Computar custo
from computarCusto import computarCusto
print(computarCusto(X, y))



## Executar GD univariável
from gduni import GD
theta, Cost_J = GD(X, y)

## traçar linha do gd
plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
plt.legend(loc='lower right', shadow=False, fontsize='x-large', numpoints=1)
plt.show()
plt.savefig('dataset_fit.png')

## plotar convergência
plt.figure(figsize=(10,6))
plt.plot(Cost_J)
plt.show()
plt.savefig('custo.png')




## predizer

predict1 = theta.T.dot(np.array([1, 3.5]));
print('Lucro de {:f} para uma população de 35,000 habitantes '.format( float(predict1*10000) ))
predict2 = theta.T.dot(np.array([1, 7]));
print('Lucro de {:f} para uma população de 70,000 habitantes '.format( float(predict2*10000) ))


# Plotar curvas de nível e plot 3D

from mpl_toolkits.mplot3d import axes3d



# Grids
theta0 = np.linspace(-10, 10, 50)
theta1 = np.linspace(-1, 4, 50)
theta0_vals, theta1_vals = np.meshgrid(theta0, theta1, indexing='xy')

custos = np.zeros((theta0.size,theta1.size))

for (i,j),v in np.ndenumerate(custos):
    custos[i,j] = computarCusto(X,y, theta=[[theta0_vals[i,j]], [theta1_vals[i,j]]])



fig1 = plt.figure(figsize=(6,5))
ax = plt.contour(theta0_vals, theta1_vals, custos, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
plt.scatter(theta[0],theta[1], c='r')

plt.xlabel('theta_0')
plt.ylabel('theta_1')

plt.show()
plt.savefig('curvas.png')






fig2 = plt.figure(figsize=(6,5))
ax = fig2.gca( projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, custos, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax.set_zlabel('Cost')
ax.set_zlim(custos.min(),custos.max())
ax.view_init(elev=15, azim=230)

plt.xlabel('theta_0')
plt.ylabel('theta_1')

plt.show()
plt.savefig('3d.png')




## regressão com várias características

datafile = '../dados/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)


X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
X = np.insert(X,0,1,axis=1)


from normalizarCaracteristica import normalizarCaracteristica
Xnorm,val_media,desvio_padrao = normalizarCaracteristica(X)

theta = np.zeros((Xnorm.shape[1],1))
theta, Cost_J = GD(Xnorm, y,theta)

plt.figure(figsize=(10,6))
plt.plot(Cost_J)
plt.show()
plt.savefig('custo_multi.png')









