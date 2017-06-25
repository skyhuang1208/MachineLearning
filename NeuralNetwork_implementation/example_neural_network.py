import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt

x= np.array([[0,0],[1,0],[0,1],[1,1]]) # XOR
y= np.array([0,1,1,0])

fit1= nn.classification(2, [3,5])
fit1.fit(x,y)

A1 = np.linspace(-2, 2, 128)
A2 = np.linspace(-2, 2, 128)
(a1, a2)= np.meshgrid(A1, A2)
fa1= a1.flatten()
fa2= a2.flatten()
B= np.transpose( np.array([fa1, fa2]) )
pB= fit1.predict(B)
#print(pB)
for i in range(len(fa1)): print(fa1[i], fa2[i], pB[i,0])
ppB= pB[:,0].reshape((128,128))
#f= 1/(np.exp(w1*a1 + w2*a2 + w12*a1*a2 + w11*a1*a1 + w22*a2*a2)+1)

plt.figure(figsize=(8,8))
plt.scatter(x[:,0], x[:,1], c=y, s=5)
plt.contour(a1, a2, ppB, [0.2, 0.5, 0.8], colors='k')
plt.show()
