import sys

if len(sys.argv) != 2:
    print("Do logistic fitting")
    exit("Usage: %s <input_file1>" % sys.argv[0])

import numpy as np
from sklearn import linear_model

x= []; y= []
with open(sys.argv[1], "r") as IFILE: # start reading data 
    for line in IFILE:
        (t, x1, x2)= line.strip().split()
        x1= float(x1)
        x2= float(x2)
        x.append([x1, x2, x1*x2, x1*x1, x2*x2])
        y.append(int(t))
x= np.array(x)
y= np.array(y)

reg= linear_model.LogisticRegression()
#reg= linear_model.LogisticRegression(max_iter= 1e7, tol= 1e-10, C=1e40)
reg.fit(x, y)
print(reg.get_params())
(w1, w2, w12, w11, w22)= reg.coef_[0]

import matplotlib.pyplot as plt

A1 = np.linspace(0, 64, 128)
A2 = np.linspace(0, 64, 128)
(a1, a2)= np.meshgrid(A1, A2)
fa1= a1.flatten() 
fa2= a2.flatten() 
B= np.transpose( np.array([fa1, fa2, fa1*fa2, fa1*fa1, fa2*fa2]) )
pB= reg.predict_proba(B)
print(len(pB))
ppB= pB[:,0].reshape((128,128))
#f= 1/(np.exp(w1*a1 + w2*a2 + w12*a1*a2 + w11*a1*a1 + w22*a2*a2)+1)

plt.figure(figsize=(8,8))
plt.scatter(x[:,0], x[:,1], c=y, s=5)
plt.contour(a1, a2, ppB, [0.2, 0.5, 0.8], colors='k')
plt.show()
