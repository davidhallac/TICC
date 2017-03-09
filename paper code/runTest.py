import numpy as np
import numpy.linalg as alg
import scipy as spy
from solveCrossTime import *
import math



def upper2Full(a, eps = 0):
    ind = (a<eps)&(a>-eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))             
    return A   



numBlocks = 3
sizeBlocks = 2
probSize = numBlocks*sizeBlocks
lamb = 0.1*np.zeros((probSize,probSize))
S = 0





np.random.seed(0)

S_inv =  np.matrix('1 0.5 0 0.4 0.5 0; 0.5 1 0.2 0 0 0.8; 0 0.2 1 0.5 0 0.4; 0.4 0 0.5 1 0.2 0; 0.5 0 0 0.2 1 0.5; 0 0.8 0.4 0 0.5 1')
S_inv = S_inv + np.abs(min(np.linalg.eig(S_inv)[0]) + 0.4849 )*np.eye(probSize)


S = np.linalg.inv(S_inv)





print S, "= S"


#COPY THIS CODE
gvx = TGraphVX()
theta = semidefinite(probSize,name='theta')
obj = -log_det(theta) + trace(S*theta)
gvx.AddNode(0, obj)
gvx.AddNode(1)
dummy = Variable(1)
gvx.AddEdge(0,1, Objective = lamb*dummy + numBlocks*dummy + sizeBlocks*dummy)
gvx.Solve(Verbose=True, MaxIters=1000, Rho = 1, EpsAbs = 1e-6, EpsRel = 1e-6)

#THIS IS THE SOLUTION
val = gvx.GetNodeValue(0,'theta')
S_est = upper2Full(val, 1e-5)

np.set_printoptions(precision=4, suppress=True)
print np.linalg.inv(S)

print S_est
