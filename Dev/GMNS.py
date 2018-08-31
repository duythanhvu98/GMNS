import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import orth


#General Minimum Noise Subspace
def GMNS_PSA(X,p):
    n = X.shape[0]; T = X.shape[1];
    
    #The exact subspace
    W_exact = np.matrix(orth(X))
    #print("W_exact: " +str(W_exact.shape))
    
    
    # 2 Subsystem
    k = 2;
    X1 = X[:int(n/k),:]; X2 = X[int(n/k):,:];    
    
    
    #W1 
    R1 = 1/T*X1*X1.T
    lamda1, W1 = LA.eig(R1)
    W1 = W1[:,:p]
    S1 = W1.T*X1;
    T1 = S1*S.T
    #print(W1.shape)



    #W2
    R2 = 1/T*X2*X2.T
    lamda2, W2= LA.eig(R2)
    W2 = W2[:,:p]
    S2 = W2.T*X2;
    T2 = S2*S.T


    #W
    W = np.matrix(np.concatenate(((W1*T1).T,(W2*T2).T), axis = 1).T)
    #print("W_es: " +str(W.shape))


    #Compare
    TS = np.trace(W.T * (np.eye(n) - W_exact *W_exact.T) * W)
    MS = np.trace(W.T * (W_exact * W_exact.T) * W)
    SEP = TS/MS
    print("SEP: "+ str(SEP))


    #PCA 
    Rxx = X*X.T
    lamda, Us = LA.eig(W.T*Rxx*W)
    print("lamda estimate:"+ str(lamda))
    return W




################################################Test 
n=100;p=3;k = 2; d = int(n/k);T = 1000;
#Initial X
A = np.matrix(np.random.rand(n,p)); S = np.matrix(np.random.rand(p,T))
N = np.random.rand(n,T);
X = A*S + N;

W = GMNS_PSA(X,p)
