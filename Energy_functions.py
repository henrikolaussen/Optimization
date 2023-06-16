import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

import BFGS

def E_cab_elast(xi, xj, lij, k):
    #xi, xj: 3-dim vectors
    L_ij = la.norm(xi-xj)
    if L_ij > lij:
        return k/(2*lij**2) * (L_ij-lij)**2
    return 0

def E_ext(X, mg, M):
    return np.sum(mg[M:]*X[M:,2])

def E_bar_elast(xi, xj, lij, c):
    return c/(2*lij**2) * (la.norm(xi-xj)-lij)**2

def E_bar_grav(xi, xj, lij, grho):
    return grho*lij/2 * (xi[2]+xj[2])


def E(X, cables, bars, k, c, mg, grho, M, mu = 0):
    E_cab_sum = 0
    for i, j , lij in cables: 
        E_cab_sum += E_cab_elast(X[i,:], X[j,:], lij, k)
            

    E_bar_sum = 0
    for i, j, lij in bars:
        E_bar_sum += E_bar_elast(X[i,:], X[j,:], lij, c) + E_bar_grav(X[i,:], X[j,:], lij, grho)
    return E_cab_sum + E_bar_sum + E_ext(X, mg, M)

def E_penalty(X, cables, bars, k, c, mg, grho, M, mu):
    X = X.reshape(-1,3)
    pen_z = 0.5 * mu * np.sum(np.maximum(0, -X[:,2])**2)
    pen_xy = 10e-3* 0.5  * ((X[0,0] - 1)**2 + (X[0,1] - 1)**2)
    return E(X, cables, bars, k,c, mg, grho, M) + pen_z + pen_xy

def gradient_E_penalty(X, cables, bars, k, c, mg, grho, M, mu):
    X = X.reshape(-1,3)
    pen = np.zeros_like(X)
    pen[:,2] -= mu * np.maximum(0, -X[:,2])
    pen[0,0] += 1e-2 * (X[0,0] - 1)
    pen[0,1] += 1e-2 * (X[0,1] - 1)
    return gradient_E(X, cables, bars, k, c, mg, grho, M) + pen.flatten()

def Quadratic_Penalty_Method(x0, mu0, cables, bars, k, c, mg_vec, grho, M):
    muk = mu0
    xk = x0
    tau = 0.01
    for i in range(8): 
        
        xk, gradient_E, x_arr, grad_arr = BFGS.BFGS_method(E_penalty, gradient_E_penalty, xk, cables, bars, k, c, mg_vec, grho, M, muk, conv_tol = tau)

        print('mu_k:', muk, 'tau_k:', tau)
        print(' ')

        muk = muk * 2
        tau = tau/10
      
    return xk, gradient_E, x_arr, grad_arr

def gradient_E(X, cables, bars, k, c, mg, grho, M, mu = 0):
    gradient = np.zeros_like(X)
  
    for i, j, lij in cables:
        norm = la.norm(X[i]-X[j])
        temp1 = 0
        if norm - lij > 0:
            temp1 = k/lij**2 * (1-lij/norm)
        
        gradient[i] += temp1 * (X[i]-X[j])
        gradient[j] += temp1 * (X[j]-X[i])

  
    for i, j, lij in bars:
        norm = la.norm(X[i]-X[j])
        temp2 = c/lij**2 * (1-lij/norm)

        gradient[i] += temp2 * (X[i]-X[j]) + np.array([0,0,grho*lij/2])
        gradient[j] += temp2 * (X[j]-X[i]) + np.array([0,0,grho*lij/2])

    
    for i in range(M, len(mg)):
        gradient[i, 2] += mg[i]

    
    gradient[0:M] = 0 

    return gradient.flatten()


