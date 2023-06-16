import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la



def StrongWolfe(E, gradient_E, xk, cables, bars, pk, k, c, mg, ghro, M, init_d, mu, a0 = 1, c1 = 0.01, c2 = 0.9, max_steps = 100, rho = 2.0):
    
    alpha_upper = a0 
    alpha_lower = 0.0 

    x_next = xk + alpha_upper * pk

    Ex = E(xk.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu) #Current energy 
    E_next = E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu) #Energy in next step

    gradE = gradient_E(xk.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu)
    gradE_next = gradient_E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu)
    ip = np.inner(pk, gradE_next)

    armijo = (E_next <= Ex + c1 * alpha_upper * init_d)
    curvatureLow = (ip >= c2 * init_d)
    curvatureHigh = (ip <= -c2 * init_d)

    steps_1 = 0
    while ((steps_1 < max_steps) and (armijo and (not curvatureLow))):

        alpha_lower = alpha_upper
        alpha_upper = alpha_upper * rho 
            
        x_next = xk + alpha_upper * pk
        E_next = E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu).flatten()
        gradE_next = gradient_E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu)
        ip = np.inner(pk, gradE_next) 

        armijo = (E_next <= Ex + c1 * alpha_upper * init_d)
        curvatureLow = (ip >= c2 * init_d)
        curvatureHigh = (ip <= -c2 * init_d)

        steps_1 +=1

    alpha = alpha_upper
    steps_2 = 0
    while not (armijo and curvatureLow and curvatureHigh) and (steps_2 < max_steps):
        if armijo and (not curvatureLow):
            alpha_lower = alpha
        else:
            alpha_upper = alpha

        alpha = (alpha_upper+alpha_lower)/2

        x_next = xk + alpha * pk
        E_next = E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu)
        gradE_next = gradient_E(x_next.reshape(-1,3), cables, bars, k, c, mg, ghro, M, mu)
        ip = np.inner(pk, gradE_next)

        armijo = (E_next <= Ex + c1 * alpha * init_d)
        curvatureLow = (ip >= c2*init_d)
        curvatureHigh = (ip <= -c2*init_d)
        
        steps_2 +=1     
    return x_next, gradE_next


def BFGS_method(E, gradient_E, x_init, cables, bars, k, c, mg, ghro, M, mu = 0, conv_tol = 1e-12, max_iter = 1000):  
    
    N = np.size(x_init.flatten())
    Hk = np.identity(N)
    
    gradE = gradient_E(x_init, cables, bars, k, c, mg, ghro, M,mu)

    xk = x_init.flatten()
    
    n = 0
    
    grad_norm = la.norm(gradE)

    x_array = np.zeros((max_iter, N)) #for convergance plots
    grad_array = np.zeros(max_iter)
    while (grad_norm > conv_tol) and (n < max_iter):
        #search direction 
        pk = - np.dot(Hk, gradE)
        
        init_d = np.inner(pk, gradE)

        #next step 
        x_next, gradE_next = StrongWolfe(E, gradient_E, xk, cables, bars, pk, k, c, mg, ghro, M, init_d, mu) 

        sk = x_next - xk 
        yk = gradE_next - gradE

        denominator = np.dot(yk, sk)            

        rho_k = 1 / (denominator + 1e-26)
      
        #for next iteration
        Hk = (np.identity(N) - rho_k * np.outer(sk, yk)) @ Hk @ (np.identity(N) - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)
        
        gradE = gradE_next 
        xk = x_next
        grad_norm = la.norm(gradE)
    
        x_array[n] = x_next
        grad_array[n] = grad_norm
        n += 1  
        
    if n < max_iter:
        print("Converged after {} iterations.".format(n))
    else:
        print("Did not converge after {} iterations.".format(max_iter))
    return xk.reshape(-1,3), gradE, x_array[:n], grad_array[:n]