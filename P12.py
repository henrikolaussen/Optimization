import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

#P12
N = 8
M = 0 #all nodes are free
c = 200
k = 10
grho = 1e-10
mg = 1e-5
mg_vec = mg* np.ones(N)

cables = np.array([[0,1,2],[1,2,2],[2,3,2],[0,3,2], [4,5,1],[5,6,1],[6,7,1],[4,7,1],[0,7,8],[1,4,8],[2,5,8],[3,6,8]]) #[i, j, lij]
bars = np.array([[0,4,10], [1,5,10], [2,6,10], [3,7,10]])
X_initial = np.zeros((N,3))

X_initial[0,:] = np.array([1,1,0])          #x0
X_initial[1,:] = np.array([-1,1,0])         #x0
X_initial[2,:] = np.array([-1,-1,0])        #x0
X_initial[3,:] = np.array([2,-1,0])         #x0
X_initial[4,:] = np.array([-0.3,0,10])      #x0
X_initial[5,:] = np.array([0.3,-0.3,10])    #x0
X_initial[6,:] = np.array([4,1,10])       #x0
X_initial[7,:] = np.array([1,0.3,10])     #x0