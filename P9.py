import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la


N = 8
M = 4
c = 1
k = 0.1
grho = 0
mg = 0
mg_vec = mg * np.ones(N)

cables = np.array([[4,5,1],[5,6,1],[6,7,1],[4,7,1], [0,7,8],[1,4,8],[2,5,8],[3,6,8]]) #[i, j, lij]
bars = np.array([[0,4,10], [1,5,10], [2,6,10], [3,7,10]])
X_initial = np.zeros((N,3))

X_initial[0,:] = np.array([1,1,0]) #p1
X_initial[1,:] = np.array([-1,1,0]) #p2
X_initial[2,:] = np.array([-1,-1,0]) #p3
X_initial[3,:] = np.array([1,-1,0]) #4
X_initial[4,:] = np.array([0.2,0.2,1]) #x0
X_initial[5,:] = np.array([-0.2,0.2,1]) #x0
X_initial[6,:] = np.array([-0.2,-0.2,1]) #x0
X_initial[7,:] = np.array([0.2,-0.2,0]) #x0

s = 0.70970
t = 9.54287

analytic = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[-s,0,t],[0,-s,t],[s,0,t],[0,s,t]])