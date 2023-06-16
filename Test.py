import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

N = 8
M = 4
c = 1
k = 0.1
grho = 10**(-17)
mg = 1/1000
mg_vec =  mg * np.ones(N)

cables = np.array([[0,4,20],[1,5,20],[2,6,20],[3,7,20], [4,5,20],[5,6,20],[6,7,20],[4,7,20]]) #[i, j, lij]

bars = np.array([])
X_initial = np.zeros((N,3))

X_initial[0,:] = np.array([1,1,20.1]) #p1
X_initial[1,:] = np.array([1,-1,20.1]) #p2
X_initial[2,:] = np.array([-1,-1,20.1]) #p3
X_initial[3,:] = np.array([-1,1,20.1]) #p
X_initial[4,:] = np.array([5,3,30]) #x0
X_initial[5,:] = np.array([2,2,30]) #x0
X_initial[6,:] = np.array([2,1,30]) #x0
X_initial[7,:] = np.array([-1,2,30]) #x0