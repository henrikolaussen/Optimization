import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

N = 8 #number of nodes in structure 
M = 4 #number of fixed nodes

k = 3 #material parameter > 0
mg = 1/6
ghro = 0
mg_vec = mg * np.ones(N)
l = 3 #resting lengths 

X_initial = np.zeros((N, 3))
X_initial[0,:] = np.array([5,5,0]) #p1
X_initial[1,:] = np.array([-5,5,0]) #p2
X_initial[2,:] = np.array([-5,-5,0]) #p3
X_initial[3,:] = np.array([5,-5,0]) #
X_initial[4,:] = np.array([1,4,0]) #x0
X_initial[5,:] = np.array([-3,1,0]) #x0
X_initial[6,:] = np.array([-6,-1,0]) #x0
X_initial[7,:] = np.array([1,2,0]) #x0

cables = np.array([[0,4,l], [1,5,l],[2,6,l],[3,7,l],[4,5,l],[4,7,l], [5,6,l], [6,7, l]])
bars = np.array([])

analytic = np.array([[5,5,0],[-5,5,0],[-5,-5,0],[5,-5,0],[2,2,-3/2],[-2,2,-3/2],[-2,-2,-3/2],[2,-2,-3/2]])
