import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

def plot_result(X, cables, bars, M, rotate):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cables
    for start_index, end_index, _ in cables:
        start_point = X[start_index]
        end_point = X[end_index]

        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                zs = [start_point[2], end_point[2]], linestyle = '--', c='black')

    # Plot bars
    for start_index, end_index, _ in bars:
        start_point = X[start_index]
        end_point = X[end_index]
    
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                zs = [start_point[2], end_point[2]], c='black')

    # Plot points
    for i in range(len(X)):
        x, y, z = X[i]

        if i < M:
            ax.scatter(x, y, z, c='black', s=80, marker='o')
            ax.text(x+0.1, y+0.1, z+0.01, str(i+1), fontsize=12)
        else:
            ax.scatter(x, y, z, c='r', s=80, marker='o')
            ax.text(x+0.1, y+0.1, z+0.01, str(i+1), fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set angle of view
    if rotate is not None:
        azim, elev = rotate
        ax.view_init(elev=elev, azim=azim)

    plt.show()
    

def plot_convergence(X_array, analytic_array, grad):
    N_, M_ = X_array.shape
    convergence = np.zeros(N_)
    for i in range(N_):
        convergence[i] = la.norm(X_array[i].flatten()-analytic_array.flatten())
    
    x = np.arange(0, N_, 1)
    
    # plot convergence on a logarithmic y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(x, convergence, label='Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error')
    ax1.set_yscale('log')  # sets the y-axis to a logarithmic scale
    ax1.set_title('Error and Gradient in Each Step')
    ax1.grid(True)
    ax1.legend()
    
    # create a second y-axis for grad data
    ax1.plot(x, grad, color='tab:orange', label='Gradient')
    plt.legend()
    
    plt.show()

def plot_convergence_P12(grad):
    x = np.arange(len(grad))
    plt.plot(x, grad, color='tab:orange', label='Gradient')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient')
    plt.yscale('log')  # sets the y-axis to a logarithmic scale
    plt.title('Gradient in Each Step')
    plt.grid(True)
    plt.legend()
    plt.show()
