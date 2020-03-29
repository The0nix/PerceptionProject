import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, real=None, color='C0', dot_color='C1'):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.real = real
        self.color = color
        self.dot_color = dot_color
        plt.ion()
        plt.show()

    def plot_trajectory(self, xs_xyz):
        plt.cla()
        if self.real is not None:
            self.ax.plot(*zip(*self.real), c='green')
        self.ax.plot(*zip(*xs_xyz), c=self.color)
        self.ax.scatter(*(xs_xyz[-1]), c=self.dot_color, s=10 ** 2)
        self.ax.scatter(0, 0, 0, marker='x', s=10 ** 2, c='red')
        plt.draw()
        plt.pause(.001)


def angles_from_C(C):
    phi = np.arctan2(C[1, 0], C[0, 0])
    theta = np.arctan2(-C[2, 0], np.sqrt(C[2, 1] ** C[2, 2] ** 2))
    psi = np.arctan2(C[2, 1], C[2, 2])
    return phi, theta, psi
