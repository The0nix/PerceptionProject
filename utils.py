import contextlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, color='C0', dot_color='C1'):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color = color
        self.dot_color = dot_color
        plt.ion()
        plt.show()

    def plot_trajectory(self, xs_xyz, C):
        plt.cla()
        self.ax.plot(*zip(*xs_xyz), c=self.color)

        self.ax.plot(*zip(xs_xyz[-1], xs_xyz[-1] + C @ np.array([0, 0, -1]) * 0.05), c='red')
        self.ax.scatter(*(xs_xyz[-1]), c=self.dot_color, s=10 ** 2)
        self.ax.scatter(0, 0, 0, marker='x', s=10 ** 2, c='red')
        plt.draw()
        plt.pause(.001)


def angles_from_C(C):
    psi = np.arctan2(C[1, 0], C[0, 0])
    theta = np.arctan2(-C[2, 0], np.sqrt(C[2, 1] ** 2 + C[2, 2] ** 2))
    phi = np.arctan2(C[2, 1], C[2, 2])
    return phi, theta, psi


def wrap_angle(angle, r1, r2, wrapper=2 * np.pi):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    while angle < r1:
        angle += wrapper

    while angle >= r2:
        angle -= wrapper

    return angle
