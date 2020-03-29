import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, color='C0', dot_color='C1'):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color = color
        self.dot_color = dot_color
        plt.ion()
        plt.show()

    def plot_trajectory(self, xs_xyz):
        plt.cla()
        self.ax.plot(*zip(*xs_xyz), c=self.color)
        self.ax.scatter(*(xs_xyz[-1]), c=self.dot_color, s=10 ** 2)
        plt.draw()
        plt.pause(.001)
