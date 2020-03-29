import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_config(config_path):
    with open(config_path, 'r') as stream:
        return yaml.safe_load(stream)

class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    def plot_trajectory(self, xs_xyz):
        self.ax.plot(*zip(*xs_xyz))
        plt.draw()
        plt.pause(.001)
