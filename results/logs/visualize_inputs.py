import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
import torch


def scatter_plot(x):
    fgr = plt.figure()
    ax = fgr.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="r", marker="o")
    plt.show()


def visualize(x):
    data = x.detach().numpy()

    if data.ndim == 2 and data.shape[1] == 3:
        points = data.shape[0]
        grid = int(np.sqrt(points))

        X = data[:, 0].reshape(grid, grid)
        Y = data[:, 1].reshape(grid, grid)
        Z = data[:, 2].reshape(grid, grid)

        fgr = plt.figure()
        ax = fgr.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z)

        ax.set_title("3D VISUALIZATION")

        plt.show()


x = torch.rand(100, 3)
visualize(x)
