import numpy as np
import matplotlib.pyplot as plt


def zprojection(image, centers=None, zlim=None, clim=None):
    if zlim is None:
        zlim = [0, image.shape[0]-1]
    if len(zlim) != 2:
        raise ValueError("Need start and stop zlim")
    projection = image[zlim[0]:zlim[1]].max(axis=0)
    plt.imshow(projection, clim=clim)
    if centers is not None:
        idx = np.where(np.logical_and(centers[:, 0] >= zlim[0], centers[:, 0] < zlim[1]))[0]
        points = centers[idx]
        plt.plot(points[:, 2], points[:, 1], 'r*')
    plt.show()
