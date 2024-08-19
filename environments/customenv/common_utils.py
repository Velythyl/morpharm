import numpy as np

def random_sphere_numpy(min_r, max_r, shape=None):
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume#

    phi = np.random.uniform(low=0, high=2 * np.pi, size=shape)
    costheta = np.random.uniform(low=-1, high=1, size=shape)
    u = np.random.uniform(low=0, high=1, size=shape)

    theta = np.arccos(costheta)
    r = np.cbrt((u * max_r ** 3) + ((1 - u) * min_r ** 3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=1)
