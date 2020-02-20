import numpy as np


class Cubes:

    """A utility class for 3D Cube counting """

    def __init__(self, n):
        self.cubes = np.zeros(shape=(n, n, n))

    def get_cubes(self):
        return self.cubes

    def set_filled(self, pos):
        # Expecting pos to be in (x,y,z)
        self.cubes[pos[0]][pos[1]][pos[2]] = 1
