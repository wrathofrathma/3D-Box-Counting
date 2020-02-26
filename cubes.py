import numpy as np
from pyrr import line, ray


class Cubes:
    def __init__(self, n):
        self.n = n
        self.xy_rays = []
        self.xz_rays = []
        self.yz_rays = []
        self.generate_xy_rays()
        self.generate_xz_rays()
        self.generate_yz_rays()

    def generate_xy_rays(self):
        for x in range(self.n):
            self.xy_rays += [[]]
            for y in range(self.n):
                self.xy_rays[x] += [
                    ray.create_from_line(
                        line.create_from_points((x, y, 0), (x, y, self.n))
                    )
                ]

    def generate_xz_rays(self):
        for x in range(self.n):
            self.xz_rays += [[]]
            for z in range(self.n):
                self.xz_rays[x] += [
                    ray.create_from_line(
                        line.create_from_points((x, 0, z), (x, self.n, z))
                    )
                ]

    def generate_yz_rays(self):
        for y in range(self.n):
            self.yz_rays += [[]]
            for z in range(self.n):
                self.yz_rays[y] += [
                    ray.create_from_line(
                        line.create_from_points((0, y, z), (self.n, y, z))
                    )
                ]

    def intersect_plane(plane):
        pass


if __name__ == "__main__":
    cubes = Cubes(3)
