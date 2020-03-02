import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pyrr import line, ray, geometric_tests, plane


class Cubes:
    def __init__(self, n):
        self.n = n
        self.cubes = np.zeros(shape=(n, n, n))
        self.xy_rays = []
        self.xz_rays = []
        self.yz_rays = []
        self.generate_xy_rays()
        self.generate_xz_rays()
        self.generate_yz_rays()

    def generate_xy_rays(self):
        # n+1 because 2 boxes |_|_| would have 3 xy rays on each level.
        for x in range(self.n + 1):
            self.xy_rays += [[]]
            for y in range(self.n + 1):
                self.xy_rays[x] += [
                    ray.create_from_line(
                        line.create_from_points((x, y, 0), (x, y, self.n))
                    )
                ]

    def generate_xz_rays(self):
        for x in range(self.n + 1):
            self.xz_rays += [[]]
            for z in range(self.n + 1):
                self.xz_rays[x] += [
                    ray.create_from_line(
                        line.create_from_points((x, 0, z), (x, self.n, z))
                    )
                ]

    def generate_yz_rays(self):
        for y in range(self.n + 1):
            self.yz_rays += [[]]
            for z in range(self.n + 1):
                self.yz_rays[y] += [
                    ray.create_from_line(
                        line.create_from_points((0, y, z), (self.n, y, z))
                    )
                ]

    def intersect_xy(self, p):
        """This method checks for intersections with the xy vectors. It marks the cubes as intersected and returns the number of intersections."""
        inters = 0
        for x in range(self.n):
            for y in range(self.n):
                if np.min(self.cubes[x, y, :]) == 0.0:
                    # If there are cubes along this area, we need to check the 4 rays on the face of the cube
                    # (x,y), (x+1,y), (x,y+1), (x+1, y+1)
                    print(geometric_tests.ray_intersect_plane(self.xy_rays[x][y], p))
                    print("Cubes left to intersect!")

        return inters

    def generate_grid(self):
        """Generates cubes to be drawn in pyplot"""
        cr = 0.5
        cube_verts = []  # List of all cube vert arrays
        cube_faces = []  # List of cube faces
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    cwo = np.array((x + cr, y + cr, z + cr))
                    cv = np.array(
                        [
                            cwo - cr,
                            cwo - np.array((-cr, cr, cr)),
                            cwo - np.array((-cr, -cr, cr)),
                            cwo - np.array((cr, -cr, cr)),
                            cwo - np.array((cr, cr, -cr)),
                            cwo - np.array((-cr, cr, -cr)),
                            cwo - np.array((-cr, -cr, -cr)),
                            cwo - np.array((cr, -cr, -cr,)),
                        ]
                    )
                    cube_verts += [cv]
                    cf = [
                        [cv[2], cv[3], cv[7], cv[6]],  # back
                        [cv[0], cv[1], cv[5], cv[4]],  # front
                        [cv[0], cv[1], cv[2], cv[3]],  # Bottom
                        [cv[4], cv[5], cv[6], cv[7]],  # top
                        [cv[4], cv[7], cv[3], cv[0]],  # left
                        [cv[1], cv[2], cv[6], cv[5]],  # right
                    ]
                    cube_faces += [cf]
        return (cube_verts, cube_faces)

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        r = [-self.n, self.n]
        X, Y = np.meshgrid(r, r)
        (grid, faces) = self.generate_grid()
        for f in faces:
            ax.add_collection3d(
                Poly3DCollection(
                    f, facecolors="cyan", linewidth=1, edgecolors="r", alpha=0.25
                )
            )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def intersect_plane(self, p):
        pass


if __name__ == "__main__":
    box_verts = np.array(
        [
            np.array([1.5, 2.3, 2.5]),  # top right
            np.array([1.5, 2.3, 1.5]),  # bottom right
            np.array([2.5, 1.3, 1.5]),  # bottom left
            np.array([2.5, 1.3, 2.5]),  # top left
        ]
    )
    bv = box_verts
    be = [
        [bv[0], bv[1]],  # Right
        [bv[1], bv[2]],  # bottom
        [bv[2], bv[3]],  # left
        [bv[3], bv[0]],  # top
    ]
    box_edges = be
    box_face = [[bv[0], bv[1], bv[2], bv[3]]]
    p = plane.create_from_points(box_verts[0], box_verts[1], box_verts[2])
    cubes = Cubes(2)
    # print(cubes.xy_rays)
    # print(cubes.cubes[0][0][1])
    # cubes.intersect_xy(p)
    # (grid, faces) = cubes.generate_grid()
    cubes.draw()
    # print(faces)
