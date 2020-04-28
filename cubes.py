import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pyrr import line, ray, geometric_tests, plane
from math import radians
from multiprocessing import Process, Lock, cpu_count, Array, Value, Queue, connection, Pool
import math
import ctypes as c


class Cubes:
    """This is a utility class to create a grid of cubes around a 3D mesh for the purposes of calculating the fractal dimension.
    Attributes:
        n (int): number of sides on each face.
        r (float): Controls the radius of the cubes. Don't touch this.
        cube_array (array): Shared memory array for multiprocessing.
        cubes (array): Array of cubes tracking whether we've intersected them.
        xy_rays (array): Array of vectors used for raycasting the mesh vertices into faces of the cubes.
        xz_rays (array): Array of vectors used for raycasting the mesh vertices into faces of the cubes.
        yz_rays (array): Array of vectors used for raycasting the mesh vertices into faces of the cubes.
        max_t (int): Maximum number of supported threads on the system.
        available_threads: Current number of available threads.
    """
    def __init__(self, n):
        """Constructor for the Cubes class.
        Parameters:
            n (int): Number of faces on each side of the cubic grid.
        """
        self.n = n
        self.r = 0.5
        self.cube_array = Array(c.c_double, n * n * n)
        self.cubes = np.frombuffer(self.cube_array.get_obj())
        self.cubes.fill(0)
        self.cubes = self.cubes.reshape((n, n, n))
        self.xy_rays = []
        self.xz_rays = []
        self.yz_rays = []
        # self.max_t = cpu_count()  # total threads
        self.max_t = 24
        self.available_threads = Value('i', self.max_t)  # current thread count
        print("Detected " + str(self.max_t) + " threads")
        print("Generating %d rays on %d threads." % (3 * (n + 1) ** 2, self.max_t))
        self.generate_xy_rays()
        self.generate_xz_rays()
        self.generate_yz_rays()

    def process_queue(self, queue, rays):
        while(not queue.empty()):
            val = queue.get()
            rays[val[0]] += val[1]


    def sub_generate_xz_rays(self, x, queue):
        """Subprocess for generating xz rays."""
        rays = []
        for z in range(self.n + 1):
            rays += [
                ray.create_from_line(
                    line.create_from_points((x, 0, z), (x, self.n, z))
                )
            ]
        queue.put((x, rays))
        queue.close()

    def sub_generate_xy_rays(self, x, queue):
        """Subprocess for generating xy rays."""
        rays = []
        for y in range(self.n + 1):
            rays += [
                ray.create_from_line(
                    line.create_from_points((x, y, 0), (x, y, self.n))
                )
            ]
        queue.put((x, rays))
        queue.close()

    def sub_generate_yz_rays(self, y, queue):
        """Subprocess for generating yz rays."""
        rays = []
        for z in range(self.n + 1):
            rays += [
                ray.create_from_line(
                    line.create_from_points((0, y, z), (self.n, y, z))
                )
            ]
        queue.put((y, rays))
        queue.close()

    def generate_xy_rays(self):
        """Populates the xy_rays array with raycasting vectors on the xy plane from 0 to n"""
        print("Generating xy")
        queue = Queue()
        processes = []
        for x in range(self.n + 1):
            self.xy_rays += [[]]
            if(self.available_threads.value >= 1):
                processes.append(
                    Process(target=self.sub_generate_xy_rays, args=(x, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1
            else:
                while True:
                    self.process_queue(queue, self.xy_rays)
                    if (processes[0].is_alive() is not True):
                        processes[0].join(timeout=1)
                        break
                processes.pop(0)
                self.available_threads.value += 1
                processes.append(
                    Process(target=self.sub_generate_xy_rays, args=(x, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1

        for p in processes:
            while True:
                self.process_queue(queue, self.xy_rays)
                if (p.is_alive() is not True):
                    p.join(timeout=1)
                    break
            self.available_threads.value += 1
        self.process_queue(queue, self.xy_rays)

    def generate_xz_rays(self):
        """Populates the xz_rays array with raycasting vectors on the xz plane from 0 to n"""
        print("Generating xz")
        queue = Queue()
        processes = []
        for x in range(self.n + 1):
            self.xz_rays += [[]]
            if(self.available_threads.value >= 1):
                processes.append(
                    Process(target=self.sub_generate_xz_rays, args=(x, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1
            else:
                while True:
                    self.process_queue(queue, self.xz_rays)
                    if (processes[0].is_alive() is not True):
                        processes[0].join(timeout=1)
                        break
                processes.pop(0)
                self.available_threads.value += 1
                processes.append(
                    Process(target=self.sub_generate_xz_rays, args=(x, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1
        for p in processes:
            while True:
                self.process_queue(queue, self.xz_rays)
                if (p.is_alive() is not True):
                    p.join(timeout=1)
                    break
            self.available_threads.value += 1
        self.process_queue(queue, self.xz_rays)


    def generate_yz_rays(self):
        """Populates the yz_rays array with raycasting vectors on the yz plane from 0 to n"""
        print("Generating yz")
        queue = Queue()
        processes = []
        for y in range(self.n + 1):
            self.yz_rays += [[]]
            if(self.available_threads.value >= 1):
                processes.append(
                    Process(target=self.sub_generate_yz_rays, args=(y, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1
            else:
                while True:
                    self.process_queue(queue, self.yz_rays)
                    if (processes[0].is_alive() is not True):
                        processes[0].join(timeout=1)
                        break
                processes.pop(0)
                self.available_threads.value += 1
                processes.append(
                    Process(target=self.sub_generate_yz_rays, args=(y, queue))
                    )
                processes[-1].start()
                self.available_threads.value -= 1

        for p in processes:
            while True:
                self.process_queue(queue, self.yz_rays)
                if (p.is_alive() is not True):
                    p.join(timeout=1)
                    break
            p.join()
            self.available_threads.value += 1
        self.process_queue(queue, self.yz_rays)

    def check_bounds(self, verts, isec):
        """Validates the bounds of the intersection with the min/max values of the min/max values of the mesh."""
        min_v = verts[0]
        max_v = verts[1]
        isec = np.array(isec)
        x = (isec[0] >= min_v[0]) & (isec[0] <= max_v[0])
        y = (isec[1] >= min_v[1]) & (isec[1] <= max_v[1])
        z = (isec[2] >= min_v[2]) & (isec[2] <= max_v[2])
        return x & y & z

    def check_intersection_xy(self, isec, xy, verts):
        """Checks for intersections of a specifiy ray on the xy plane for the mesh"""
        if isec is not None:
            x = xy[0]
            y = xy[1]
            for z in range(self.n):
                if self.cubes[x][y][z] == 0 and isec[2] >= z and isec[2] <= z + 1:
                    if self.check_bounds(verts, isec):
                        self.cubes[x][y][z] = 1
                        return 1
        return 0

    def check_intersection_xz(self, isec, xz, verts):
        """Checks for intersections of a specifiy ray on the xz plane for the mesh"""
        if isec is not None:
            x = xz[0]
            z = xz[1]
            for y in range(self.n):
                if self.cubes[x][y][z] == 0 and isec[1] >= y and isec[1] <= y + 1:
                    if self.check_bounds(verts, isec):
                        self.cubes[x][y][z] = 1
                        return 1
        return 0

    def check_intersection_yz(self, isec, yz, verts):
        """Checks for intersections of a specifiy ray on the yz plane for the mesh"""
        if isec is not None:
            y = yz[0]
            z = yz[1]
            for x in range(self.n):
                if self.cubes[x][y][z] == 0 and isec[0] >= x and isec[0] <= x + 1:
                    if self.check_bounds(verts, isec):
                        self.cubes[x][y][z] = 1
                        return 1
        return 0

    def sub_intersect_xz(self, x, p, min_verts, max_verts):
        """Subprocess for calculating intersections on the xz plane."""
        i = 0
        for z in range(self.n):
            if np.min(self.cubes[x, :, z]) == 0.0:
                bl = geometric_tests.ray_intersect_plane(self.xz_rays[x][z], p)
                br = geometric_tests.ray_intersect_plane(self.xz_rays[x + 1][z], p)
                tl = geometric_tests.ray_intersect_plane(self.xz_rays[x][z + 1], p)
                tr = geometric_tests.ray_intersect_plane(
                    self.xz_rays[x + 1][z + 1], p
                )
                i += self.check_intersection_xz(
                    bl, (x, z), (min_verts, max_verts)
                )
                i += self.check_intersection_xz(
                    br, (x, z), (min_verts, max_verts)
                )
                i += self.check_intersection_xz(
                    tl, (x, z), (min_verts, max_verts)
                )
                i += self.check_intersection_xz(
                    tr, (x, z), (min_verts, max_verts)
                )
        return i

    def intersect_xz(self, verts):
        """This method checks for intersections with the xz vectors. It marks the cubes as intersected and returns the number of intersections."""
        p = plane.create_from_points(verts[0], verts[1], verts[2])
        max_verts = (np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2]))
        min_verts = (np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2]))
        inters = 0
        for x in range(self.n):
          inters += self.sub_intersect_xz(x, p, min_verts, max_verts)
        return inters

    def sub_intersect_xy(self, x, p, min_verts, max_verts):
        """Subprocess for calculating intersections on the xy plane."""
        i = 0
        for y in range(self.n):
            if np.min(self.cubes[x, y, :]) == 0.0:
                bl = geometric_tests.ray_intersect_plane(self.xy_rays[x][y], p)
                br = geometric_tests.ray_intersect_plane(self.xy_rays[x + 1][y], p)
                tl = geometric_tests.ray_intersect_plane(self.xy_rays[x][y + 1], p)
                tr = geometric_tests.ray_intersect_plane(
                    self.xy_rays[x + 1][y + 1], p
                )
                i += self.check_intersection_xy(
                    bl, (x, y), (min_verts, max_verts)
                )
                i += self.check_intersection_xy(
                    br, (x, y), (min_verts, max_verts)
                )
                i += self.check_intersection_xy(
                    tl, (x, y), (min_verts, max_verts)
                )
                i += self.check_intersection_xy(
                    tr, (x, y), (min_verts, max_verts)
                )
        return i


    def intersect_xy(self, verts):
        """This method checks for intersections with the xz vectors. It marks the cubes as intersected and returns the number of intersections."""
        p = plane.create_from_points(verts[0], verts[1], verts[2])
        max_verts = (np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2]))
        min_verts = (np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2]))
        inters = 0
        for x in range(self.n):
          inters += self.sub_intersect_xy(x, p, min_verts, max_verts)
        return inters

    def sub_intersect_yz(self, y, p, min_verts, max_verts):
        i = 0
        for z in range(self.n):
            if np.min(self.cubes[:, y, z]) == 0.0:
                bl = geometric_tests.ray_intersect_plane(self.yz_rays[y][z], p)
                br = geometric_tests.ray_intersect_plane(self.yz_rays[y + 1][z], p)
                tl = geometric_tests.ray_intersect_plane(self.yz_rays[y][z + 1], p)
                tr = geometric_tests.ray_intersect_plane(
                    self.yz_rays[y + 1][z + 1], p
                )
                i += self.check_intersection_yz(
                    bl, (y, z), (min_verts, max_verts)
                )
                i += self.check_intersection_yz(
                    br, (y, z), (min_verts, max_verts)
                )
                i += self.check_intersection_yz(
                    tl, (y, z), (min_verts, max_verts)
                )
                i += self.check_intersection_yz(
                    tr, (y, z), (min_verts, max_verts)
                )
        return i

    def intersect_yz(self, verts):
        """This method checks for intersections with the yz vectors. It marks the cubes as intersected and returns the number of intersections."""
        p = plane.create_from_points(verts[0], verts[1], verts[2])
        max_verts = (np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2]))
        min_verts = (np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2]))
        inters = 0
        for y in range(self.n):
          inters += self.sub_intersect_yz(y, p, min_verts, max_verts)
        return inters

    def generate_grid(self):
        """Generates cubes to be drawn in pyplot"""
        cr = self.r
        cube_verts = []  # List of all cube vert arrays
        cube_faces = []  # List of cube faces
        colors = []
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
                    if self.cubes[x][y][z] == 0.0:
                        colors += ["cyan"]
                    else:
                        colors += ["r"]

        return (cube_verts, cube_faces, colors)

    def draw(self, mesh_verts, mesh_faces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        r = [-self.n, self.n]
        X, Y = np.meshgrid(r, r)
        (grid, faces, colors) = self.generate_grid()
        for f, c in zip(faces, colors):
            if c == "r":
                alpha = 0.25
                ax.add_collection3d(
                    Poly3DCollection(
                        f, facecolors=c, linewidth=1, edgecolors="r", alpha=alpha
                    )
                )
        for face in mesh_faces:
            f = (
                np.array(
                    [mesh_verts[face[0]], mesh_verts[face[1]], mesh_verts[face[2]]]
                ),
            )
            ax.add_collection3d(
                Poly3DCollection(
                    verts=f,
                    zsort="average",
                    facecolors="cyan",
                    linewidth=0.5,
                    edgecolors="r",
                    alpha=0.1,
                )
            )

        ax.set_xlim3d(0, n + 1)
        ax.set_ylim3d(0, n + 1)
        ax.set_zlim3d(0, n + 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def intersect_mesh(self, mesh):
        print("Generating intersections")
        verts = mesh.get_vertices()
        faces = mesh.get_faces()
        mutex = Lock()
        # Face based partitioning
        # We'll apply a face based partitioning scheme and add in logic for if (n_faces < max_threads), then we'll let faces generate more to fill it out.
        
        # First, we calculate our partition_size based on the n_max_threads & n_faces
        n_faces = len(faces)
        n_max_threads = self.max_t  # Just for visual clarity we'll rename it locally.
        part_size = math.floor(n_faces / n_max_threads)
        part_size = max(part_size, 1) # if n_max_threads > n_faces, part_size would be < 1.

        # Create partitions
        partitions = [faces[i: i + part_size] for i in range(0, n_faces, part_size)]
        # Overflow partition calc
        if(len(partitions) > n_max_threads):
          new_parts = partitions[:n_max_threads]
          # Stick overflow on the last index. We could spread it, but fuck it.
          if(len(partitions[n_max_threads:]) > 1):
            # Let's just nest this list comprehension and pretend it's not a double nested for loop
            overflow = np.array([p for p in [ z for z in partitions[n_max_threads:]]])
            overflow = overflow.reshape(overflow.shape[0]*overflow.shape[1],overflow.shape[2])
          else:
            overflow = partitions[-1]

          new_parts[-1] = np.append(new_parts[-1], overflow,axis=0)
          partitions = new_parts
        print("Created %d partitions of %d avg size" % (len(partitions), part_size))

        icount = Value('d', 0)
        # Create processes

        processes = []
        for partition in partitions:
          processes.append(Process(target=self.intersect_partition, args=(verts, partition, icount, mutex)))
          processes[-1].start()
        # Wait for all
        for p in processes:
          p.join()
        # connection.wait(p.sentinel for p in processes)
        return icount.value

    def intersect_partition(self, verts, part, count, mutex):
      """Threaded portion of the mesh intersection"""
      icount = 0
      for face in part:
        v = np.array([verts[face[0]], verts[face[1]], verts[face[2]]])
        icount += self.intersect_face(v)
      mutex.acquire()
      count.value += icount
      mutex.release()

    def intersect_face(self, verts):
        icount = 0
        icount = self.intersect_xy(verts)
        icount += self.intersect_xz(verts)
        icount += self.intersect_yz(verts)
        return icount

    def get_grid_scale(self):
        return float(self.n)

    # def center_mesh(self, mesh):
    #     verts = mesh.get_vertices()
    #     xmax = np.max(verts[:, 0])
    #     ymax = np.max(verts[:, 1])
    #     zmax = np.max(verts[:, 2])
    #     xmin = np.min(verts[:, 0])
    #     ymin = np.min(verts[:, 1])
    #     zmin = np.min(verts[:, 2])
    #     xdiff = abs(xmax - xmin)
    #     ydiff = abs(ymax - ymin)
    #     zdiff = abs(zmax - zmin)
    #     scale = max(xdiff, ydiff, zdiff)
    #     scale = self.n / scale
    #     scale = np.array([scale, scale, scale])
    #     mesh.set_scale(scale)
    #     center = (self.n / 2.0, self.n / 2.0, self.n / 2.0)
    #     mesh.set_position(center)
