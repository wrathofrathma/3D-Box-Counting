import math
from math import radians, floor, ceil

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numba import jit, njit, prange
from numba.typed import List
from lsystem.graphics.mesh import MeshObject
from uuid import uuid4


###########
########### Code from https://github.com/adamlwgriffiths/Pyrr/blob/master/pyrr/plane.py
########### I've rehosted strictly because @njit decorators breaks with @parameters_as_numpy_arrays decorators
#########
def create(normal=None, distance=0.0, dtype=None):
    """Creates a plane oriented toward the normal, at distance below the origin.
    If no normal is provided, the plane will by created at the origin with a normal
    of [0, 0, 1].

    Negative distance indicates the plane is facing toward the origin.

    :rtype: numpy.array
    :return: A plane with the specified normal at a distance from the origin of
    -distance.
    """
    if normal is None:
        normal = [0.0, 0.0, 1.0]
    return np.array([normal[0], normal[1], normal[2], distance], dtype=dtype)

def create_from_points(vector1, vector2, vector3, dtype=None):
    """Create a plane from 3 co-planar vectors.

    The vectors must all lie on the same
    plane or an exception will be thrown.

    The vectors must not all be in a single line or
    the plane is undefined.

    The order the vertices are passed in will determine the
    normal of the plane.

    :param numpy.array vector1: a vector that lies on the desired plane.
    :param numpy.array vector2: a vector that lies on the desired plane.
    :param numpy.array vector3: a vector that lies on the desired plane.
    :raise ValueError: raised if the vectors are co-incident (in a single line).
    :rtype: numpy.array
    :return: A plane that contains the 3 specified vectors.
    """
    dtype = dtype or vector1.dtype

    # make the vectors relative to vector2
    relV1 = vector1 - vector2
    relV2 = vector3 - vector2

    # cross our relative vectors
    normal = np.cross(relV1, relV2)
    if np.count_nonzero(normal) == 0:
        raise ValueError("Vectors are co-incident")

    # create our plane
    return create_from_position(position=vector2, normal=normal, dtype=dtype)

def create_from_position(position, normal, dtype=None):
    """Creates a plane at position with the normal being above the plane
    and up being the rotation of the plane.

    :param numpy.array position: The position of the plane.
    :param numpy.array normal: The normal of the plane. Will be normalized
        during construction.
    :rtype: numpy.array
    :return: A plane that crosses the specified position with the specified
        normal.
    """
    dtype = dtype or position.dtype
    # -d = a * x  + b * y + c * z
    n = (normal.T / np.sqrt(np.sum(normal**2,axis=-1))).T
    d = -np.sum(n * position)
    return create(n, -d, dtype)


@njit
def intersect_plane(pl, ray, front_only=False):
    """Calculates the intersection point of a ray and a plane.
    :param numpy.array ray: The ray to test for intersection.
    :param numpy.array pl: The plane to test for intersection.
    :param boolean front_only: Specifies if the ray should
    only hit the front of the plane.
    Collisions from the rear of the plane will be
    ignored.
    :return The intersection point, or None
    if the ray is parallel to the plane.
    Returns None if the ray intersects the back
    of the plane and front_only is True.
    """
    """
    Distance to plane is defined as
    t = (pd - p0.n) / rd.n
    where:
    rd is the ray direction
    pd is the point on plane . plane normal
    p0 is the ray position
    n is the plane normal
    if rd.n == 0, the ray is parallel to the
    plane.
    """

    p = pl[3]
    n = pl[:3]
    rd_n = np.sum(ray[1]*n, axis=-1)

    if rd_n == 0.0:
        return None

    if front_only == True:
        if rd_n >= 0.0:
            return None

    pd = np.sum(p * n, axis=-1)
    p0_n = np.sum(ray[0]* n, axis=-1)
    t = (pd - p0_n) / rd_n
    return ray[0] + (ray[1] * t)
############ END Pyrr code
############
############
@njit
def line_intersect_plane(plane, line):
    """Check if the ray intersection is on our line.
    """
    # Create pyrr's ray
    a = line[0]
    d = line[1] - line[0]
    d = (d.T / np.sqrt(np.sum(d**2, axis=-1))).T
    ray = (a,d)

    b = line[1]
    c = intersect_plane(plane, ray)
    if(c is None):
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    if(np.dot((c-a), (b-c))<0):
        c[0]=-1
    return c

def generate_grid(n: int, cubes):
    """Generates cubes to be drawn in pyplot"""
    cr = 0.5
    cube_verts = []  # List of all cube vert arrays
    cube_faces = []  # List of cube faces
    colors = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
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
                if cubes[x][y][z] == 0.0:
                    colors += ["c"]
                else:
                    colors += ["r"]

    return (cube_verts, cube_faces, colors)


def save(n, cubes, mesh_verts, mesh_faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    r = [-n, n]
    X, Y = np.meshgrid(r, r)
    (grid, faces, colors) = generate_grid(n,cubes)
    for f, c in zip(faces, colors):
        alpha = 0.1
        if(c=="c"):
            continue
        ax.add_collection3d(
            Poly3DCollection(
                f, facecolors=c, linewidth=1, edgecolors=c, alpha=alpha
            )
        )
        # if c == "r":
        #     alpha = 0.25
        #     ax.add_collection3d(
        #         Poly3DCollection(
        #             f, facecolors=c, linewidth=1, edgecolors="r", alpha=alpha
        #         )
        #     )
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
                facecolors="c",
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
    # plt.show()
    fname = "boxcount-" + str(uuid4()) + ".png"
    plt.savefig(fname)

@njit
def get_cs(n, isec, ray):
    cs = []
    if(min(isec) < 0 or max(isec) > n):
        return cs
    mx = np.arange(1,n+1)
    mn = np.arange(0,n)
    # Get the indices of the part of the cube the intersection might be in
    xs = np.array((floor(isec[0]), ceil(isec[0])))
    ys = np.array((floor(isec[1]), ceil(isec[1])))
    zs = np.array((floor(isec[2]), ceil(isec[2])))
    # Check these indices are in the cube bounds....
    x = np.argwhere(((xs >= 0) & (xs< n))==True)
    y = np.argwhere(((ys >= 0) & (ys< n))==True)
    z = np.argwhere(((zs >= 0) & (zs< n))==True)

    for xi in x:
        for yj in y:
            for zk in z:
                cs.append((xs[xi[0]],ys[yj[0]],zs[zk[0]]))
    return cs

@njit(parallel=True)
def pintersect(cubes, verts, rays, xy, xz, yz, n):
    for i in prange(len(rays)):
        for j in range(len(xy)):
            isec = line_intersect_plane(xy[j], rays[i])
            if(isec[0]!=-1):
            # if(isec is not None):
                for c in get_cs(n, isec, rays[i]):
                    if(cubes[c[0]][c[1]][c[2]]==0):
                        cubes[c[0]][c[1]][c[2]] = 1

            isec = line_intersect_plane(xz[j], rays[i])
            if(isec[0]!=-1):
            # if(isec is not None):
                for c in get_cs(n, isec, rays[i]):
                    if(cubes[c[0]][c[1]][c[2]]==0):
                        cubes[c[0]][c[1]][c[2]] = 1

            isec = line_intersect_plane(yz[j], rays[i])
            if(isec[0]!=-1):
            # if(isec is not None):
                for c in get_cs(n, isec, rays[i]):
                    if(cubes[c[0]][c[1]][c[2]]==0):
                        cubes[c[0]][c[1]][c[2]] = 1

    icount = np.count_nonzero(cubes)
    return icount


def generate_planes(n: int):
    """Generates the planes of the cubes"""
    xy = List([create_from_points(np.array((0,0,z), dtype=np.float64),np.array((1,0,z), dtype=np.float64), np.array((0,1,z), dtype=np.float64)) for z in range(n+1)])
    xz = List([create_from_points(np.array((0,y,0), dtype=np.float64),np.array((1,y,0), dtype=np.float64), np.array((0,y,1), dtype=np.float64)) for y in range(n+1)])
    yz = List([create_from_points(np.array((x,0,0), dtype=np.float64),np.array((x,1,0), dtype=np.float64), np.array((x,0,1), dtype=np.float64)) for x in range(n+1)])
    return (xy, xz, yz)

def generate_rays(verts, faces):
    rays = []
    for face in faces:
        p1 = np.array(verts[face[0]], dtype=np.float64)
        p2 = np.array(verts[face[1]], dtype=np.float64)
        p3 = np.array(verts[face[2]], dtype=np.float64)
        rays+=[(p1,p2)]
        rays+=[(p1,p3)]
        rays+=[(p2,p3)]
    return rays

def intersect_mesh(mesh: MeshObject, n, save_plot: bool=False):
    # Cube intersection tracker
    cubes = np.zeros(shape=(n,n,n), dtype=np.int)
    # Planes to define the cubes
    xy, xz, yz = generate_planes(n)

    # Get relevant mesh data
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    rays = List(generate_rays(verts, faces))
    intersections = pintersect(cubes, verts, rays, xy, xz, yz, n)

    if(save_plot):
        save(n, cubes, verts, faces)

    return intersections
