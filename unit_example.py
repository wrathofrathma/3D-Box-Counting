import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pyrr import line, ray, plane, geometric_tests

# SINGULAR CUBE STUFF
###############################################
# (0,0,0) would be (0.5,0.5,0.5) the first cube in positive x,y,z
cube_array_pos = (
    1,
    1,
    1,
)
cube_world_origin = np.array((1.5, 1.5, 1.5))  # Actual world position of cube
cube_radius = 0.5  # Radius, giving us planes on even coordinates
cr = cube_radius
cwo = cube_world_origin

cube_verts = np.array(
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

cube_faces = [
    [cube_verts[2], cube_verts[3], cube_verts[7], cube_verts[6]],  # back
    [cube_verts[0], cube_verts[1], cube_verts[5], cube_verts[4]],  # front
    [cube_verts[0], cube_verts[1], cube_verts[2], cube_verts[3]],  # Bottom
    [cube_verts[4], cube_verts[5], cube_verts[6], cube_verts[7]],  # top
    [cube_verts[4], cube_verts[7], cube_verts[3], cube_verts[0]],  # left
    [cube_verts[1], cube_verts[2], cube_verts[6], cube_verts[5]],  # right
]
################################################
# Box stuff (Hard coding the verts since I don't feel like porting the rotations yet)
box_verts = np.array(
    [
        np.array([1.5, 2.3, 2.5]),  # top right
        np.array([1.5, 2.3, 1.5]),  # bottom right
        np.array([2.5, 1.3, 1.5]),  # bottom left
        np.array([2.5, 1.3, 2.5]),  # top left
    ]
)
bv = box_verts
be = [[bv[0], bv[1]], [bv[1], bv[2]], [bv[2], bv[3]], [bv[3], bv[0]]]
box_edges = be
box_face = [[bv[0], bv[1], bv[2], bv[3]]]
#########################################################################
# Coding the rays & planes from previous data
box_lines = [line.create_from_points(v[0], v[1]) for v in be]
print(box_lines)
box_rays = [ray.create_from_line(l) for l in box_lines]
print(box_rays)
planes = [
    plane.create_xz(distance=2),  # back
    plane.create_xz(distance=1),  # front
    plane.create_xy(distance=1),  # bottom
    plane.create_xy(distance=2),  # top
    plane.create_yz(distance=1),  # left
    plane.create_yz(distance=2),  # right
]
intersections = [
    False,  # back
    False,  # front
    False,  # bottom
    False,  # top
    False,  # left
    False,  # right
]
ray_intersects = []
v_intersect = []
# For each line in our mesh(our square)....
# -- Check intersections against every cube that's not collided with(our singular cube).
for r in box_rays:
    for i in range(6):
        if not intersections[i]:
            isec = geometric_tests.ray_intersect_plane(r, planes[i])
            if isec is not None:
                # Need to check the bounds
                print(isec)
                ys = [y[1] for y in cube_faces[i]]
                xs = [x[0] for x in cube_faces[i]]
                zs = [z[2] for z in cube_faces[i]]
                maxy = max(ys)
                maxx = max(xs)
                maxz = max(zs)
                miny = min(ys)
                minx = min(xs)
                minz = min(zs)
                if isec[0] < minx or isec[0] > maxx:
                    continue
                if isec[1] < miny or isec[1] > maxy:
                    continue
                if isec[2] < minz or isec[2] > maxz:
                    continue
                print("Intersection found on i=" + str(i))
                print("ray: " + str(r))
                print("Isec: " + str(isec))
                intersections[i] = True
                ray_intersects += [r]
                v_intersect += [isec]

print("Number of intersections found: " + str(len(ray_intersects)))
for r in ray_intersects:
    print(r)
print(v_intersect)

########################################################################
# Matplotlib stuff
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
r = [-4, 4]
X, Y = np.meshgrid(r, r)
ax.scatter3D(cube_verts[:, 0], cube_verts[:, 1], cube_verts[:, 2])

cf = []
for i in range(6):
    if intersections[i] == False:
        cf += [cube_faces[i]]
ax.add_collection3d(
    Poly3DCollection(cf, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.25,)
)
ax.add_collection3d(
    Poly3DCollection(
        box_face, facecolors="g", linewidths=1, edgecolors="b", alpha=0.25,
    )
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#
