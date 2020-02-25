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

cube1_verts = np.array(
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
cwo = np.array((1.5, 1.5, 2.5))
cube2_verts = np.array(
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

cube1_faces = [
    [cube1_verts[2], cube1_verts[3], cube1_verts[7], cube1_verts[6]],  # back
    [cube1_verts[0], cube1_verts[1], cube1_verts[5], cube1_verts[4]],  # front
    [cube1_verts[0], cube1_verts[1], cube1_verts[2], cube1_verts[3]],  # Bottom
    [cube1_verts[4], cube1_verts[5], cube1_verts[6], cube1_verts[7]],  # top
    [cube1_verts[4], cube1_verts[7], cube1_verts[3], cube1_verts[0]],  # left
    [cube1_verts[1], cube1_verts[2], cube1_verts[6], cube1_verts[5]],  # right
]
cube2_faces = [
    [cube2_verts[2], cube2_verts[3], cube2_verts[7], cube2_verts[6]],  # back
    [cube2_verts[0], cube2_verts[1], cube2_verts[5], cube2_verts[4]],  # front
    [cube2_verts[0], cube2_verts[1], cube2_verts[2], cube2_verts[3]],  # Bottom
    [cube2_verts[4], cube2_verts[5], cube2_verts[6], cube2_verts[7]],  # top
    [cube2_verts[4], cube2_verts[7], cube2_verts[3], cube2_verts[0]],  # left
    [cube2_verts[1], cube2_verts[2], cube2_verts[6], cube2_verts[5]],  # right
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
be = [
    [bv[0], bv[1]],  # Right
    [bv[1], bv[2]],  # bottom
    [bv[2], bv[3]],  # left
    [bv[3], bv[0]],  # top
]
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
intersections1 = [
    False,  # back
    False,  # front
    False,  # bottom
    False,  # top
    False,  # left
    False,  # right
]
ray_intersects1 = []
v_intersect1 = []
intersections2 = [
    False,  # back
    False,  # front
    False,  # bottom
    False,  # top
    False,  # left
    False,  # right
]
ray_intersects2 = []
v_intersect2 = []
# For each line in our mesh(our square)....
# -- Check intersections against every cube that's not collided with(our singular cube).
for r in box_rays:
    for i in range(6):
        if not intersections1[i]:
            isec = geometric_tests.ray_intersect_plane(r, planes[i])
            if isec is not None:
                # Need to check the bounds
                print(isec)
                ys = [y[1] for y in cube1_faces[i]]
                xs = [x[0] for x in cube1_faces[i]]
                zs = [z[2] for z in cube1_faces[i]]
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
                intersections1[i] = True
                ray_intersects1 += [r]
                v_intersect1 += [isec]

for r in box_rays:
    for i in range(6):
        if not intersections2[i]:
            isec = geometric_tests.ray_intersect_plane(r, planes[i])
            if isec is not None:
                # Need to check the bounds
                print(isec)
                ys = [y[1] for y in cube2_faces[i]]
                xs = [x[0] for x in cube2_faces[i]]
                zs = [z[2] for z in cube2_faces[i]]
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
                intersections2[i] = True
                ray_intersects2 += [r]
                v_intersect2 += [isec]
print("Number of intersections found on c1: " + str(len(ray_intersects1)))
for r in ray_intersects1:
    print(r)
print(v_intersect1)
print("Number of intersections found on c2: " + str(len(ray_intersects2)))
for r in ray_intersects2:
    print(r)
print(v_intersect2)

########################################################################
# Matplotlib stuff
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
r = [-4, 4]
X, Y = np.meshgrid(r, r)
ax.scatter3D(cube1_verts[:, 0], cube1_verts[:, 1], cube1_verts[:, 2])

cf1 = []
for i in range(6):
    if intersections1[i] == False:
        cf1 += [cube1_faces[i]]
ax.add_collection3d(
    Poly3DCollection(cf1, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.25,)
)
cf2 = []
for i in range(6):
    if intersections2[i] == False:
        cf2 += [cube2_faces[i]]
ax.add_collection3d(
    Poly3DCollection(
        cf2, facecolors="purple", linewidths=1, edgecolors="r", alpha=0.25,
    )
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
