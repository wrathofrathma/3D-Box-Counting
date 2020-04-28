import sys
import math
from time import time
from copy import deepcopy

import numpy as np
import pywavefront as pwf

from lsystem.boxcounting3d.cubes import Cubes

# This entire setup assumes that we're working with the MeshObject used in the L'System Visualizer project

def calc_fractal_dim3D(mesh):
  """Wrapper function that calls calc_fractal_dim and narrows in on a good resolution for n"""

  # TODO testing
  # So in theory our accuracy should decrease if the cubes become smaller than the diameter of the smallest point of the mesh..
  # This would cause mesh parts to go unregistered. To prevent this, we can limit the cube side length to
  # 1 / n >= diameter. But this doesn't allow us to zoom in very far. For the pipe our diameter is 0.2...
  # This would seem to limit us to an n of 5 accurately?
  # This would limit us to 5 if we were doing a single instance of the mesh which would occupy a 1 unit length space.
  # But if we scale up, this value changes. If we have 2 pipes == side by side like this, then they're effectively scaled
  # to 1/2 size to fit. So our subdivision max becomes 1/2 * diamater? so our new max becomes 0.1.
  # We can ensure this factor to hold because we rescale the mesh by a factor of the greatest difference between
  # two points on the same axis. So a singular mesh, this factor would be 1x, two meshes laid length wise
  # would be 2 units long, or 2 units between the max two.
  # The question here is, is the factor we care about related to just the scale factor of n, or is it a
  # factor of the mesh itself? i.e. a dumb 1/n, or is it a ratio of how the mesh will be rescaled.
  # If it's a factor of the mesh x,y,z ratio, part of it would include the max delta of x y and z.
  radius = 0.1 # Hard coded for our pipe mesh. We'll need to adjust this if we want to support others.
  deltas = get_deltas(mesh)
  delta = max(deltas) # This is used both for our scale factor & for our min sidelength calc.
  # Min sidelength should be a function of 1/delta * diameter.
  # 0.1
  min_side = 1 / delta * 2 * radius
  max_n = int(1 / min_side)

# [(1, -1), (2, 3.0), (3, -1), (4, 2.0), (5, 1.7958889470453636), (6, 1.7737056144690833), (7, 1.8270874753469162),
# (8, 1.6666666666666667), (9, 1.9886214170079384), (10, 1.9344984512435675), (11, 1.8991141703557868), (12, 2.0),
# (13, 1.843072575667113), (14, 1.9415887339401865), (15, 1.8809659432212678), (16, 2.0), (17, 1.8972374314701483),
# (18, 1.959249866272526), (19, 2.020976693588105), (20, 2.004969940947389), (21, 1.98083127440203),
# (22, 2.02814952505589), (23, 2.0065637842976924), (24, 2.000545807561736), (25, 1.9878352824614485),
# (26, 2.0577703197939745), (27, 1.9928407195028168), (28, 2.023549459585051), (29, 2.043601646865157)]
  print("Min sidelength %f" % min_side)
  print("Max detected n: %f" % max_n)
  dims = []
  with open("somefile.db","w+") as f:
    for n in range(2, min(max_n+1,30)):
      dmesh = deepcopy(mesh)
      line = "n: {:d} s: {:f} | dim: {:f}".format(n, 1.0/n, calc_fractal_dim(n, dmesh, deltas))
      f.write(line)

 
  return 0

def get_deltas(mesh):
  """Calculates the deltas of x,y,z and returns a tuple of these values.
  """
  verts = mesh.get_vertices()
  xmax = np.max(verts[:, 0])
  ymax = np.max(verts[:, 1])
  zmax = np.max(verts[:, 2])
  xmin = np.min(verts[:, 0])
  ymin = np.min(verts[:, 1])
  zmin = np.min(verts[:, 2])
  delta_x = abs(xmax - xmin)
  delta_y = abs(ymax - ymin)
  delta_z = abs(zmax - zmin)
  return (delta_x, delta_y, delta_z)

def calc_fractal_dim(n: int, mesh, deltas):
  """Calculates the fractal dimensions for nxnxn cubes around the given mesh"""
  scale = center_mesh(mesh, n, deltas) # Center the mesh in teh cubic volume
  side_length = 1.0 / n  # We normalize the mesh, so the side length is important here.
  cubes = Cubes(n)
  start = time()
  intersections = cubes.intersect_mesh(mesh)
  end = time() - start
  print("Vertice count: %d" % len(mesh.get_vertices()))
  print("Faces in the mesh: %d" % len(mesh.get_faces()))
  print("N-Cubes: %d" % n**3)
  print("Side length: %f" % side_length)
  if(intersections > 0):
    print("Intersection count %d" % intersections)
    dim = math.log(intersections) / math.log(1 / side_length)
    print("Fractal dimension: " + str(dim))
  else:
    print("No intersections detected")
    dim = -1
  print("Took %f seconds" % end)
  return dim

def center_mesh(mesh, n, deltas):
  """Centers the 3d mesh in the space of our cube grid"""
  scale = max(deltas)
  scale = n / scale
  scale = np.array([scale, scale, scale])
  mesh.set_scale(scale)
  center = (n / 2.0, n / 2.0, n / 2.0)
  mesh.set_position(center)
  return scale

# def main():
#   # model = 'SierpinskiTetrahedron.obj'
#   model = 'MengerSponge.obj'
#   scene = pwf.Wavefront(model, collect_faces=True)

#   faces = np.array(scene.mesh_list[0].faces)
#   vertices = np.array(scene.vertices)

#   mesh = MeshObject(vertexes=vertices, indices=faces)
#   # calc_fractal_dim(10, mesh)
#   # calc_fractal_dim(20, mesh)
#   calc_fractal_dim(30, mesh)
#   # calc_fractal_dim(40, mesh)
#   # calc_fractal_dim(50, mesh)
#   # calc_fractal_dim(100, mesh)
#   # calc_fractal_dim(10, mesh)
#   # calc_fractal_dim(20, mesh)

