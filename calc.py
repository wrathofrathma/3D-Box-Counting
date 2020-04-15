import sys
import math
from time import time
from copy import deepcopy

import numpy as np
import pywavefront as pwf

from cubes import Cubes

# This entire setup assumes that we're working with the MeshObject used in the L'System Visualizer project

def calc_fractal_dim(n: int, mesh):
  """Calculates the fractal dimensions for nxnxn cubes around the given mesh"""
  side_length = 1.0 / n  # We normalize the mesh, so the side length is important here.
  cmesh = deepcopy(mesh) # Making deep copies to make sure the mesh functions appropriately on multiple runs
  cubes = Cubes(n)
  cubes.center_mesh(cmesh)
  start = time()
  intersections = cubes.intersect_mesh(cmesh)
  end = time() - start
  dim = math.log(intersections) / math.log(1 / side_length)
  print("Vertice count: %d" % len(mesh.get_vertices()))
  print("Faces in the mesh: %d" % len(mesh.get_faces()))
  print("N-Cubes: %d" % n**3)
  print("Intersection count %d" % intersections)
  print("Side length: %f" % side_length)
  print("Fractal dimension: " + str(dim))
  print("Took %f seconds" % end)

def center_mesh(mesh, n):
  """Centers the 3d mesh in the space of our cube grid"""
  verts = mesh.get_vertices()
  xmax = np.max(verts[:, 0])
  ymax = np.max(verts[:, 1])
  zmax = np.max(verts[:, 2])
  xmin = np.min(verts[:, 0])
  ymin = np.min(verts[:, 1])
  zmin = np.min(verts[:, 2])
  xdiff = abs(xmax - xmin)
  ydiff = abs(ymax - ymin)
  zdiff = abs(zmax - zmin)
  scale = max(xdiff, ydiff, zdiff)
  scale = n / scale
  scale = np.array([scale, scale, scale])
  mesh.set_scale(scale)
  center = (n / 2.0, n / 2.0, n / 2.0)
  mesh.set_position(center)

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

