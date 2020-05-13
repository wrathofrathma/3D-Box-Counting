import sys
import math
from time import time
from copy import deepcopy

import numpy as np
import pywavefront as pwf
import pickle
from lsystem.boxcounting3d.cubes import intersect_mesh
from lsystem.graphics.mesh import MeshObject
from math import floor
from openpyxl import Workbook
from scipy import stats
from uuid import uuid4

# This entire setup assumes that we're working with the MeshObject used in the L'System Visualizer project

def dump_mesh(mesh: MeshObject, path: str):
  """Dumps a MeshObject to disk
  """
  pickle.dump(mesh,open(path, "wb"))

def load_mesh(path: str):
  """Loads a dumped meshobject from disk

  Returns:
  MeshObject: Loaded mesh object, or None if not found/invalid.
  """
  return pickle.load(open(path,"rb"))

def calc_fractal_dim3D(mesh, diameter = 0.2):
  """Wrapper function that calls calc_fractal_dim and narrows in on a good resolution for n"""

  wb = Workbook()
  headers = ["n", "dim", "side length"]
  sheet = wb.create_sheet(title="Fractal Data")
  sheet.append(headers)

  cmesh = deepcopy(mesh)
  cmesh.set_vertexes(center_on_origin(cmesh))
  deltas = get_deltas(cmesh)
  delta = max(deltas) # This is used both for our scale factor & for our min sidelength calc.

  min_side = (1 / delta) * diameter # 1/delta defines how the mesh scale will impact the diameter.
  max_n = int(1/min_side)

  dims = []
  n = 4 # Starting at 4x4x4 because 2x2x2 gives extremely inaccurate results in terms of the regression
  n_samples = 20 # The number of samples to create between the starting value and max dimension.
  step = max(floor((max_n-n)/n_samples), 1) # Step needs to be at least one

  print("Evaluating over n=range({:d},{:d}) with {:d} samples".format(n,max_n, floor((max_n-n)/step)))
  while(n<=max_n):
    if(n!=max_n):
      n+=1
      continue
    dmesh = deepcopy(cmesh)
    # line = "n: {:d} s: {:f} | dim: {:f}".format(n, 1.0/n, calc_fractal_dim(n, dmesh, deltas))
    if(n==max_n or n==4):
      save=True
    else:
      save=False
    dim = calc_fractal_dim(n, dmesh, deltas, save)
    print("n: {:d} | side_length: {:f} | dim: {:f}".format(n, 1.0/n, dim))
    dims += [(n, dim)]
    sheet.append([n, dim, 1.0/n])
    n+=step

  dims = np.array(dims)
  x = dims[:,0]
  y = dims[:,1]

  wb.save(filename="fractaldim" + str(uuid4()) + ".xlsx")
  wb.close()

  # slope, intercept, r_val, p_val, std_err = stats.linregress(x,y)
  # print(stats.linregress(x,y))
  return dims[-1][1]

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

def calc_fractal_dim(n: int, mesh: MeshObject, deltas, save=False):
  """Calculates the fractal dimensions for nxnxn cubes around the given mesh"""
  scale = center_mesh(mesh, n, deltas) # Center the mesh in teh cubic volume
  side_length = 1.0 / n  # We normalize the mesh, so the side length is important here.
  start = time()
  intersections = intersect_mesh(mesh,n, save)
  end = time() - start
  # print("Vertice count: %d" % len(mesh.get_vertices()))
  # print("Faces in the mesh: %d" % len(mesh.get_faces()))
  # print("N-Cubes: %d" % n**3)
  # print("Side length: %f" % side_length)
  if(intersections > 0):
    # print("Intersection count %d" % intersections)
    dim = math.log(intersections) / math.log(1 / side_length)
    # print("Fractal dimension: " + str(dim))
  else:
    print("No intersections detected")
    dim = -1
  # print("Took %f seconds" % end)
  return dim

def center_on_origin(mesh):
  verts = mesh.get_vertices()
  dx, dy, dz = get_deltas(mesh)

  xmax = np.max(verts[:, 0])
  ymax = np.max(verts[:, 1])
  zmax = np.max(verts[:, 2])

  xmin = np.min(verts[:, 0])
  ymin = np.min(verts[:, 1])
  zmin = np.min(verts[:, 2])

  xavg = (xmax + xmin)/2.0
  yavg = (ymax + ymin)/2.0
  zavg = (zmax + zmin)/2.0


  verts[:,0] -= xavg
  verts[:,1] -= yavg
  verts[:,2] -= zavg
  return verts

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
#   # mesh = load_mesh("objmesh")
#   # print(mesh.get_vertices())

#   calc_fractal_dim3D(mesh)

if __name__=="__main__":
  model = 'SierpinskiTetrahedron.obj'
  model = 'MengerSponge.obj'
  scene = pwf.Wavefront(model, collect_faces=True)

  faces = np.array(scene.mesh_list[0].faces)
  vertices = np.array(scene.vertices)

  mesh = MeshObject(vertexes=vertices, indices=faces)
  calc_fractal_dim3D(mesh, 0.05)
#   # calc_fractal_dim(20, mesh)
#   calc_fractal_dim(30, mesh)
#   # calc_fractal_dim(40, mesh)
#   # calc_fractal_dim(50, mesh)
#   # calc_fractal_dim(100, mesh)
#   # calc_fractal_dim(10, mesh)
#   # calc_fractal_dim(20, mesh)

# 4 iterations of the L'System for the Hilbert Curve
