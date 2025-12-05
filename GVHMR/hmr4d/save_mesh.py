import os
import numpy as np
import torch

def save_obj(path, verts: np.ndarray, faces: np.ndarray, verts_rgb=None):
    """
    Save mesh as simple OBJ.
    verts: (V,3) numpy
    faces: (F,3) numpy, zero-based indices
    verts_rgb: optional (V,3) 0..1 floats for colors
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, v in enumerate(verts):
            if verts_rgb is not None:
                r, g, b = verts_rgb[i]
                f.write("v %.6f %.6f %.6f %.6f %.6f %.6f\n" % (v[0], v[1], v[2], r, g, b))
            else:
                f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for face in faces:
            # OBJ is 1-based
            f.write("f %d %d %d\n" % (face[0]+1, face[1]+1, face[2]+1))

def save_ply_points(path, points: np.ndarray, colors: np.ndarray = None):
    """
    Save point cloud in ascii PLY (very portable).
    points: (N,3)
    colors: optional (N,3) 0..1 floats
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = points.shape[0]
    with open(path, "w") as f:
        if colors is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex %d\n" % N)
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points:
                f.write("%.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex %d\n" % N)
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(points, colors):
                r, g, b = (np.clip(c, 0, 1) * 255).astype(int)
                f.write("%.6f %.6f %.6f %d %d %d\n" % (p[0], p[1], p[2], r, g, b))
