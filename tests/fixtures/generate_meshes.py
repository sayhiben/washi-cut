from __future__ import annotations
import numpy as np
import trimesh

def make_tetrahedron(side_mm: float = 10.0) -> trimesh.Trimesh:
    """
    Regular tetrahedron centered near origin. Edge lengths not exact to 'side_mm' but scaled approximately.
    For unfolding tests we only need a small; watertight; convex shell.
    """
    # Four vertices of a regular tetrahedron (scaled)
    v = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1]
    ], dtype=float)
    v = v / np.linalg.norm(v[0] - v[1]) * side_mm  # approx scale the edge
    faces = np.array([
        [0,1,2],
        [0,3,1],
        [0,2,3],
        [1,3,2]
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=True)
    return mesh

def make_cube(side_mm: float = 20.0) -> trimesh.Trimesh:
    """
    Triangulated cube with side length ~ side_mm.
    """
    s = side_mm / 2.0
    v = np.array([
        [-s,-s,-s], [ s,-s,-s], [ s, s,-s], [-s, s,-s],
        [-s,-s, s], [ s,-s, s], [ s, s, s], [-s, s, s],
    ], dtype=float)
    # 12 triangles; two per face
    faces = np.array([
        [0,1,2], [0,2,3],  # bottom
        [4,5,6], [4,6,7],  # top
        [0,1,5], [0,5,4],  # front
        [1,2,6], [1,6,5],  # right
        [2,3,7], [2,7,6],  # back
        [3,0,4], [3,4,7],  # left
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=True)
    return mesh
