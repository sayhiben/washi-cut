from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import trimesh
from collections import defaultdict
from .utils import unit

@dataclass(frozen=True)
class Face2D:
    """
    One coplanar polygonal face of the dice shell; with:
      - global face id
      - ordered boundary vertex ids
      - 2D local coordinates for the face (in its own plane)
      - 3D unit normal
    """
    fid: int
    vert_ids: List[int]
    local2d: np.ndarray
    normal3d: np.ndarray

def _order_loop_from_edges(edge_verts: np.ndarray) -> List[int]:
    """
    Given undirected boundary edges (pairs of mesh vertex ids), order them into a single closed loop.
    Assumes a simple; hole-free face boundary.
    """
    neighbors = defaultdict(list)
    for u, v in edge_verts:
        u, v = int(u), int(v)
        neighbors[u].append(v)
        neighbors[v].append(u)

    start = int(edge_verts[0][0])
    loop = [start]
    prev = None
    curr = start
    while True:
        nbrs = neighbors[curr]
        if len(nbrs) != 2:
            raise ValueError("Non-manifold boundary; expected degree-2 vertices for face loop")
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, curr = curr, nxt
    return loop

def _face_local_2d(vertices3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a stable 2D coordinate system for a planar polygon.
    Returns: (local2d Nx2; p0 3d; u 3d; v 3d)
    """
    if len(vertices3d) < 3:
        raise ValueError("Face has < 3 vertices")
    p0 = vertices3d[0]
    # choose non-zero edge
    i1 = 1
    while i1 < len(vertices3d) and np.linalg.norm(vertices3d[i1] - p0) == 0:
        i1 += 1
    e1 = vertices3d[i1] - p0
    if np.linalg.norm(e1) == 0:
        raise ValueError("Degenerate face; duplicate vertices")

    # find a second non-collinear edge to define the normal
    n = None
    for i in range(i1 + 1, len(vertices3d)):
        e2 = vertices3d[i] - p0
        cr = np.cross(e1, e2)
        if np.linalg.norm(cr) > 1e-9:
            n = unit(cr)
            break
    if n is None:
        raise ValueError("Collinear face; cannot compute normal")

    u = unit(e1)
    v = unit(np.cross(n, u))
    diffs = vertices3d - p0
    x = diffs @ u
    y = diffs @ v
    local2d = np.column_stack([x, y])
    return local2d, p0, u, v

def extract_faces_and_adjacency(
    mesh: trimesh.Trimesh,
) -> tuple[
    list[Face2D],
    dict[int, list[int]],
    dict[tuple[int, int], tuple[int, int]],
]:
    """
    Group coplanar triangles into polygonal faces; extract ordered boundaries; build face adjacency on shared edges.
    Returns:
      - faces: list[Face2D]
      - adj: dict face_id -> list of neighboring face_ids
      - shared_edge: dict (i, j) -> tuple(global vertex ids (a, b)) for the common edge
    """
    # Accessing facets triggers computation inside trimesh
    _ = mesh.facets
    boundaries = mesh.facets_boundary
    normals = mesh.facets_normal

    faces: list[Face2D] = []
    if len(boundaries) == 0:
        # fallback: treat each triangle as its own face
        for fid, tri in enumerate(mesh.faces):
            loop_vids = tri.tolist()
            verts3d = mesh.vertices[loop_vids]
            local2d, _, _, _ = _face_local_2d(verts3d)
            n = unit(np.cross(verts3d[1] - verts3d[0], verts3d[2] - verts3d[0]))
            faces.append(Face2D(fid=fid, vert_ids=loop_vids, local2d=local2d, normal3d=n))
        # adjacency via face_adjacency
        adj: dict[int, list[int]] = defaultdict(list)
        shared_edge: dict[tuple[int, int], tuple[int, int]] = {}
        for (f0, f1), e in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
            adj[f0].append(f1)
            adj[f1].append(f0)
            e = (int(e[0]), int(e[1]))
            e = (min(e[0], e[1]), max(e[0], e[1]))
            shared_edge[(f0, f1)] = e
            shared_edge[(f1, f0)] = e
        return faces, adj, shared_edge

    faces: list[Face2D] = []
    for fid, edges in enumerate(boundaries):  # Kx2 vertex index pairs
        loop_vids = _order_loop_from_edges(edges)
        verts3d = mesh.vertices[loop_vids]
        local2d, _, _, _ = _face_local_2d(verts3d)
        faces.append(Face2D(
            fid=fid,
            vert_ids=loop_vids,
            local2d=local2d,
            normal3d=unit(normals[fid])
        ))

    # adjacency via shared edges
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f in faces:
        vids = f.vert_ids
        for i in range(len(vids)):
            a = vids[i]
            b = vids[(i + 1) % len(vids)]
            e = (min(a, b), max(a, b))
            edge_to_faces[e].append(f.fid)

    adj: dict[int, list[int]] = defaultdict(list)
    shared_edge: dict[tuple[int, int], tuple[int, int]] = {}
    for e, fids in edge_to_faces.items():
        if len(fids) == 2:
            f0, f1 = fids
            adj[f0].append(f1)
            adj[f1].append(f0)
            shared_edge[(f0, f1)] = e
            shared_edge[(f1, f0)] = e

    return faces, adj, shared_edge
