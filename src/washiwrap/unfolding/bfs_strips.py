from __future__ import annotations
from collections import deque, defaultdict
from typing import Dict, List, Tuple
import numpy as np

from ..faces import Face2D
from ..utils import unit, rotation2d, reflect_along_unit_axis
from .base import StripNet, UnfoldResult

def _place_child_on_parent(parent_face: Face2D, parent_coords: np.ndarray, child_face: Face2D, shared_edge: tuple[int,int]) -> np.ndarray:
    """
    Hinge the child across the shared edge; reflect "outward"; return child's global 2D coordinates.
    """
    va, vb = shared_edge
    # Indices in parent's and child's boundary loops
    def idx(loop, val): return loop.index(val)
    try:
        pa, pb = idx(parent_face.vert_ids, va), idx(parent_face.vert_ids, vb)
    except ValueError:
        pa, pb = idx(parent_face.vert_ids, vb), idx(parent_face.vert_ids, va)
        va, vb = vb, va
    try:
        ca, cb = idx(child_face.vert_ids, va), idx(child_face.vert_ids, vb)
    except ValueError:
        ca, cb = idx(child_face.vert_ids, vb), idx(child_face.vert_ids, va)

    a_g = parent_coords[pa]
    b_g = parent_coords[pb]
    u_g = unit(b_g - a_g)

    a_l = child_face.local2d[ca]
    b_l = child_face.local2d[cb]
    u_l = unit(b_l - a_l)

    # rotate child's edge onto parent's edge; then reflect across that axis
    ang = np.arctan2(u_g[1], u_g[0]) - np.arctan2(u_l[1], u_l[0])
    R = rotation2d(float(ang))
    Ref = reflect_along_unit_axis(u_g)
    S = Ref @ R
    child = (child_face.local2d - a_l) @ S.T + a_g
    return child

def unfold_bfs_strips(
    faces: list[Face2D],
    adj: dict[int, list[int]],
    shared_edge: dict[tuple[int,int], tuple[int,int]],
    tape_width_mm: float,
) -> UnfoldResult:
    """
    Default robust strategy:
    - Grow a spanning forest over the face graph;
    - Place neighbor faces breadth-first; if adding a face would exceed the tape height bound,
      skip it for the current strip and place it later as a new strip.
    """
    face_lut = {f.fid: f for f in faces}
    degrees = {f.fid: len(adj.get(f.fid, [])) for f in faces}
    unplaced = set(face_lut.keys())
    strips: list[StripNet] = []

    def pick_root() -> int:
        return max(unplaced, key=lambda fid: degrees[fid])

    while unplaced:
        root = pick_root()
        root_face = face_lut[root]
        strip_coords: dict[int, np.ndarray] = {root: root_face.local2d.copy()}
        unplaced.remove(root)

        y_min = float(strip_coords[root][:,1].min())
        y_max = float(strip_coords[root][:,1].max())

        q = deque()
        parent_of: dict[int, int] = {}
        for n in adj.get(root, []):
            if n in unplaced:
                parent_of[n] = root
                q.append(n)

        while q:
            child = q.popleft()
            if child not in unplaced:
                continue
            parent = parent_of[child]
            if parent not in strip_coords:
                continue
            parent_face = face_lut[parent]
            child_face = face_lut[child]
            hinge = shared_edge[(parent, child)]
            child_coords = _place_child_on_parent(parent_face, strip_coords[parent], child_face, hinge)

            cmin, cmax = float(child_coords[:,1].min()), float(child_coords[:,1].max())
            new_y_min = min(y_min, cmin)
            new_y_max = max(y_max, cmax)

            if (new_y_max - new_y_min) <= tape_width_mm + 1e-6:
                strip_coords[child] = child_coords
                y_min, y_max = new_y_min, new_y_max
                unplaced.remove(child)
                for n2 in adj.get(child, []):
                    if n2 in unplaced and n2 not in parent_of:
                        parent_of[n2] = child
                        q.append(n2)
            else:
                # skip now; will be handled by next strip
                continue

        strips.append(StripNet(strip_coords, order=None))

    return UnfoldResult(strips)
