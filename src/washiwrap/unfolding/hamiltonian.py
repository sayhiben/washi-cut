from __future__ import annotations
import time
from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..faces import Face2D
from ..geometry import polygon_from_coords
from ..utils import unit, rotation2d, reflect_along_unit_axis
from .base import StripNet, UnfoldResult

class NoHamiltonianPath(Exception):
    """Raised when a Hamiltonian ribbon cannot be found under the constraints."""

def _place_child_on_parent(parent_face: Face2D, parent_coords: np.ndarray, child_face: Face2D, shared_edge: tuple[int,int]) -> np.ndarray:
    """
    Same hinge placement as BFS; kept local for isolation.
    """
    va, vb = shared_edge
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

    ang = np.arctan2(u_g[1], u_g[0]) - np.arctan2(u_l[1], u_l[0])
    S = reflect_along_unit_axis(u_g) @ rotation2d(float(ang))
    child = (child_face.local2d - a_l) @ S.T + a_g
    return child

def _min_height_coarse(geom) -> float:
    """
    Cheap heuristic: check a few angles to estimate min height; avoids full 0..180 scan during search.
    """
    from shapely import affinity
    best = float("inf")
    for deg in (0, 30, 45, 60, 90, 120, 150):
        g = affinity.rotate(geom, deg, origin=(0,0))
        h = g.bounds[3] - g.bounds[1]
        if h < best:
            best = h
    return best

def find_hamiltonian_ribbon(
    faces: list[Face2D],
    adj: dict[int, list[int]],
    shared_edge: dict[tuple[int,int], tuple[int,int]],
    tape_width_mm: float,
    beam: int = 24,
    timeout_s: float = 2.0,
    seed: int = 0,
    overlap_tol: float = 1e-4,
) -> UnfoldResult:
    """
    Attempt to find a Hamiltonian path visiting each face once; placed as one ribbon.
    Backtracks on overlaps; uses a small beam to guide expansion by current coarse height.

    Args:
        faces: Facets to unfold.
        adj: face adjacency list.
        shared_edge: mapping of shared edges between faces.
        tape_width_mm: maximum allowed strip height.
        beam: beam width guiding the search.
        timeout_s: soft time limit for the search.
        seed: random seed for tie-breaking.
        overlap_tol: allowed area overlap (mm^2) when placing a face; higher values permit small
            numerical noise. Default 1e-4.
    """
    rng = np.random.default_rng(seed)
    face_lut = {f.fid: f for f in faces}
    # Ensure adjacency list contains all faces
    adj = {fid: adj.get(fid, []) for fid in face_lut}
    # Start at the highest-degree face with at least one neighbor
    start_candidates = [fid for fid, nbs in adj.items() if nbs]
    if not start_candidates:
        raise NoHamiltonianPath("Mesh has no connected faces for Hamiltonian search")
    start = max(start_candidates, key=lambda fid: len(adj[fid]))
    start_face = face_lut[start]

    # initial state: one polygon
    placed_coords: dict[int, np.ndarray] = {start: start_face.local2d.copy()}
    placed_union = polygon_from_coords(placed_coords[start])
    path: list[int] = [start]
    best_partial = [(placed_coords, placed_union, path)]

    deadline = time.time() + max(0.1, timeout_s)

    while best_partial:
        # Beam select by current coarse min-height
        if time.time() > deadline:
            break
        scored = []
        for coords, geom, path in best_partial:
            h = _min_height_coarse(geom)
            scored.append((h, coords, geom, path))
        scored.sort(key=lambda t: t[0])
        best_partial = []
        for _, coords, geom, path in scored[:max(1, beam)]:
            if time.time() > deadline:
                break
            last = path[-1]
            used = set(path)
            # Order neighbors: prefer those that keep height low
            candidates = []
            for nb in adj.get(last, []):
                if nb in used:
                    continue
                child_face = face_lut[nb]
                hinge = shared_edge[(last, nb)]
                child_coords = _place_child_on_parent(face_lut[last], coords[last], child_face, hinge)
                child_poly = polygon_from_coords(child_coords)

                new_union = unary_union([geom, child_poly])
                # Overlap check: if union area < sum areas by more than epsilon; interiors overlapped
                overlap = (geom.area + child_poly.area) - new_union.area
                if overlap > overlap_tol:
                    continue

                # Score by coarse min-height of union
                h = _min_height_coarse(new_union)
                candidates.append((h, nb, child_coords, new_union))

            # If no candidates; dead end
            if not candidates:
                continue

            # Try better candidates first
            candidates.sort(key=lambda t: t[0])

            for h, nb, child_coords, new_union in candidates:
                coords2 = dict(coords)
                coords2[nb] = child_coords
                path2 = path + [nb]

                if len(coords2) == len(faces):
                    # Full ribbon; final acceptance if min-height ≤ tape width
                    if _min_height_coarse(new_union) <= tape_width_mm + 1e-6:
                        return UnfoldResult([StripNet(coords2, order=path2)])
                    # Otherwise keep searching; maybe another ordering fits better
                best_partial.append((coords2, new_union, path2))

    raise NoHamiltonianPath("Could not find a Hamiltonian ribbon within limits")
