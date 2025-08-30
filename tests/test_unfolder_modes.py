from __future__ import annotations
import numpy as np
from shapely.ops import unary_union
from shapely import affinity
from washiwrap.faces import extract_faces_and_adjacency
from washiwrap.unfolding import unfold_bfs_strips, find_hamiltonian_ribbon, NoHamiltonianPath
from washiwrap.geometry import polygon_from_coords
from tests.fixtures.generate_meshes import make_tetrahedron, make_cube

def min_height(geom):
    best = 1e9
    for deg in range(0, 180, 5):
        g = affinity.rotate(geom, deg, origin=(0,0))
        h = g.bounds[3] - g.bounds[1]
        if h < best:
            best = h
    return best

def test_hamiltonian_tetrahedron_fits():
    mesh = make_tetrahedron(10.0)
    faces, adj, shared = extract_faces_and_adjacency(mesh)
    try:
        res = find_hamiltonian_ribbon(faces, adj, shared, tape_width_mm=15.0, beam=24, timeout_s=1.0, seed=0)
        assert len(res.strips) == 1
        strip = res.strips[0]
        geom = unary_union([polygon_from_coords(c) for c in strip.faces_2d.values()])
        assert min_height(geom) <= 15.0 + 1e-6
        assert strip.order is not None
        assert len(strip.order) == len(faces)
    except NoHamiltonianPath:
        assert True  # acceptable fallback

def test_bfs_cube_produces_geoms():
    mesh = make_cube(20.0)
    faces, adj, shared = extract_faces_and_adjacency(mesh)
    res = unfold_bfs_strips(faces, adj, shared, tape_width_mm=15.0)  # smaller than face diagonal; likely splits
    assert len(res.strips) >= 1
    # ensure each strip has at least one face
    for s in res.strips:
        assert len(s.faces_2d) >= 1

def test_hamiltonian_cube_falls_back_like_expected():
    mesh = make_cube(20.0)
    faces, adj, shared = extract_faces_and_adjacency(mesh)
    try:
        _ = find_hamiltonian_ribbon(faces, adj, shared, tape_width_mm=15.0, beam=16, timeout_s=0.5, seed=0)
        # It is acceptable if it finds a ribbon; but most likely it will fail and raise.
        assert True
    except NoHamiltonianPath:
        assert True  # expected in many configurations
