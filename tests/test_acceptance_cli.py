from __future__ import annotations
import os
from washiwrap.cli import run
from washiwrap.config import AppConfig
from tests.fixtures.generate_meshes import make_tetrahedron

def test_cli_generates_svg(tmpdir_path):
    # Save a small tetrahedron to STL
    import trimesh
    mesh = make_tetrahedron(10.0)
    stl_path = os.path.join(tmpdir_path, "tetra.stl")
    mesh.export(stl_path)

    cfg = AppConfig(
        stl_path=stl_path,
        tape_width_mm=15.0,
        out_svg_path=os.path.join(tmpdir_path, "out.svg"),
        mode="hamiltonian",  # should easily fit
        ham_allow_fallback=True
    )
    out = run(cfg)
    assert os.path.isfile(out)
    # Basic sanity checks on the file contents
    with open(out, "r", encoding="utf-8") as f:
        data = f.read()
    assert "<svg" in data
    assert "mm" in data
    assert "<path" in data
