from __future__ import annotations
import argparse
import sys
import numpy as np
from shapely.geometry import Polygon

from .config import AppConfig
from .mesh_io import load_mesh_mm
from .faces import extract_faces_and_adjacency
from .geometry import polygon_from_coords, union_strip_polygons
from .layout import layout_strips
from .svg_export import export_svg
from .unfolding import unfold_bfs_strips, find_hamiltonian_ribbon, NoHamiltonianPath
from .unfolding.base import UnfoldResult

def _to_geometries(result: UnfoldResult, shrink_mm: float) -> list:
    """
    Convert StripNet -> shapely geometry by unioning per-strip polygons.
    """
    geoms = []
    for strip in result.strips:
        polys = [polygon_from_coords(coords) for coords in strip.faces_2d.values()]
        geom = union_strip_polygons(polys, shrink_mm=shrink_mm)
        if not geom.is_empty:
            geoms.append(geom)
    return geoms

def run(config: AppConfig) -> str:
    """
    Orchestrate the pipeline:
      STL -> faces+adj -> unfold -> union -> layout -> SVG
    """
    config.validate()
    np.random.seed(config.seed)

    mesh = load_mesh_mm(config.stl_path, config.stl_unit)
    faces, adj, shared = extract_faces_and_adjacency(mesh)

    # Unfold
    if config.mode == "hamiltonian":
        try:
            result = find_hamiltonian_ribbon(
                faces=faces,
                adj=adj,
                shared_edge=shared,
                tape_width_mm=config.tape_width_mm,
                beam=config.ham_beam,
                timeout_s=config.ham_timeout_s,
                seed=config.seed
            )
        except NoHamiltonianPath:
            if not config.ham_allow_fallback:
                raise
            result = unfold_bfs_strips(faces, adj, shared, tape_width_mm=config.tape_width_mm)
    else:
        result = unfold_bfs_strips(faces, adj, shared, tape_width_mm=config.tape_width_mm)

    # Geometry; layout; and export
    strip_geoms = _to_geometries(result, shrink_mm=config.shrink_mm)
    layout = layout_strips(strip_geoms, tape_width_mm=config.tape_width_mm, gap_mm=config.gap_mm, margin_mm=config.margin_mm, duplicates=config.duplicates)
    export_svg(layout.geoms, layout.canvas_w_mm, layout.canvas_h_mm, config.out_svg_path)
    return config.out_svg_path

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Cricut-ready SVG decals for washi tape wrapping of dice blanks.")
    p.add_argument("stl", help="Path to dice blank STL (or any mesh format supported by trimesh)")
    p.add_argument("--tape-width", type=float, required=True, help="Washi tape width in mm, e.g. 15")
    p.add_argument("--out", default="washi_wrap.svg", help="Output SVG path (default: washi_wrap.svg)")
    p.add_argument("--stl-unit", choices=["mm", "inch"], default="mm", help="Unit of input mesh (default: mm)")
    p.add_argument("--shrink", type=float, default=0.0, help="Optional inward offset per face in mm before union (helps avoid edge overhang)")
    p.add_argument("--gap", type=float, default=2.0, help="Gap between strips in the SVG, mm")
    p.add_argument("--margin", type=float, default=1.0, help="Canvas margin (all sides) in mm")
    p.add_argument("--duplicates", type=int, default=1, help="Duplicate the strip set horizontally this many times")
    p.add_argument("--mode", choices=["bfs", "hamiltonian"], default="bfs", help="Unfolding mode; default bfs")
    p.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaking")
    # Hamiltonian tuning
    p.add_argument("--ham-beam", type=int, default=24, help="Beam width for Hamiltonian search (higher = slower; more thorough)")
    p.add_argument("--ham-timeout", type=float, default=2.0, help="Soft time limit in seconds for Hamiltonian search")
    p.add_argument("--no-ham-fallback", action="store_true", help="Disable fallback to BFS if Hamiltonian search fails")
    return p

def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = AppConfig(
        stl_path=args.stl,
        tape_width_mm=args.tape_width,
        out_svg_path=args.out,
        stl_unit=args.stl_unit,
        shrink_mm=args.shrink,
        gap_mm=args.gap,
        margin_mm=args.margin,
        duplicates=args.duplicates,
        mode=args.mode,
        seed=args.seed,
        ham_beam=args.ham_beam,
        ham_timeout_s=args.ham_timeout,
        ham_allow_fallback=(not args.no_ham_fallback),
    )
    try:
        out_path = run(cfg)
        print(f"OK; wrote SVG to: {out_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
