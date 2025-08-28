from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from shapely import affinity
from shapely.ops import unary_union
from .geometry import rotate_to_min_height, translate_to_positive, geometry_to_svg_paths

@dataclass(frozen=True)
class LayoutResult:
    geoms: list  # list of placed geometries
    canvas_w_mm: float
    canvas_h_mm: float

def layout_strips(strip_geoms: list, tape_width_mm: float, gap_mm: float, margin_mm: float, duplicates: int) -> LayoutResult:
    """
    Rotate each strip to minimize height; then place them left-to-right.
    Vertically center each within the available tape band height.
    Duplicate the entire layout horizontally if requested.
    """
    processed = []
    for geom in strip_geoms:
        _, g = rotate_to_min_height(geom)
        g = translate_to_positive(g)
        minx, miny, maxx, maxy = g.bounds
        height = maxy - miny
        ypad = max(0.0, (tape_width_mm - height) * 0.5)
        g = affinity.translate(g, xoff=0.0, yoff=ypad)
        processed.append(g)

    # Place left to right
    placed = []
    x = 0.0
    for g in processed:
        minx, miny, maxx, maxy = g.bounds
        placed.append(affinity.translate(g, xoff=x, yoff=0.0))
        x = maxx + gap_mm
    set_width = (x - gap_mm) if placed else 0.0

    # Duplicate sets
    all_geoms = []
    for i in range(max(1, duplicates)):
        xoff = i * (set_width + gap_mm)
        for g in placed:
            all_geoms.append(affinity.translate(g, xoff=xoff, yoff=0.0))

    # Compute canvas bounds
    if all_geoms:
        union_all = unary_union(all_geoms)
        minx, miny, maxx, maxy = union_all.bounds
        canvas_w = (maxx - minx) + 2 * margin_mm
    else:
        canvas_w = 100.0  # fallback width
    canvas_h = tape_width_mm + 2 * margin_mm

    # Shift all shapes by margins
    shifted = [affinity.translate(g, xoff=margin_mm, yoff=margin_mm) for g in all_geoms]

    return LayoutResult(shifted, canvas_w, canvas_h)
