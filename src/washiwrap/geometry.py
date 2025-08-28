from __future__ import annotations
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely import affinity
from typing import List, Tuple
import numpy as np

def polygon_from_coords(coords: np.ndarray) -> Polygon:
    """Create a shapely Polygon from an Nx2 array of coordinates; fix invalids with buffer(0)."""
    poly = Polygon(coords)
    return poly if poly.is_valid else poly.buffer(0)

def union_strip_polygons(polys: list[Polygon], shrink_mm: float = 0.0):
    """
    Union all polygons in a strip to a single geometry; optionally shrink each slightly first.
    """
    if shrink_mm != 0.0:
        processed = []
        for p in polys:
            s = p.buffer(-abs(shrink_mm), join_style=2)
            if s.is_empty:
                s = p
            processed.append(s)
        polys = processed
    if not polys:
        return GeometryCollection()
    merged = unary_union(polys)
    return merged

def rotate_to_min_height(geom) -> tuple[float, object]:
    """
    Scan 0..179 degrees; find rotation angle that minimizes height (maxy - miny).
    """
    best_angle = 0.0
    best_height = float("inf")
    best_geom = geom
    for deg in range(0, 180, 1):
        g = affinity.rotate(geom, deg, origin=(0, 0))
        minx, miny, maxx, maxy = g.bounds
        h = maxy - miny
        if h < best_height:
            best_height = h
            best_angle = deg
            best_geom = g
    return best_angle, best_geom

def translate_to_positive(geom):
    minx, miny, maxx, maxy = geom.bounds
    return affinity.translate(geom, xoff=-minx, yoff=-miny)

def geometry_to_svg_paths(geom) -> list[str]:
    """
    Convert shapely Polygon/MultiPolygon to SVG path commands.
    """
    def ring_to_d(ring) -> str:
        coords = list(ring.coords)
        if not coords:
            return ""
        d = [f"M {coords[0][0]:.3f},{coords[0][1]:.3f}"]
        for x, y in coords[1:]:
            d.append(f"L {x:.3f},{y:.3f}")
        d.append("Z")
        return " ".join(d)

    paths = []
    if geom.is_empty:
        return paths

    if isinstance(geom, Polygon):
        paths.append(ring_to_d(geom.exterior))
        for hole in geom.interiors:
            paths.append(ring_to_d(hole))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            paths.extend(geometry_to_svg_paths(poly))
    else:
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                paths.extend(geometry_to_svg_paths(g))
    return paths
