from __future__ import annotations
from .geometry import geometry_to_svg_paths

def export_svg(geoms: list, canvas_w_mm: float, canvas_h_mm: float, out_path: str) -> None:
    """
    Write an SVG with mm units; one path per polygon ring.
    """
    paths = []
    for g in geoms:
        for d in geometry_to_svg_paths(g):
            if not d:
                continue
            paths.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="0.1"/>')

    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{canvas_w_mm:.3f}mm" height="{canvas_h_mm:.3f}mm"
     viewBox="0 0 {canvas_w_mm:.3f} {canvas_h_mm:.3f}">
  <!-- WashiWrap export; units in millimeters -->
  {"".join(paths)}
</svg>
'''
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
