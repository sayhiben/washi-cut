# WashiWrap

Generate Cricut‑ready SVG decals that wrap polyhedral dice blanks with washi tape.  
Two unfolding modes:
- **BFS strips** (default): robust; produces multiple decals if needed to satisfy a tape width; good baseline.
- **Hamiltonian** (opt‑in with `--mode hamiltonian`): tries to place all faces in a single serpentine ribbon; falls back to BFS if it cannot satisfy constraints (configurable).

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Usage

```bash
washiwrap path/to/blank.stl --tape-width 15 --out out.svg
```

Key options:

* `--mode bfs|hamiltonian` (default bfs)
* `--ham-beam 24` beam width used to guide the Hamiltonian search
* `--ham-timeout 2.0` seconds; soft limit; stops search and falls back if exceeded
* `--no-ham-fallback` to **disable** fallback to BFS when Hamiltonian fails
* `--stl-unit mm|inch` input unit; default mm
* `--shrink 0.1` shrink faces inward by 0.1 mm prior to union; reduces overhang on edges
* `--duplicates 3` tile the final strip set horizontally
* `--gap 2` gap between strips in the exported SVG
* `--margin 1` all‑sides margin on the SVG

The SVG uses millimeter units; Cricut Design Space imports at true size.

## Notes

* Input should be a clean; convex; watertight shell. For typical dice blanks this is the case.
* Hamiltonian ribbon might not exist without overlaps for some solids; the solver falls back by default.

## Tests

```bash
pip install -e "[test]"  # or just rely on project deps
pytest
```

