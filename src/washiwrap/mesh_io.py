from __future__ import annotations
import trimesh

def load_mesh_mm(stl_path: str, stl_unit: str = "mm") -> trimesh.Trimesh:
    """
    Load a mesh and return a trimesh.Trimesh in millimeters.
    Accepts STL; OBJ; or any format trimesh can read.
    If a Scene is loaded; geometries are concatenated.
    """
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        # concatenate all geometries into one mesh
        if not mesh.geometry:
            raise ValueError("Scene contains no geometry")
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a mesh")

    # Normalize vertices and topology
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    if stl_unit == "inch":
        mesh.apply_scale(25.4)

    return mesh
