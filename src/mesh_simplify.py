"""
Quadric Error Decimation based mesh simplification for SMPL-X meshes.

Implements the classic Garland-Heckbert quadric error metric for mesh
simplification, suitable for reducing the vertex count of body meshes
generated from the SMPL-X parametric model while preserving visual
fidelity.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def numpy_to_o3d(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Convert numpy arrays to an Open3D TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


def o3d_to_numpy(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """Convert an Open3D TriangleMesh back to numpy arrays."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    return verts, faces


def simplify_mesh(vertices: np.ndarray,
                  faces: np.ndarray,
                  target_faces: int,
                  agressiveness: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a triangle mesh using quadric error decimation.

    Parameters
    ----------
    vertices : (N, 3) float
    faces : (M, 3) int
    target_faces : int
        Desired number of output triangles. Should be <= len(faces).
    agressiveness : int
        Kept for API compatibility with older Open3D versions. Newer
        Open3D exposes ``maximum_error`` instead; we simply pass the
        default (inf) here.

    Returns
    -------
    new_vertices : (N', 3) float
    new_faces : (M', 3) int
    """
    if target_faces >= len(faces):
        return vertices, faces

    mesh = numpy_to_o3d(vertices, faces)
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces),
    )
    mesh.compute_vertex_normals()
    return o3d_to_numpy(mesh)


def compute_geometric_error(verts_ref: np.ndarray,
                            faces_ref: np.ndarray,
                            verts_simp: np.ndarray,
                            faces_simp: np.ndarray) -> dict:
    """Compute geometric error between the original and the simplified mesh.

    Reports both vertex-to-vertex RMS error (after nearest-neighbour
    matching) and Hausdorff-style symmetric max distance in millimetres
    (assuming input vertices are in metres, which is the SMPL-X convention).
    """
    from scipy.spatial import cKDTree

    tree_ref = cKDTree(verts_ref)
    dist_simp_to_ref, _ = tree_ref.query(verts_simp, k=1)
    tree_simp = cKDTree(verts_simp)
    dist_ref_to_simp, _ = tree_simp.query(verts_ref, k=1)

    rms = float(np.sqrt(np.mean(np.concatenate([dist_simp_to_ref,
                                                dist_ref_to_simp]) ** 2)))
    max_sym = float(max(dist_simp_to_ref.max(), dist_ref_to_simp.max()))
    mean_sym = float(np.mean(np.concatenate([dist_simp_to_ref,
                                              dist_ref_to_simp])))
    return {
        "rms_m": rms,
        "max_m": max_sym,
        "mean_m": mean_sym,
        "rms_mm": rms * 1000.0,
        "max_mm": max_sym * 1000.0,
        "mean_mm": mean_sym * 1000.0,
    }