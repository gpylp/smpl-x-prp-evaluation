"""
Evaluation utilities on the AGORA dataset.

Loads the AGORA ``camera_meta`` pickle files (DataFrame format with one
row per image and per-person columns stored as Python lists), projects
the SMPL-X ground-truth 3D keypoints onto the image using the camera
parameters, runs the 2D-driven reconstruction, and reports:

* 3D MPJPE / PA-MPJPE (joint error in mm)
* Reprojection error (pixel)
* Mesh simplification error (RMS vertex distance in mm)
"""

from __future__ import annotations

import os
import pickle
import sys
import types
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# pandas 2.2+ compatibility shim
# ---------------------------------------------------------------------------
# AGORA pickles were produced with an older pandas that exposed
# ``pandas.core.indexes.numeric``. Newer pandas merged that into
# ``pandas.Index``. Register a stub so ``pickle.load`` succeeds.

def _install_pandas_shim() -> None:
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    shim = types.ModuleType("pandas.core.indexes.numeric")

    class _NumericIndex(pd.Index):
        pass

    class _IntegerIndex(_NumericIndex):
        pass

    class _Float64Index(_NumericIndex):
        pass

    class _Int64Index(_NumericIndex):
        pass

    class _UInt64Index(_NumericIndex):
        pass

    shim.NumericIndex = _NumericIndex
    shim.IntegerIndex = _IntegerIndex
    shim.Float64Index = _Float64Index
    shim.Int64Index = _Int64Index
    shim.UInt64Index = _UInt64Index
    sys.modules["pandas.core.indexes.numeric"] = shim


def load_agora_metadata(meta_path: str) -> pd.DataFrame:
    """Load an AGORA camera_meta pickle file.

    Each row corresponds to one image; columns X, Y, Z are *lists* of
    per-person values (one entry per person in the image).
    """
    _install_pandas_shim()
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def get_first_person_row(df: pd.DataFrame, idx: int = 0) -> dict:
    """Return the per-person arrays for the first person of row ``idx``.

    AGORA stores 3D keypoints in **millimetres**, whereas SMPL-X uses
    metres; this helper converts to metres on the fly.
    """
    row = df.iloc[idx]
    return {
        "img_path": row["imgPath"],
        "joints_3d": (np.stack([row["X"], row["Y"], row["Z"]], axis=-1).astype(np.float32)
                      / 1000.0),
        "gender": row["gender"][0] if isinstance(row["gender"], (list, np.ndarray)) else row["gender"],
        "age": row["age"][0] if isinstance(row["age"], (list, np.ndarray)) else row["age"],
    }


def project_3d_to_2d(joints_3d: np.ndarray,
                     focal_length: float = 1000.0,
                     image_size: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """Weak-perspective projection used to synthesise 2D keypoints from
    the AGORA 3D ground truth. This simulates an off-the-shelf 2D
    keypoint detector (e.g. OpenPose) so that we have a fair 2D input
    for the reconstruction pipeline.
    """
    # Centre the 3D joints about their mean (no translation in 2D view)
    centred = joints_3d - joints_3d.mean(axis=0, keepdims=True)
    # Use first two dimensions for the projection (XY plane)
    pts2d = centred[:, :2]
    # Normalise so that the spread fills ~60 % of the image
    rng = max(pts2d.max() - pts2d.min(), 1e-6)
    target = 0.6 * min(image_size)
    scale = target / rng
    pts2d = pts2d * scale
    pts2d += np.array(image_size[::-1]) / 2.0
    return pts2d.astype(np.float32)


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Per Joint Position Error in millimetres."""
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    return float(np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean()) * 1000.0


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned MPJPE in millimetres.

    Aligns ``pred`` to ``gt`` via Procrustes (rigid rotation + translation +
    isotropic scale) and reports joint error after alignment.
    """
    from scipy.spatial import procrustes
    # scipy returns (m1, m2, disparity); m2 is pred warped to match m1 (gt).
    gt_arr = np.asarray(gt, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    _, pred_aligned, _ = procrustes(gt_arr, pred_arr)
    return float(np.sqrt(((pred_aligned - gt_arr) ** 2).sum(axis=-1)).mean()) * 1000.0


def reprojection_error_px(pred_2d: np.ndarray, gt_2d: np.ndarray) -> float:
    return float(np.sqrt(((pred_2d - gt_2d) ** 2).sum(axis=-1)).mean())


# Pelvis joint index in the SMPL-X 22-body joint layout (matches
# SMPLX_BODY_JOINT_NAMES in recon_2d.py: index 0 == "pelvis").
PELVIS_INDEX = 0


def pelvis_subtract(joints: np.ndarray, pelvis_idx: int = PELVIS_INDEX) -> np.ndarray:
    """Subtract the pelvis joint from every joint to make the pose
    translation-invariant. Useful when comparing two predictions whose
    global ``transl`` may differ wildly (e.g. raw-pose vs VPoser)."""
    return joints - joints[pelvis_idx:pelvis_idx + 1]


def pelvis_mpjpe(pred: np.ndarray, gt: np.ndarray,
                pelvis_idx: int = PELVIS_INDEX) -> float:
    """Pelvis-relative MPJPE in millimetres.

    Subtracts the pelvis joint from both prediction and ground truth
    before computing the per-joint Euclidean error. This removes the
    unknown global translation that a pure 2D-driven fitter cannot
    recover, leaving only the pose/shape error.
    """
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    pred_p = pelvis_subtract(pred, pelvis_idx)
    gt_p = pelvis_subtract(gt, pelvis_idx)
    return float(np.sqrt(((pred_p - gt_p) ** 2).sum(axis=-1)).mean()) * 1000.0


def pelvis_pa_mpjpe(pred: np.ndarray, gt: np.ndarray,
                    pelvis_idx: int = PELVIS_INDEX) -> float:
    """Pelvis-relative + Procrustes-aligned MPJPE in millimetres.

    First subtracts the pelvis joint (translation-invariant) and then
    applies a rigid rotation + isotropic scale (Procrustes) to align the
    prediction with the ground truth. This is the most permissive 3D
    metric: only the relative pose of the limbs matters.
    """
    from scipy.spatial import procrustes
    pred_p = pelvis_subtract(np.asarray(pred, dtype=np.float64), pelvis_idx)
    gt_p = pelvis_subtract(np.asarray(gt, dtype=np.float64), pelvis_idx)
    _, pred_aligned, _ = procrustes(gt_p, pred_p)
    return float(np.sqrt(((pred_aligned - gt_p) ** 2).sum(axis=-1)).mean()) * 1000.0


def pelvis_reprojection_error_px(pred_2d: np.ndarray, gt_2d: np.ndarray,
                                  pelvis_idx: int = PELVIS_INDEX) -> float:
    """2D reprojection error after aligning both sets by their
    pelvis (centroid + rotation + isotropic scale in 2D)."""
    p = pred_2d - pred_2d[pelvis_idx:pelvis_idx + 1]
    g = gt_2d - gt_2d[pelvis_idx:pelvis_idx + 1]
    # Procrustes in 2D: scale + rotation only (no translation after shift)
    from scipy.spatial import procrustes
    _, p_aligned, _ = procrustes(g, p)
    return float(np.sqrt(((p_aligned - g) ** 2).sum(axis=-1)).mean())


def evaluate(pred_joints: np.ndarray, gt_joints: np.ndarray,
             pred_2d: np.ndarray = None, gt_2d: np.ndarray = None,
             pelvis_idx: int = PELVIS_INDEX) -> dict:
    """One-stop evaluation: compute the four standard metrics.

    Returns a dict with keys:
        mpjpe, pa_mpjpe, pelvis_mpjpe, pelvis_pa_mpjpe,
        reproj_px (only if both ``pred_2d`` and ``gt_2d`` are given).
    """
    out = dict(
        mpjpe=mpjpe(pred_joints, gt_joints),
        pa_mpjpe=pa_mpjpe(pred_joints, gt_joints),
        pelvis_mpjpe=pelvis_mpjpe(pred_joints, gt_joints, pelvis_idx),
        pelvis_pa_mpjpe=pelvis_pa_mpjpe(pred_joints, gt_joints, pelvis_idx),
    )
    if pred_2d is not None and gt_2d is not None:
        out["reproj_px"] = reprojection_error_px(pred_2d, gt_2d)
        out["pelvis_reproj_px"] = pelvis_reprojection_error_px(
            pred_2d, gt_2d, pelvis_idx)
    return out