"""
End-to-end pipeline: 2D-driven SMPL-X reconstruction + mesh simplification
+ evaluation on the AGORA benchmark.

Examples
--------
Run on the first 5 train images and produce mesh outputs::

    python src/run_pipeline.py --num-images 5 --num-steps 150

Run with mesh simplification at three reduction ratios and dump results::

    python src/run_pipeline.py --num-images 20 --ratios 1.0 0.5 0.25 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Make ``src`` importable when running as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from eval import (load_agora_metadata, get_first_person_row,
                  project_3d_to_2d, mpjpe, pa_mpjpe, reprojection_error_px)
from recon_2d import build_smplx_model, fit_smplx_to_2d, SMPLX_BODY_JOINT_NAMES
from mesh_simplify import simplify_mesh, compute_geometric_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", default="dataset/agora/camera_meta/val",
                   help="Directory containing AGORA camera_meta pkl files")
    p.add_argument("--model-dir", default="models",
                   help="Directory containing SMPL-X npz models")
    p.add_argument("--out-dir", default="output",
                   help="Output directory")
    p.add_argument("--num-images", type=int, default=10,
                   help="Number of images to process")
    p.add_argument("--num-steps", type=int, default=120,
                   help="Optimisation steps for the 2D-driven fitter")
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1],
                   help="Face retention ratios for mesh simplification")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-shape", type=float, default=1e-3)
    p.add_argument("--lambda-pose", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mesh_dir = os.path.join(args.out_dir, "meshes")
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load AGORA metadata
    # ------------------------------------------------------------------
    pkl_files = sorted(
        [os.path.join(args.meta_dir, f)
         for f in os.listdir(args.meta_dir)
         if f.endswith(".pkl")]
    )
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {args.meta_dir}")

    print(f"[1/4] Loading {len(pkl_files)} metadata file(s)...")
    dfs = [load_agora_metadata(p) for p in pkl_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"      Total images: {len(df)}")

    # We sample ``num_images`` rows.
    sample_idx = list(range(min(args.num_images, len(df))))

    # ------------------------------------------------------------------
    # 2. Load SMPL-X model
    # ------------------------------------------------------------------
    print(f"[2/4] Loading SMPL-X (neutral) model from {args.model_dir}...")
    model = build_smplx_model(args.model_dir, gender="neutral")
    faces = np.asarray(model.faces, dtype=np.int64)
    print(f"      Template: {model.v_template.shape[0]} verts, "
          f"{faces.shape[0]} faces")

    # ------------------------------------------------------------------
    # 3. Per-image reconstruction + simplification
    # ------------------------------------------------------------------
    print(f"[3/4] Running 2D-driven reconstruction on {len(sample_idx)} images...")

    rows = []
    n_orig_faces = int(faces.shape[0])

    # Pre-extract up to 22 body joint indices (the first 22 of SMPL-X)
    n_body = min(22, len(SMPLX_BODY_JOINT_NAMES))

    for k, idx in enumerate(sample_idx):
        try:
            info = get_first_person_row(df, idx)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip row {idx}: {exc}")
            continue

        joints_3d_gt_full = info["joints_3d"]        # (N, 3) in metres, N varies
        n_joints_avail = joints_3d_gt_full.shape[0]
        n_joints = min(n_joints_avail, n_body)
        if n_joints < 6:
            print(f"  skip row {idx}: only {n_joints} joints")
            continue
        joints_3d_gt = joints_3d_gt_full[:n_joints]

        # We pad / truncate the SMPL-X body joints to match.
        kp_2d = project_3d_to_2d(joints_3d_gt)
        kp_2d_noisy = kp_2d + np.random.normal(0, 2.5, kp_2d.shape).astype(np.float32)

        # Pad keypoints to at least 22 (NaN for missing) to keep the
        # optimiser API stable. We don't pad the SMPL-X joint list,
        # instead we constrain via the joint index list.
        if kp_2d_noisy.shape[0] < 22:
            padded = np.full((22, 2), np.nan, dtype=np.float32)
            padded[:kp_2d_noisy.shape[0]] = kp_2d_noisy
            kp_2d_full = padded
        else:
            kp_2d_full = kp_2d_noisy

        t0 = time.time()
        result = fit_smplx_to_2d(
            model,
            keypoints_2d=kp_2d_full,
            num_steps=args.num_steps,
            lr=args.lr,
            lambda_shape=args.lambda_shape,
            lambda_pose=args.lambda_pose,
            device=args.device,
        )
        t_fit = time.time() - t0

        verts = result["vertices"]
        joints_3d_pred = result["joints_3d"][:n_joints]

        err_mpjpe = mpjpe(joints_3d_pred, joints_3d_gt)
        err_pampjpe = pa_mpjpe(joints_3d_pred, joints_3d_gt)

        cam = result["camera"]
        proj_2d = (joints_3d_pred @ cam.rot.T) * cam.scale + cam.trans
        err_reproj = reprojection_error_px(proj_2d, kp_2d_noisy)

        # Mesh simplification at multiple ratios
        simp_stats = {}
        for ratio in args.ratios:
            target = max(4, int(round(n_orig_faces * ratio)))
            if ratio >= 1.0:
                target = n_orig_faces
            v_s, f_s = simplify_mesh(verts, faces, target)
            geom = compute_geometric_error(verts, faces, v_s, f_s)
            simp_stats[f"{ratio:.2f}"] = {
                "target_faces": target,
                "actual_faces": int(f_s.shape[0]),
                "rms_mm": geom["rms_mm"],
                "max_mm": geom["max_mm"],
                "mean_mm": geom["mean_mm"],
            }

        rows.append({
            "row_idx": int(idx),
            "img_path": info["img_path"],
            "mpjpe_mm": err_mpjpe,
            "pa_mpjpe_mm": err_pampjpe,
            "reproj_px": err_reproj,
            "fit_time_s": t_fit,
            "n_verts": int(verts.shape[0]),
            "n_faces": int(faces.shape[0]),
            "mesh_stats": simp_stats,
        })

        print(f"  [{k + 1:>3}/{len(sample_idx)}] row={idx} "
              f"MPJPE={err_mpjpe:6.1f}mm PA-MPJPE={err_pampjpe:6.1f}mm "
              f"Reproj={err_reproj:5.1f}px t={t_fit:4.1f}s")

    # ------------------------------------------------------------------
    # 4. Aggregate & save
    # ------------------------------------------------------------------
    if not rows:
        print("No rows were processed successfully.")
        return

    df_res = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "results.csv")
    df_res_flat = df_res.drop(columns=["mesh_stats"]).copy()
    df_res_flat.to_csv(summary_path, index=False)
    print(f"[4/4] Per-image results -> {summary_path}")

    # Mesh-simplification aggregate
    agg_rows = []
    for ratio in args.ratios:
        key = f"{ratio:.2f}"
        rms_vals, max_vals, n_faces_vals = [], [], []
        for r in rows:
            stats = r["mesh_stats"].get(key)
            if stats is None:
                continue
            rms_vals.append(stats["rms_mm"])
            max_vals.append(stats["max_mm"])
            n_faces_vals.append(stats["actual_faces"])
        if not rms_vals:
            continue
        agg_rows.append({
            "ratio": key,
            "n_faces_avg": int(np.mean(n_faces_vals)),
            "rms_mm_mean": float(np.mean(rms_vals)),
            "rms_mm_std": float(np.std(rms_vals)),
            "max_mm_mean": float(np.mean(max_vals)),
            "max_mm_std": float(np.std(max_vals)),
        })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(args.out_dir, "simplify_summary.csv"), index=False)
    print(f"Mesh simplification summary:")
    print(df_agg.to_string(index=False))

    # Combined JSON for reproducibility
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"per_image": rows, "simplify": agg_rows}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()