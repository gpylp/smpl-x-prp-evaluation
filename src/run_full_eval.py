"""Full-batch evaluation of two SMPL-X fitters on the AGORA validation set.

This script replaces the default 16-image evaluation (selected in
``compare_fitters.py``) with a configurable batch size (default 200
images) and writes a per-image CSV plus an aggregated summary CSV.

The output CSV columns are identical to those of ``compare_fitters.py``
so that downstream plot scripts do not need to be changed.

Usage
-----
    python src/run_full_eval.py --num-images 200 --out-dir output_full_eval
    python src/run_full_eval.py --num-images 500 --include-hmr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make src importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from eval import (load_agora_metadata, get_first_person_row,
                  project_3d_to_2d, evaluate)
from recon_2d import build_smplx_model, fit_smplx_to_2d
from vposer_loader import load_vposer_v1
from compare_fitters import fit_smplx_with_vposer


def _build_eval_image(meta_files, total_images, seed=0):
    """Yield (image_id, img_path, joints_3d_m, kp_2d_noisy, gt_2d_clean) tuples.

    Walks the AGORA validation pickles and synthesises a 2D keypoint
    cloud with isotropic Gaussian noise (sigma=2.5 px) per image.
    """
    rng = np.random.default_rng(seed)
    count = 0
    for meta_path in meta_files:
        if count >= total_images:
            break
        df = load_agora_metadata(meta_path)
        for idx in range(len(df)):
            if count >= total_images:
                break
            row_dict = get_first_person_row(df, idx)
            joints_3d_m = row_dict["joints_3d"]  # (22, 3), metres
            gt_2d = project_3d_to_2d(joints_3d_m)
            kp_2d = gt_2d + rng.normal(0, 2.5, gt_2d.shape).astype(np.float32)
            yield (count, row_dict["img_path"], joints_3d_m, kp_2d.astype(np.float32), gt_2d)
            count += 1


def _safe_metric(fn, default=np.nan):
    """Run a metric function and return ``default`` on any exception."""
    try:
        return float(fn())
    except Exception:
        return float(default)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=200,
                   help="Number of AGORA val images to evaluate")
    p.add_argument("--meta-dir", default=r"dataset/agora/camera_meta/val",
                   help="Directory with AGORA val pickles")
    p.add_argument("--out-dir", default=r"output_full_eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-shape", type=float, default=1e-3)
    p.add_argument("--lambda-pose", type=float, default=1e-3)
    p.add_argument("--include-hmr", action="store_true",
                   help="If set, also run an HMR baseline (requires HMR installation).")
    p.add_argument("--skip-method", choices=["A", "B", "none"], default="none",
                   help="Skip one of the two fitters to save time")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_dir / "per_image.csv"
    summary_csv = out_dir / "summary.csv"

    # --- load models ---
    print("[init] building SMPL-X model ...", flush=True)
    model = build_smplx_model(gender="neutral", device="cpu")
    print("[init] loading VPoser ...", flush=True)
    vposer = load_vposer_v1(device="cpu")

    # --- gather meta files ---
    meta_dir = Path(args.meta_dir)
    meta_files = sorted(meta_dir.glob("*.pkl"))
    print(f"[init] found {len(meta_files)} val pickles", flush=True)
    if not meta_files:
        raise SystemExit(f"No pickle files found in {meta_dir}")

    # --- iterate images ---
    rows = []
    t_total = time.time()
    for i, img_path, joints_3d_m, kp_2d, gt_2d in _build_eval_image(
            meta_files, args.num_images, args.seed):
        t0 = time.time()
        out = {"i": i, "img_path": img_path}
        if args.skip_method != "A":
            res_a = fit_smplx_to_2d(
                model, kp_2d,
                num_steps=args.num_steps,
                lr=args.lr,
                lambda_shape=args.lambda_shape,
                lambda_pose=args.lambda_pose,
            )
            ev_a = evaluate(res_a["joints_3d"], joints_3d_m,
                            res_a["kp_2d"], gt_2d)
            out.update({f"A_{k}": v for k, v in ev_a.items()})
            out["A_betas_norm"] = float(np.linalg.norm(res_a["betas"]))
            out["A_pose_norm"] = float(np.linalg.norm(res_a["body_pose"]))
            out["A_time_s"] = time.time() - t0
        t1 = time.time()
        if args.skip_method != "B":
            res_b = fit_smplx_with_vposer(
                model, vposer, kp_2d,
                num_steps=args.num_steps, lr=args.lr,
                lambda_shape=args.lambda_shape,
                lambda_kl=args.lambda_pose,
            )
            ev_b = evaluate(res_b["joints_3d"], joints_3d_m,
                            res_b["kp_2d"], gt_2d)
            out.update({f"B_{k}": v for k, v in ev_b.items()})
            out["B_betas_norm"] = float(np.linalg.norm(res_b["betas"]))
            out["B_pose_norm"] = float(np.linalg.norm(res_b["body_pose"]))
            if "pose_z" in res_b:
                out["B_z_norm"] = float(np.linalg.norm(res_b["pose_z"]))
            out["B_time_s"] = time.time() - t1
        rows.append(out)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (args.num_images - i - 1)
            print(f"[{i+1:4d}/{args.num_images}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    # --- write per-image CSV ---
    df = pd.DataFrame(rows)
    df.to_csv(per_image_csv, index=False)
    print(f"[done] per-image CSV -> {per_image_csv}", flush=True)

    # --- summarise ---
    metric_cols = [c for c in df.columns
                   if any(suffix in c for suffix in
                          ["mpjpe", "reproj", "pa_mpjpe"])]
    summary = df[metric_cols].agg(["mean", "std"]).T
    summary.to_csv(summary_csv)
    print(f"[done] summary CSV  -> {summary_csv}", flush=True)
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
