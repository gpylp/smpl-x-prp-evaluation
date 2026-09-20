"""Ablation study for the 2D-driven SMPL-X fitter (Method A).

Varies one hyperparameter at a time while keeping the rest at defaults,
and records MPJPE / PA-MPJPE / reprojection error on AGORA val.

Usage
------
    python src/ablation.py --num-images 50 --num-steps 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from eval import (load_agora_metadata, get_first_person_row,
                  project_3d_to_2d, mpjpe, pa_mpjpe, pelvis_mpjpe,
                  pelvis_pa_mpjpe, reprojection_error_px,
                  pelvis_reprojection_error_px)
from recon_2d import build_smplx_model, fit_smplx_to_2d


# ---------------------------------------------------------------------------
# Ablation dimensions
# ---------------------------------------------------------------------------

# Default values (Method A baseline)
DEFAULTS = dict(
    num_steps=150,
    lr=0.05,
    lambda_shape=1e-3,
    lambda_pose=1e-3,
    noise_std=2.5,
)

# One-at-a-time variations
ABLATIONS = {
    "num_steps":       [60, 120, 200],
    "lr":              [0.005, 0.02, 0.05, 0.1],
    "lambda_shape":    [0.0, 1.0, 10.0, 100.0, 1000.0],
    "lambda_pose":     [0.0, 1.0, 10.0, 100.0, 1000.0],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir",
                   default="dataset/agora/camera_meta/val")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out-dir", default="output_ablation")
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of images to process in parallel (not used, kept for API)")
    p.add_argument("--log-path",
                   default="terminal_ablation.txt")
    return p.parse_args()


def run_one(cfg: dict, model, kp_2d_full: np.ndarray,
            joints_3d_gt: np.ndarray, n_joints: int,
            device: str) -> dict:
    """Run a single ablation configuration on one image."""
    try:
        t0 = time.time()
        res = fit_smplx_to_2d(
            model,
            keypoints_2d=kp_2d_full,
            num_steps=cfg["num_steps"],
            lr=cfg["lr"],
            lambda_shape=cfg["lambda_shape"],
            lambda_pose=cfg["lambda_pose"],
            device=device,
            verbose=False,
        )
        t_elapsed = time.time() - t0

        j3d = res["joints_3d"][:n_joints]
        err_mpjpe = mpjpe(j3d, joints_3d_gt)
        err_pa = pa_mpjpe(j3d, joints_3d_gt)
        err_pel = pelvis_mpjpe(j3d, joints_3d_gt)
        err_pel_pa = pelvis_pa_mpjpe(j3d, joints_3d_gt)
        cam = res["camera"]
        proj = (j3d @ cam.rot.T) * cam.scale + cam.trans
        # compare with noiseless 2D for a fair reproj metric
        kp_clean = project_3d_to_2d(joints_3d_gt)
        err_reproj = reprojection_error_px(proj, kp_clean)
        err_pel_reproj = pelvis_reprojection_error_px(proj, kp_clean)

        return dict(
            ok=True,
            mpjpe=err_mpjpe,
            pa_mpjpe=err_pa,
            pelvis_mpjpe=err_pel,
            pelvis_pa_mpjpe=err_pel_pa,
            reproj=err_reproj,
            pelvis_reproj=err_pel_reproj,
            time_s=t_elapsed,
            betas_norm=float(np.linalg.norm(res["betas"])),
            pose_norm=float(np.linalg.norm(res["body_pose"])),
        )
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, mpjpe=np.nan, pa_mpjpe=np.nan,
                     pelvis_mpjpe=np.nan, pelvis_pa_mpjpe=np.nan,
                     reproj=np.nan, pelvis_reproj=np.nan,
                     time_s=np.nan,
                     betas_norm=np.nan, pose_norm=np.nan)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Redirect stdout to a log file so we can read it later even if the
    # terminal output is lost (Tee-Object + PowerShell mishandles pipes).
    log_fh = open(args.log_path, "w", encoding="utf-8")
    _stdout = sys.stdout
    sys.stdout = _TeeWriter(_stdout, log_fh)

    try:
        _main(args)
    finally:
        sys.stdout = _stdout
        log_fh.close()


class _TeeWriter:
    """Write to two streams: the original stdout and a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


def _main(args: argparse.Namespace) -> None:

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pkl_files = sorted(
        os.path.join(args.meta_dir, f)
        for f in os.listdir(args.meta_dir) if f.endswith(".pkl")
    )
    if not pkl_files:
        raise FileNotFoundError(f"No pkl in {args.meta_dir}")

    print(f"[1/4] Loading AGORA ({len(pkl_files)} files)...")
    df = pd.concat([load_agora_metadata(p) for p in pkl_files],
                   ignore_index=True)
    n_rows = min(args.num_images, len(df))

    print(f"[2/4] Loading SMPL-X neutral from {args.model_dir}...")
    model = build_smplx_model(args.model_dir, gender="neutral")

    # ------------------------------------------------------------------
    # Collect per-image 2D / 3D data (fixed across all ablations)
    # ------------------------------------------------------------------
    print(f"[3/4] Preparing {n_rows} image(s) of 2D keypoints...")
    samples = []
    for idx in range(n_rows):
        try:
            info = get_first_person_row(df, idx)
        except Exception:  # noqa: BLE001
            continue
        joints_gt = info["joints_3d"][:22]
        if joints_gt.shape[0] < 6:
            continue
        kp_clean = project_3d_to_2d(joints_gt)
        # apply noise once — same noise for all configs
        rng = np.random.RandomState(args.seed + idx)
        kp_noisy = kp_clean + rng.normal(
            0, DEFAULTS["noise_std"], kp_clean.shape).astype(np.float32)
        if kp_noisy.shape[0] < 22:
            padded = np.full((22, 2), np.nan, dtype=np.float32)
            padded[:kp_noisy.shape[0]] = kp_noisy
            kp_noisy = padded
        samples.append(dict(
            idx=idx, joints_3d_gt=joints_gt,
            kp_2d=kp_noisy, n_joints=joints_gt.shape[0],
        ))
    if not samples:
        print("No valid samples found.")
        return
    print(f"       {len(samples)} valid samples.")

    # ------------------------------------------------------------------
    # Run ablations
    # ------------------------------------------------------------------
    results = []   # one row per (ablation_key, value, image_idx)

    # 1) Baseline with defaults
    print("\n[4/4] Running ablations...\n")
    print(f"{'Ablation':<20} {'Value':>12}  "
          f"{'MPJPE':>8}  {'PA-MPJPE':>8}  {'PelMPJPE':>9}  {'PelPA':>7}  "
          f"{'Reproj':>7}  {'PelRe':>7}  "
          f"{'Time':>6}  {'|b|':>6}  {'|p|':>6}  "
          f"{'N_ok':>5}")
    print("-" * 120)

    def _agg(rows):
        ok = [r for r in rows if r["ok"]]
        if not ok:
            return dict(mpjpe=np.nan, pa_mpjpe=np.nan,
                        pelvis_mpjpe=np.nan, pelvis_pa_mpjpe=np.nan,
                        reproj=np.nan, pelvis_reproj=np.nan,
                        time_s=np.nan,
                        betas_norm=np.nan, pose_norm=np.nan, n_ok=0)
        return dict(
            mpjpe=float(np.nanmean([r["mpjpe"] for r in ok])),
            pa_mpjpe=float(np.nanmean([r["pa_mpjpe"] for r in ok])),
            pelvis_mpjpe=float(np.nanmean([r["pelvis_mpjpe"] for r in ok])),
            pelvis_pa_mpjpe=float(np.nanmean([r["pelvis_pa_mpjpe"] for r in ok])),
            reproj=float(np.nanmean([r["reproj"] for r in ok])),
            pelvis_reproj=float(np.nanmean([r["pelvis_reproj"] for r in ok])),
            time_s=float(np.nanmean([r["time_s"] for r in ok])),
            betas_norm=float(np.nanmean([r["betas_norm"] for r in ok])),
            pose_norm=float(np.nanmean([r["pose_norm"] for r in ok])),
            n_ok=len(ok),
        )

    baseline_cfg = DEFAULTS.copy()
    baseline_rows = []
    for s in samples:
        row = run_one(baseline_cfg, model, s["kp_2d"],
                      s["joints_3d_gt"], s["n_joints"], args.device)
        baseline_rows.append(row)
    agg = _agg(baseline_rows)
    print(f"{'BASELINE (all default)':<20} {'default':>12}  "
          f"{agg['mpjpe']:>8.1f}  {agg['pa_mpjpe']:>8.1f}  "
          f"{agg['pelvis_mpjpe']:>9.1f}  {agg['pelvis_pa_mpjpe']:>7.1f}  "
          f"{agg['reproj']:>7.2f}  {agg['pelvis_reproj']:>7.2f}  "
          f"{agg['time_s']:>6.1f}  "
          f"{agg['betas_norm']:>6.2f}  {agg['pose_norm']:>6.2f}  "
          f"{agg['n_ok']:>5}")
    results.append({"ablation": "BASELINE", "value": "default",
                    **agg, "per_image": baseline_rows})

    # 2) Per-dimension variations
    for key, values in ABLATIONS.items():
        for val in values:
            cfg = DEFAULTS.copy()
            cfg[key] = val
            rows = []
            for s in samples:
                row = run_one(cfg, model, s["kp_2d"],
                              s["joints_3d_gt"], s["n_joints"],
                              args.device)
                rows.append(row)
                results.append({"ablation": key, "value": val,
                               "image_idx": s["idx"], **row})
            agg = _agg(rows)
            val_str = str(val)
            print(f"{key:<20} {val_str:>12}  "
                  f"{agg['mpjpe']:>8.1f}  {agg['pa_mpjpe']:>8.1f}  "
                  f"{agg['pelvis_mpjpe']:>9.1f}  {agg['pelvis_pa_mpjpe']:>7.1f}  "
                  f"{agg['reproj']:>7.2f}  {agg['pelvis_reproj']:>7.2f}  "
                  f"{agg['time_s']:>6.1f}  "
                  f"{agg['betas_norm']:>6.2f}  {agg['pose_norm']:>6.2f}  "
                  f"{agg['n_ok']:>5}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(args.out_dir, "ablation_all.csv"), index=False)

    # Aggregate table
    agg_rows = []
    for (ablation, value), grp in df_all.groupby(["ablation", "value"]):
        agg_rows.append(dict(
            ablation=ablation, value=str(value),
            mpjpe_mean=float(grp["mpjpe"].mean()),
            mpjpe_std=float(grp["mpjpe"].std()),
            pa_mpjpe_mean=float(grp["pa_mpjpe"].mean()),
            pa_mpjpe_std=float(grp["pa_mpjpe"].std()),
            pelvis_mpjpe_mean=float(grp["pelvis_mpjpe"].mean()),
            pelvis_mpjpe_std=float(grp["pelvis_mpjpe"].std()),
            pelvis_pa_mpjpe_mean=float(grp["pelvis_pa_mpjpe"].mean()),
            pelvis_pa_mpjpe_std=float(grp["pelvis_pa_mpjpe"].std()),
            reproj_mean=float(grp["reproj"].mean()),
            reproj_std=float(grp["reproj"].std()),
            pelvis_reproj_mean=float(grp["pelvis_reproj"].mean()),
            pelvis_reproj_std=float(grp["pelvis_reproj"].std()),
            time_mean=float(grp["time_s"].mean()),
            n_ok=int(grp["ok"].sum()),
        ))
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    print(f"\nSaved -> {args.out_dir}/ablation_all.csv "
          f"({len(df_all)} rows)")
    print(f"Saved -> {args.out_dir}/ablation_summary.csv "
          f"({len(df_agg)} rows)")

    # JSON for reproducibility
    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w") as f:
        json.dump({"defaults": DEFAULTS,
                   "ablations": {k: list(v) for k, v in ABLATIONS.items()},
                   "summary": agg_rows}, f, indent=2)


if __name__ == "__main__":
    main()
