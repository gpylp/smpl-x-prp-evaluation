"""Visualise 2D-driven SMPL-X reconstruction.

For each AGORA sample we produce three side-by-side panels:
    1. Observed 2D keypoints (synthesised by weak-perspective projection of GT).
    2. Projected skeleton from the fitted SMPL-X parameters.
    3. 3D mesh rendering (two views: front + side).

Output: PNG files in --out-dir, one per sample, plus a grid montage.

Usage
-----
    python src/visualize.py --num-images 6 --out-dir output_viz
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from eval import (load_agora_metadata, get_first_person_row,
                  project_3d_to_2d, mpjpe, pa_mpjpe,
                  reprojection_error_px)
from recon_2d import build_smplx_model, fit_smplx_to_2d, Camera


# Skeleton pairs for the 22 SMPL-X body joints (pelvis-rooted topology).
# Indices match SMPLX_BODY_JOINT_NAMES in recon_2d.py.
SKELETON_PAIRS = [
    (0, 1), (1, 4), (4, 7),                    # pelvis -> L hip -> L knee -> L ankle
    (0, 2), (2, 5), (5, 8),                    # pelvis -> R hip -> R knee -> R ankle
    (0, 3), (3, 6), (6, 9), (9, 12),           # pelvis -> spine -> neck
    (12, 15),                                  # neck -> head
    (9, 16), (16, 18), (18, 20),               # neck -> L shoulder -> L elbow -> L wrist
    (9, 17), (17, 19), (19, 21),               # neck -> R shoulder -> R elbow -> R wrist
    (12, 13), (13, 16),                        # collar -> L shoulder
    (12, 14), (14, 17),                        # collar -> R shoulder
]


def render_2d_overlay(ax, kp_obs: np.ndarray, kp_pred: np.ndarray,
                      title: str = "", reproj: float = None) -> None:
    """Overlay observed (blue) and projected (red) 2D keypoints."""
    ax.scatter(kp_obs[:, 0], kp_obs[:, 1], c="tab:blue", s=70,
               label="observed (clean)", edgecolors="navy", linewidths=1.0,
               zorder=3)
    if kp_pred is not None and len(kp_pred) > 0:
        ax.scatter(kp_pred[:, 0], kp_pred[:, 1], c="tab:red", s=70,
                   marker="x", label="predicted", linewidths=2.0, zorder=4)
    for i, j in SKELETON_PAIRS:
        if i < len(kp_obs) and j < len(kp_obs):
            ax.plot([kp_obs[i, 0], kp_obs[j, 0]],
                    [kp_obs[i, 1], kp_obs[j, 1]],
                    c="tab:blue", alpha=0.5, lw=2.5)
        if kp_pred is not None and len(kp_pred) > 0 \
                and i < len(kp_pred) and j < len(kp_pred):
            ax.plot([kp_pred[i, 0], kp_pred[j, 0]],
                    [kp_pred[i, 1], kp_pred[j, 1]],
                    c="tab:red", alpha=0.5, lw=1.8, ls="--")
    ax.invert_yaxis()                 # image coordinates
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    if reproj is not None:
        ax.text(0.02, 0.96, f"reproj {reproj:.1f} px",
                transform=ax.transAxes, fontsize=10,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


def render_mesh(ax, vertices: np.ndarray, faces: np.ndarray,
                faces_per_render: int = 8000,
                view: tuple = (15, -75),
                title: str = "", color="lightsalmon",
                plot_joints: bool = False, joints_3d: np.ndarray = None) -> None:
    """Render SMPL-X mesh (subsampled) on a 3D axis with optional skeleton."""
    if faces.shape[0] > faces_per_render:
        step = max(1, faces.shape[0] // faces_per_render)
        faces = faces[::step]
    v = vertices[faces]                # (F, 3, 3)
    mesh = Poly3DCollection(v, alpha=0.85, facecolor=color,
                           edgecolor="darkred", linewidth=0.05)
    ax.add_collection3d(mesh)
    span = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    centre = vertices.mean(axis=0)
    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if plot_joints and joints_3d is not None:
        ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2],
                   c="black", s=20, depthshade=False)
        for i, j in SKELETON_PAIRS:
            if i < len(joints_3d) and j < len(joints_3d):
                ax.plot([joints_3d[i, 0], joints_3d[j, 0]],
                        [joints_3d[i, 1], joints_3d[j, 1]],
                        [joints_3d[i, 2], joints_3d[j, 2]],
                        c="black", lw=1.5)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def visualise_sample(model, info: dict, image_size: tuple[int, int],
                     num_steps: int, device: str, rng: np.random.RandomState,
                     noise_std: float = 2.5) -> Optional[dict]:
    """Reconstruct one image, return figure + metrics (or None on failure)."""
    joints_gt = info["joints_3d"][:22]
    if joints_gt.shape[0] < 6:
        return None
    kp_clean = project_3d_to_2d(joints_gt, image_size=image_size)
    kp_noisy = kp_clean + rng.normal(0, noise_std, kp_clean.shape).astype(np.float32)
    if kp_noisy.shape[0] < 22:
        padded = np.full((22, 2), np.nan, dtype=np.float32)
        padded[:kp_noisy.shape[0]] = kp_noisy
        kp_noisy = padded

    try:
        res = fit_smplx_to_2d(model, kp_noisy, num_steps=num_steps,
                              device=device, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  fit failed: {exc}")
        return None

    j3d_pred = res["joints_3d"][:joints_gt.shape[0]]
    cam = res["camera"]
    kp_proj = cam.project(j3d_pred)
    err_reproj = reprojection_error_px(kp_proj, kp_clean[:kp_proj.shape[0]])
    err_mpjpe = mpjpe(j3d_pred, joints_gt)
    err_pa = pa_mpjpe(j3d_pred, joints_gt)

    # Build figure: 2 row x 3 col layout
    # Row 0: clean 2D | noisy 2D | projected 2D
    # Row 1: 3D front view | 3D side view | (stats panel)
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1.4],
                           hspace=0.25, wspace=0.15)
    ax_clean = fig.add_subplot(gs[0, 0])
    ax_noisy = fig.add_subplot(gs[0, 1])
    ax_proj  = fig.add_subplot(gs[0, 2])
    ax3d_f   = fig.add_subplot(gs[1, 0], projection="3d")
    ax3d_s   = fig.add_subplot(gs[1, 1], projection="3d")
    ax_stat  = fig.add_subplot(gs[1, 2])
    ax_stat.axis("off")

    render_2d_overlay(ax_clean, kp_clean[:22], kp_proj,
                      "Ground-truth (clean)", reproj=err_reproj)
    render_2d_overlay(ax_noisy, kp_clean[:22], kp_noisy[:22],
                      "Noisy input")
    render_2d_overlay(ax_proj, kp_clean[:22], kp_proj,
                      "Predicted projection")

    faces = model.faces if isinstance(model.faces, np.ndarray) else np.asarray(model.faces)
    render_mesh(ax3d_f, res["vertices"], faces, view=(10, -90),
                title=f"Front view", plot_joints=True,
                joints_3d=j3d_pred)
    render_mesh(ax3d_s, res["vertices"], faces, view=(10, 0),
                title=f"Side view", plot_joints=True,
                joints_3d=j3d_pred)

    # Stats panel ---------------------------------------------------
    stats_lines = [
        f"Sample: {os.path.basename(info['img_path'])}",
        f"Gender: {info.get('gender', '?')}",
        "",
        f"Reproj:    {err_reproj:6.1f} px",
        f"MPJPE:     {err_mpjpe:6.1f} mm",
        f"PA-MPJPE:  {err_pa:6.1f} mm",
        f"|β|:       {np.linalg.norm(res['betas']):5.2f}",
        f"|pose|:    {np.linalg.norm(res['body_pose']):5.2f}",
        "",
        "2D noise: ±2.5 px Gaussian",
        "Optim:    Adam, 200 steps",
    ]
    ax_stat.text(0.05, 0.95, "\n".join(stats_lines),
                 transform=ax_stat.transAxes,
                 fontsize=11, family="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor="lightyellow",
                           edgecolor="0.6", alpha=0.9))
    fig.suptitle(f"2D-driven SMPL-X reconstruction", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return dict(fig=fig, mpjpe=err_mpjpe, pa_mpjpe=err_pa,
                reproj=err_reproj, name=info["img_path"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", default="dataset/agora/camera_meta/val")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out-dir", default="output_viz")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, nargs=2, default=[720, 1280])
    p.add_argument("--log-path", default="terminal_viz.txt")
    return p.parse_args()


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        for s in self.streams:
            s.write(d)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_fh = open(args.log_path, "w", encoding="utf-8")
    _stdout = sys.stdout
    sys.stdout = _TeeWriter(_stdout, log_fh)
    try:
        _main(args)
    finally:
        sys.stdout = _stdout
        log_fh.close()


def _main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    pkl_files = sorted(
        os.path.join(args.meta_dir, f)
        for f in os.listdir(args.meta_dir) if f.endswith(".pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl in {args.meta_dir}")
    print(f"[1/3] Loading AGORA ({len(pkl_files)} files)...")
    df = pd.concat([load_agora_metadata(p) for p in pkl_files],
                   ignore_index=True)

    print(f"[2/3] Loading SMPL-X neutral from {args.model_dir}...")
    model = build_smplx_model(args.model_dir, gender="neutral")

    image_size = tuple(args.image_size)
    rng = np.random.RandomState(args.seed)
    print(f"[3/3] Visualising {args.num_images} sample(s)...")
    metrics = []
    for i in range(args.num_images):
        try:
            info = get_first_person_row(df, i)
        except Exception as exc:
            print(f"  sample {i}: skipped ({exc})")
            continue
        out = visualise_sample(model, info, image_size, args.num_steps,
                                args.device, rng)
        if out is None:
            continue
        name = os.path.basename(info["img_path"])
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.png")
        out["fig"].savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(out["fig"])
        metrics.append(dict(name=name, mpjpe=out["mpjpe"],
                             pa_mpjpe=out["pa_mpjpe"],
                             reproj=out["reproj"]))
        print(f"  sample {i:03d}: MPJPE {out['mpjpe']:.1f} mm, "
              f"PA-MPJPE {out['pa_mpjpe']:.1f} mm, "
              f"reproj {out['reproj']:.1f} px -> {out_path}")

    # ---------------- montage ----------------
    if metrics:
        n = len(metrics)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 7 * rows))
        axes = np.atleast_2d(axes)
        for k, m in enumerate(metrics):
            ax = axes[k // cols, k % cols]
            img = plt.imread(os.path.join(args.out_dir,
                                            f"sample_{k:03d}.png"))
            ax.imshow(img)
            ax.set_title(f"#{k}: PA-MPJPE {m['pa_mpjpe']:.1f} mm "
                         f"({m['name']})", fontsize=11)
            ax.axis("off")
        for k in range(n, rows * cols):
            axes[k // cols, k % cols].axis("off")
        fig.tight_layout()
        montage_path = os.path.join(args.out_dir, "montage.png")
        fig.savefig(montage_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\nMontage -> {montage_path}")

    pd.DataFrame(metrics).to_csv(
        os.path.join(args.out_dir, "viz_metrics.csv"), index=False)
    print(f"Metrics -> {args.out_dir}/viz_metrics.csv")


if __name__ == "__main__":
    main()
