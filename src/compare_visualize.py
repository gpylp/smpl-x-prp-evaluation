"""Side-by-side comparison: Method A (raw pose) vs Method B (VPoser).

For each AGORA sample we produce a single figure with:
    Row 0 (3 panels):
        - GT 2D skeleton    (clean projection of AGORA joints)
        - Method A reproj   (raw axis-angle optimiser)
        - Method B reproj   (VPoser prior)
    Row 1 (3 panels):
        - GT 3D joints (from the model T-pose using GT transl)
        - Method A 3D mesh (+ skeleton) -- one viewpoint
        - Method B 3D mesh (+ skeleton) -- same viewpoint
    Stats column: per-method metrics + deltas.

Output: PNG files in --out-dir, one per sample, plus a montage.

Usage
-----
    python src/compare_visualize.py --num-images 6 --out-dir output_compare_viz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
                  pelvis_mpjpe, pelvis_pa_mpjpe,
                  reprojection_error_px,
                  pelvis_reprojection_error_px)
from recon_2d import build_smplx_model, fit_smplx_to_2d, Camera
from compare_fitters import fit_smplx_with_vposer, load_vposer_v1


# 22 SMPL-X body joints (matches SMPLX_BODY_JOINT_NAMES in recon_2d.py)
SKELETON_PAIRS = [
    (0, 1), (1, 4), (4, 7),
    (0, 2), (2, 5), (5, 8),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 15),
    (9, 16), (16, 18), (18, 20),
    (9, 17), (17, 19), (19, 21),
    (12, 13), (13, 16),
    (12, 14), (14, 17),
]


def render_2d(ax, kp_obs, kp_pred, title, reproj=None, color_pred="tab:red",
              show_legend=True):
    ax.scatter(kp_obs[:, 0], kp_obs[:, 1], c="tab:blue", s=55,
               edgecolors="navy", linewidths=1.0, zorder=3,
               label="ground truth" if show_legend else None)
    if kp_pred is not None and len(kp_pred) > 0:
        ax.scatter(kp_pred[:, 0], kp_pred[:, 1], c=color_pred, s=55,
                   marker="x", linewidths=2.0, zorder=4,
                   label="predicted" if show_legend else None)
    for i, j in SKELETON_PAIRS:
        if i < len(kp_obs) and j < len(kp_obs):
            ax.plot([kp_obs[i, 0], kp_obs[j, 0]],
                    [kp_obs[i, 1], kp_obs[j, 1]],
                    c="tab:blue", alpha=0.4, lw=2)
        if kp_pred is not None and len(kp_pred) > 0 \
                and i < len(kp_pred) and j < len(kp_pred):
            ax.plot([kp_pred[i, 0], kp_pred[j, 0]],
                    [kp_pred[i, 1], kp_pred[j, 1]],
                    c=color_pred, alpha=0.55, lw=1.5, ls="--")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    if reproj is not None:
        ax.text(0.03, 0.95, f"reproj {reproj:.1f} px",
                transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85, edgecolor="0.7"))
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


def render_mesh(ax, vertices, faces, joints_3d, title, color,
                faces_per_render=8000, view=(10, -90),
                view_label="front"):
    if faces.shape[0] > faces_per_render:
        step = max(1, faces.shape[0] // faces_per_render)
        faces = faces[::step]
    v = vertices[faces]
    mesh = Poly3DCollection(v, alpha=0.85, facecolor=color,
                           edgecolor="black", linewidth=0.04)
    ax.add_collection3d(mesh)
    span = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    centre = vertices.mean(axis=0)
    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2],
               c="black", s=18, depthshade=False)
    for i, j in SKELETON_PAIRS:
        if i < len(joints_3d) and j < len(joints_3d):
            ax.plot([joints_3d[i, 0], joints_3d[j, 0]],
                    [joints_3d[i, 1], joints_3d[j, 1]],
                    [joints_3d[i, 2], joints_3d[j, 2]],
                    c="black", lw=1.3)
    ax.set_title(f"{title}\n({view_label} view)", fontsize=10,
                 fontweight="bold")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", default="dataset/agora/camera_meta/val")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--vposer-dir",
                   default=r"models\vposer\v02_05\snapshots\vposer_v1_0\vposer_v1_0")
    p.add_argument("--out-dir", default="output_compare_viz")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--num-steps", type=int, default=120)
    p.add_argument("--noise-std", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, nargs=2, default=[720, 1280])
    p.add_argument("--log-path", default="terminal_compare_viz.txt")
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

    print(f"[2/3] Loading SMPL-X neutral + VPoser v1.0...")
    model = build_smplx_model(args.model_dir, gender="neutral")
    vposer_dir = (args.vposer_dir
                  if os.path.isabs(args.vposer_dir)
                  else os.path.join(_ROOT, args.vposer_dir))
    vposer, _ = load_vposer_v1(vposer_dir, device=args.device)

    rng = np.random.RandomState(args.seed)
    image_size = tuple(args.image_size)
    print(f"[3/3] Comparing {args.num_images} sample(s)...\n")

    metrics = []
    for i in range(args.num_images):
        try:
            info = get_first_person_row(df, i)
        except Exception as exc:
            print(f"  sample {i}: skipped ({exc})")
            continue

        joints_gt = info["joints_3d"][:22]
        if joints_gt.shape[0] < 6:
            print(f"  sample {i}: only {joints_gt.shape[0]} joints, skip")
            continue

        kp_clean = project_3d_to_2d(joints_gt, image_size=image_size)
        kp_noisy = kp_clean + rng.normal(0, args.noise_std,
                                          kp_clean.shape).astype(np.float32)
        if kp_noisy.shape[0] < 22:
            padded = np.full((22, 2), np.nan, dtype=np.float32)
            padded[:kp_noisy.shape[0]] = kp_noisy
            kp_noisy = padded

        # ----- Method A -----
        t0 = time.time()
        try:
            res_a = fit_smplx_to_2d(model, kp_noisy,
                                    num_steps=args.num_steps,
                                    lr=0.05, lambda_shape=1e-3,
                                    lambda_pose=1e-3,
                                    device=args.device, verbose=False)
            ok_a = True
        except Exception as exc:
            print(f"  [A] row {i} FAILED: {exc}")
            ok_a = False
        t_a = time.time() - t0

        # ----- Method B -----
        t0 = time.time()
        try:
            res_b = fit_smplx_with_vposer(model, vposer, kp_noisy,
                                          num_steps=args.num_steps,
                                          lr=0.05,
                                          lambda_shape=1e-2,
                                          lambda_kl=1e-3,
                                          lambda_reproj=1.0,
                                          device=args.device, verbose=False)
            ok_b = True
        except Exception as exc:
            print(f"  [B] row {i} FAILED: {exc}")
            ok_b = False
        t_b = time.time() - t0

        if not (ok_a and ok_b):
            continue

        # ---- metrics ----
        n_joints = joints_gt.shape[0]
        j3d_a = res_a["joints_3d"][:n_joints]
        j3d_b = res_b["joints_3d"][:n_joints]
        cam_a, cam_b = res_a["camera"], res_b["camera"]
        proj_a = cam_a.project(j3d_a)
        proj_b = cam_b.project(j3d_b)
        # Evaluate against the clean 2D projection so the reproj error
        # reflects pose quality, not noise.
        kp_clean_for_eval = kp_clean[:n_joints]
        rep_a = reprojection_error_px(proj_a, kp_clean_for_eval)
        rep_b = reprojection_error_px(proj_b, kp_clean_for_eval)
        pel_rep_a = pelvis_reprojection_error_px(proj_a, kp_clean_for_eval)
        pel_rep_b = pelvis_reprojection_error_px(proj_b, kp_clean_for_eval)
        mpjpe_a = mpjpe(j3d_a, joints_gt); pa_a = pa_mpjpe(j3d_a, joints_gt)
        mpjpe_b = mpjpe(j3d_b, joints_gt); pa_b = pa_mpjpe(j3d_b, joints_gt)
        pel_a = pelvis_mpjpe(j3d_a, joints_gt)
        pel_b = pelvis_mpjpe(j3d_b, joints_gt)
        pel_pa_a = pelvis_pa_mpjpe(j3d_a, joints_gt)
        pel_pa_b = pelvis_pa_mpjpe(j3d_b, joints_gt)

        # ---- figure ----
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.9],
                               height_ratios=[1, 1.3],
                               hspace=0.20, wspace=0.18)
        ax_gt2d   = fig.add_subplot(gs[0, 0])
        ax_a2d    = fig.add_subplot(gs[0, 1])
        ax_b2d    = fig.add_subplot(gs[0, 2])
        ax_stat   = fig.add_subplot(gs[0, 3])
        ax_a3d    = fig.add_subplot(gs[1, 0], projection="3d")
        ax_b3d    = fig.add_subplot(gs[1, 1], projection="3d")
        ax_extra  = fig.add_subplot(gs[1, 2:])
        ax_extra.axis("off")

        render_2d(ax_gt2d, kp_clean[:22], None,
                  "GT 2D (clean)", show_legend=True)
        render_2d(ax_a2d, kp_clean[:22], proj_a,
                  f"Method A — raw pose", reproj=rep_a,
                  color_pred="tab:red", show_legend=False)
        render_2d(ax_b2d, kp_clean[:22], proj_b,
                  f"Method B — VPoser", reproj=rep_b,
                  color_pred="tab:green", show_legend=False)

        # Stats panel
        rows_text = [
            f"Sample: {os.path.basename(info['img_path'])}",
            f"Gender: {info.get('gender', '?')}",
            "",
            f"{'Metric':<15}{'A raw':>10}{'B VPoser':>12}{'Δ (B-A)':>12}",
            "-" * 51,
            f"{'MPJPE (mm)':<15}{mpjpe_a:>10.1f}{mpjpe_b:>12.1f}"
            f"{mpjpe_b - mpjpe_a:+11.1f}",
            f"{'PA-MPJPE (mm)':<15}{pa_a:>10.1f}{pa_b:>12.1f}"
            f"{pa_b - pa_a:+11.1f}",
            f"{'Pel-MPJPE':<15}{pel_a:>10.1f}{pel_b:>12.1f}"
            f"{pel_b - pel_a:+11.1f}",
            f"{'Pel-PA-MPJPE':<15}{pel_pa_a:>10.1f}{pel_pa_b:>12.1f}"
            f"{pel_pa_b - pel_pa_a:+11.1f}",
            f"{'Reproj (px)':<15}{rep_a:>10.1f}{rep_b:>12.1f}"
            f"{rep_b - rep_a:+11.1f}",
            f"{'Pel-Reproj':<15}{pel_rep_a:>10.1f}{pel_rep_b:>12.1f}"
            f"{pel_rep_b - pel_rep_a:+11.1f}",
            f"{'|β|':<15}{np.linalg.norm(res_a['betas']):>10.2f}"
            f"{np.linalg.norm(res_b['betas']):>12.2f}",
            f"{'|pose|':<15}{np.linalg.norm(res_a['body_pose']):>10.2f}"
            f"{np.linalg.norm(res_b['body_pose']):>12.2f}",
            f"{'|z|':<15}{'-':>10}"
            f"{np.linalg.norm(res_b.get('pose_z', [0])):>12.2f}",
            f"{'time (s)':<15}{t_a:>10.2f}{t_b:>12.2f}{t_b - t_a:+11.2f}",
        ]
        ax_stat.text(0.04, 0.95, "\n".join(rows_text),
                     transform=ax_stat.transAxes,
                     family="monospace", fontsize=9,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.6",
                               facecolor="lightyellow",
                               edgecolor="0.6", alpha=0.95))
        ax_stat.axis("off")
        ax_stat.set_title("Metrics comparison", fontsize=10, fontweight="bold")

        faces = (model.faces if isinstance(model.faces, np.ndarray)
                 else np.asarray(model.faces))
        render_mesh(ax_a3d, res_a["vertices"], faces, j3d_a,
                    "Method A (raw pose)", color="lightsalmon",
                    view=(10, -90), view_label="front")
        render_mesh(ax_b3d, res_b["vertices"], faces, j3d_b,
                    "Method B (VPoser)", color="lightseagreen",
                    view=(10, -90), view_label="front")

        # Visual delta commentary
        delta_text = (
            f"{'Δ PA-MPJPE':>12} {pa_b - pa_a:+7.1f} mm\n"
            f"{'Δ Pel-PA-MPJPE':>12} {pel_pa_b - pel_pa_a:+7.1f} mm\n"
            f"{'Δ Pel-Reproj':>12} {pel_rep_b - pel_rep_a:+7.1f} px\n"
            f"{'Δ |pose|':>12} {np.linalg.norm(res_b['body_pose']) - np.linalg.norm(res_a['body_pose']):+7.2f}\n"
            f"{'Δ Time':>12} {t_b - t_a:+7.1f} s\n"
        )
        better = (
            "VPoser is BETTER on pose-prior quality" if pel_pa_b < pel_pa_a
            else "Raw-pose is BETTER on pose-prior quality"
        )
        ax_extra.text(0.05, 0.55, delta_text,
                      transform=ax_extra.transAxes,
                      family="monospace", fontsize=12,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round,pad=0.5",
                                facecolor="whitesmoke",
                                edgecolor="0.6", alpha=0.95))
        ax_extra.text(0.05, 0.20, better,
                      transform=ax_extra.transAxes,
                      fontsize=12, fontweight="bold",
                      color="darkgreen" if pel_pa_b < pel_pa_a else "darkred",
                      bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="honeydew" if pel_pa_b < pel_pa_a
                                else "mistyrose",
                                edgecolor="0.6"))

        fig.suptitle(
            f"Qualitative comparison: 2D-driven SMPL-X — raw pose vs VPoser",
            fontsize=13, fontweight="bold", y=0.995)
        out_path = os.path.join(args.out_dir, f"compare_{i:03d}.png")
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

        metrics.append(dict(name=os.path.basename(info["img_path"]),
                             mpjpe_a=mpjpe_a, pa_a=pa_a,
                             pel_a=pel_a, pel_pa_a=pel_pa_a,
                             rep_a=rep_a, pel_rep_a=pel_rep_a, t_a=t_a,
                             mpjpe_b=mpjpe_b, pa_b=pa_b,
                             pel_b=pel_b, pel_pa_b=pel_pa_b,
                             rep_b=rep_b, pel_rep_b=pel_rep_b, t_b=t_b))
        print(f"  sample {i:03d}: "
              f"A PelPA={pel_pa_a:6.1f}mm  B PelPA={pel_pa_b:6.1f}mm  "
              f"Δ={pel_pa_b-pel_pa_a:+6.1f}mm  -> {out_path}")

    # ---------------- montage ----------------
    if metrics:
        n = len(metrics)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18 * cols, 9 * rows))
        axes = np.atleast_2d(axes)
        for k, m in enumerate(metrics):
            ax = axes[k // cols, k % cols]
            img = plt.imread(os.path.join(args.out_dir,
                                            f"compare_{k:03d}.png"))
            ax.imshow(img)
            ax.set_title(
                f"#{k}: Δ PA-MPJPE {m['pa_b']-m['pa_a']:+.1f} mm  "
                f"({m['name']})", fontsize=12)
            ax.axis("off")
        for k in range(n, rows * cols):
            axes[k // cols, k % cols].axis("off")
        fig.tight_layout()
        montage_path = os.path.join(args.out_dir, "montage.png")
        fig.savefig(montage_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"\nMontage -> {montage_path}")

    # ---------------- summary csv ----------------
    pd.DataFrame(metrics).to_csv(
        os.path.join(args.out_dir, "compare_viz_metrics.csv"), index=False)
    print(f"Metrics -> {args.out_dir}/compare_viz_metrics.csv")


if __name__ == "__main__":
    main()
