"""Side-by-side comparison of two SMPL-X fitters on the AGORA benchmark.

Method A — *2D-driven lightweight fitter* (existing ``recon_2d.fit_smplx_to_2d``)
    Optimises raw ``body_pose`` (axis-angle) directly against reprojection
    loss + L2 shape / pose regularisers.  Fast (~10 s / frame on CPU).

Method B — *SMPLify-X-style fitter with VPoser* (this script)
    Optimises a 32-D latent ``z`` decoded by VPoser + L2 pose prior + VPoser
    KL-divergence regulariser + shape prior.  Closer to the original
    SMPLify-X pipeline (Pavlakos et al., CVPR'19) and produces more
    anatomically plausible poses.

Both methods receive identical inputs (a noisy 2D keypoint cloud) and
are evaluated against the same AGORA 3D ground truth, so the
difference in MPJPE / PA-MPJPE / Reproj reflects only the fitter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Make ``src`` importable when invoked as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from eval import (load_agora_metadata, get_first_person_row,
                  project_3d_to_2d, mpjpe, pa_mpjpe, pelvis_mpjpe,
                  pelvis_pa_mpjpe, reprojection_error_px,
                  pelvis_reprojection_error_px, evaluate)
from recon_2d import build_smplx_model, fit_smplx_to_2d, Camera
from vposer_loader import load_vposer_v1


# ---------------------------------------------------------------------------
# Method B: SMPLify-X-style fitter using VPoser latent z
# ---------------------------------------------------------------------------

def _project_weak(joints_3d: torch.Tensor, scale: torch.Tensor,
                  rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Weak-perspective projection ``p_2d = scale * (R @ J_3d) + t``."""
    return (joints_3d @ rot.T) * scale + trans


def fit_smplx_with_vposer(model: torch.nn.Module,
                          vposer: torch.nn.Module,
                          keypoints_2d: np.ndarray,
                          num_steps: int = 150,
                          lr: float = 0.05,
                          lambda_shape: float = 1e-2,
                          lambda_kl: float = 1e-3,
                          lambda_reproj: float = 1.0,
                          focal_length: float = 1000.0,
                          image_size: tuple[int, int] = (720, 1280),
                          device: str = "cpu",
                          verbose: bool = False) -> dict:
    """Fit SMPL-X using the VPoser pose prior.

    The latent body pose is parameterised by a 32-D vector ``z`` that
    decodes through ``vposer.decode(z)`` into 21 joint axis-angle
    rotations (SMPL-X has 21 body joints).  Hands and face are left at
    their neutral values to keep the search tractable.

    Returns the same dictionary shape as :func:`recon_2d.fit_smplx_to_2d`
    plus ``pose_z`` (the learned latent).
    """
    model = model.to(device)
    vposer = vposer.to(device)
    vposer.eval()
    for p in vposer.parameters():
        p.requires_grad = False

    # Initialise parameters --------------------------------------------------
    betas = torch.zeros(1, model.num_betas, device=device, requires_grad=True)
    z = torch.zeros(1, vposer.latentD, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([betas, z, global_orient, transl], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(num_steps // 3, 1), gamma=0.5)

    # Camera guess from T-pose joints -----------------------------------------
    with torch.no_grad():
        # Decode a zero latent to get the T-pose body
        body_pose_aa = vposer.decode(z, output_type='aa')  # (1, 1, 21, 3)
        body_pose_flat = body_pose_aa.view(1, -1)
        out0 = model(betas=betas, body_pose=body_pose_flat,
                     global_orient=global_orient, transl=transl)
    joints0 = out0.joints[0].detach().cpu().numpy()[:, :3]
    cam = Camera(scale=1.0, rot=np.eye(3)[:2], trans=np.array(image_size) / 2.0)
    cam = _refine_camera_from_pts(cam, keypoints_2d, joints0,
                                   focal_length=focal_length,
                                   image_size=image_size)

    valid_mask = ~np.isnan(keypoints_2d).any(axis=1)
    n_use = min(22, int(valid_mask.sum()))
    joint_idx = np.where(valid_mask)[0][:n_use]
    kp_2d_t = torch.tensor(keypoints_2d[joint_idx],
                           dtype=torch.float32, device=device)
    joint_idx_t = torch.tensor(joint_idx, dtype=torch.long, device=device)

    cam_scale = torch.tensor(float(cam.scale), dtype=torch.float32,
                             device=device, requires_grad=True)
    cam_trans = torch.tensor(cam.trans.astype(np.float32),
                             dtype=torch.float32, device=device,
                             requires_grad=True)
    rot_mat = torch.tensor(cam.rot.astype(np.float32),
                           dtype=torch.float32, device=device)
    optimizer.add_param_group({"params": [cam_scale, cam_trans], "lr": lr * 0.5})

    loss_hist = []
    for step in range(num_steps):
        optimizer.zero_grad()
        # Decode latent → body pose
        body_pose_aa = vposer.decode(z, output_type='aa')  # (1, 1, 21, 3)
        body_pose_flat = body_pose_aa.view(1, -1)
        out = model(betas=betas, body_pose=body_pose_flat,
                    global_orient=global_orient, transl=transl)
        joints_3d = out.joints[0, joint_idx_t]

        proj_2d = _project_weak(joints_3d, cam_scale, rot_mat, cam_trans)
        reproj_loss = ((proj_2d - kp_2d_t) ** 2).sum(dim=-1).mean()

        # VPoser KL: encode a near-zero pose and read its distribution.
        # We approximate KL(z || N(0,I)) using the prior N(0, softplus(0)).
        with torch.no_grad():
            # VPoser's encode returns a Normal; at z=0 the posterior mean
            # should be ~0.  We compute the KL analytically:
            posterior = vposer.encode(body_pose_flat.detach())
            # KL(q(z|x) || N(0,I)) closed-form for diagonal Normal:
            mu, sigma = posterior.mean, posterior.scale
            kl = (sigma ** 2 + mu ** 2 - 1.0 - 2.0 * torch.log(sigma + 1e-8)).sum() * 0.5
            kl_val = float(kl.item())

        shape_reg = (betas ** 2).mean()
        loss = (lambda_reproj * reproj_loss
                + lambda_shape * shape_reg
                + lambda_kl * kl_val)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(float(reproj_loss.detach().cpu()))
        if verbose and (step + 1) % max(num_steps // 5, 1) == 0:
            print(f"  step {step + 1}/{num_steps}  "
                  f"reproj={reproj_loss.item():.3f}  kl={kl_val:.3f}")

    with torch.no_grad():
        body_pose_aa = vposer.decode(z, output_type='aa')
        body_pose_flat = body_pose_aa.view(1, -1)
        out = model(betas=betas, body_pose=body_pose_flat,
                    global_orient=global_orient, transl=transl)

    return {
        "vertices": out.vertices[0].detach().cpu().numpy(),
        "joints_3d": out.joints[0].detach().cpu().numpy()[:, :3],
        "camera": Camera(scale=float(cam_scale.item()),
                         rot=cam.rot.copy(),
                         trans=cam_trans.detach().cpu().numpy().copy()),
        "betas": betas.detach().cpu().numpy().reshape(-1),
        "body_pose": body_pose_flat.detach().cpu().numpy().reshape(-1),
        "global_orient": global_orient.detach().cpu().numpy().reshape(-1),
        "transl": transl.detach().cpu().numpy().reshape(-1),
        "pose_z": z.detach().cpu().numpy().reshape(-1),
        "loss_hist": loss_hist,
    }


def _refine_camera_from_pts(cam: Camera,
                            keypoints_2d: np.ndarray,
                            joints_3d_init: np.ndarray,
                            focal_length: float = 1000.0,
                            image_size: tuple[int, int] = (720, 1280)) -> Camera:
    """Re-implementation of :func:`recon_2d.guess_camera_from_keypoints`.

    Inlined here to avoid importing the original module's helper into the
    VPoser path (kept self-contained for clarity).
    """
    n = min(len(keypoints_2d), len(joints_3d_init))
    valid_2d = ~np.isnan(keypoints_2d[:n]).any(axis=1)
    if valid_2d.sum() < 2:
        return cam
    pts2d = keypoints_2d[:n][valid_2d]
    pts3d = joints_3d_init[:n][valid_2d]
    bbox_size_2d = max(pts2d[:, 0].max() - pts2d[:, 0].min(),
                       pts2d[:, 1].max() - pts2d[:, 1].min()) + 1e-6
    bbox_size_3d = max(pts3d[:, 0].max() - pts3d[:, 0].min(),
                       pts3d[:, 1].max() - pts3d[:, 1].min(),
                       pts3d[:, 2].max() - pts3d[:, 2].min()) + 1e-6
    scale = float(focal_length / max(bbox_size_3d, 1e-6) * bbox_size_2d / max(bbox_size_3d, 1e-6))
    scale = float(np.clip(scale, 100.0, 5000.0))
    rot = np.eye(3)[:2]
    trans = np.array([image_size[1] / 2.0, image_size[0] / 2.0])
    return Camera(scale=scale, rot=rot, trans=trans)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", default="dataset/agora/camera_meta/val",
                   help="Directory containing AGORA camera_meta pkl files")
    p.add_argument("--model-dir", default="models",
                   help="Directory containing SMPL-X npz models")
    p.add_argument("--vposer-dir",
                   default=r"models\vposer\v02_05\snapshots\vposer_v1_0\vposer_v1_0",
                   help="Directory containing VPoser v1.0 model + config")
    p.add_argument("--out-dir", default="output_compare",
                   help="Output directory")
    p.add_argument("--num-images", type=int, default=10,
                   help="Number of images to process")
    p.add_argument("--num-steps", type=int, default=120,
                   help="Optimisation steps for each method")
    p.add_argument("--noise-std", type=float, default=2.5,
                   help="Std-dev (px) of Gaussian noise added to 2D keypoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load AGORA + models
    # ------------------------------------------------------------------
    pkl_files = sorted(
        [os.path.join(args.meta_dir, f)
         for f in os.listdir(args.meta_dir) if f.endswith(".pkl")]
    )
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files in {args.meta_dir}")
    print(f"[1/5] Loading AGORA metadata ({len(pkl_files)} file(s))...")
    df = pd.concat([load_agora_metadata(p) for p in pkl_files], ignore_index=True)

    print(f"[2/5] Loading SMPL-X (neutral) from {args.model_dir}...")
    model = build_smplx_model(args.model_dir, gender="neutral")

    vposer_dir = args.vposer_dir
    if not os.path.isabs(vposer_dir):
        vposer_dir = os.path.join(_ROOT, vposer_dir)
    print(f"[3/5] Loading VPoser v1.0 from {vposer_dir}...")
    vposer, vposer_meta = load_vposer_v1(vposer_dir, device=args.device)
    print(f"       latentD={vposer_meta['latentD']}  num_joints={vposer_meta['data_shape'][1]}")

    sample_idx = list(range(min(args.num_images, len(df))))

    # ------------------------------------------------------------------
    # 2. Run both fitters per image
    # ------------------------------------------------------------------
    print(f"[4/5] Fitting on {len(sample_idx)} images "
          f"({args.num_steps} steps × 2 methods)...\n")

    rows_a, rows_b = [], []

    for k, idx in enumerate(sample_idx):
        try:
            info = get_first_person_row(df, idx)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip row {idx}: {exc}")
            continue

        joints_3d_gt = info["joints_3d"][:22]
        if joints_3d_gt.shape[0] < 6:
            print(f"  skip row {idx}: only {joints_3d_gt.shape[0]} joints")
            continue

        kp_2d = project_3d_to_2d(joints_3d_gt)
        kp_2d_noisy = kp_2d + np.random.normal(
            0, args.noise_std, kp_2d.shape).astype(np.float32)
        if kp_2d_noisy.shape[0] < 22:
            padded = np.full((22, 2), np.nan, dtype=np.float32)
            padded[:kp_2d_noisy.shape[0]] = kp_2d_noisy
            kp_2d_full = padded
        else:
            kp_2d_full = kp_2d_noisy

        # --- Method A: 2D-driven fitter (no VPoser) -----------------------
        t0 = time.time()
        try:
            res_a = fit_smplx_to_2d(model, kp_2d_full,
                                    num_steps=args.num_steps,
                                    lr=0.05, lambda_shape=1e-3, lambda_pose=1e-3,
                                    device=args.device, verbose=False)
            ok_a = True
        except Exception as exc:  # noqa: BLE001
            print(f"  [A] row {idx} FAILED: {exc}")
            ok_a = False
        t_a = time.time() - t0

        # --- Method B: SMPLify-X + VPoser --------------------------------
        t0 = time.time()
        try:
            res_b = fit_smplx_with_vposer(model, vposer, kp_2d_full,
                                          num_steps=args.num_steps,
                                          lr=0.05,
                                          lambda_shape=1e-2, lambda_kl=1e-3,
                                          lambda_reproj=1.0,
                                          device=args.device, verbose=False)
            ok_b = True
        except Exception as exc:  # noqa: BLE001
            print(f"  [B] row {idx} FAILED: {exc}")
            ok_b = False
        t_b = time.time() - t0

        # ------------------------------------------------------------------
        # Evaluate against GT 3D joints
        # ------------------------------------------------------------------
        n_joints = joints_3d_gt.shape[0]

        def _eval(res, cam, ok):
            if not ok:
                return dict(mpjpe=np.nan, pa_mpjpe=np.nan,
                            pelvis_mpjpe=np.nan, pelvis_pa_mpjpe=np.nan,
                            reproj=np.nan, pelvis_reproj=np.nan,
                            betas_norm=np.nan, pose_norm=np.nan,
                            z_norm=np.nan, time_s=np.nan)
            j3d = res["joints_3d"][:n_joints]
            err_mpjpe = mpjpe(j3d, joints_3d_gt)
            err_pa = pa_mpjpe(j3d, joints_3d_gt)
            err_pel = pelvis_mpjpe(j3d, joints_3d_gt)
            err_pel_pa = pelvis_pa_mpjpe(j3d, joints_3d_gt)
            proj = (j3d @ cam.rot.T) * cam.scale + cam.trans
            err_reproj = reprojection_error_px(proj, kp_2d_noisy)
            err_pel_reproj = pelvis_reprojection_error_px(proj, kp_2d_noisy)
            return dict(
                mpjpe=err_mpjpe, pa_mpjpe=err_pa,
                pelvis_mpjpe=err_pel, pelvis_pa_mpjpe=err_pel_pa,
                reproj=err_reproj, pelvis_reproj=err_pel_reproj,
                betas_norm=float(np.linalg.norm(res["betas"])),
                pose_norm=float(np.linalg.norm(res["body_pose"])),
                z_norm=float(np.linalg.norm(res.get("pose_z", [0.0])))
                       if "pose_z" in res else np.nan,
                time_s=np.nan,
            )

        stats_a = _eval(res_a, res_a["camera"], ok_a) if ok_a else _eval(None, None, False)
        stats_b = _eval(res_b, res_b["camera"], ok_b) if ok_b else _eval(None, None, False)
        if ok_a:
            stats_a["time_s"] = t_a
        if ok_b:
            stats_b["time_s"] = t_b

        rows_a.append({"row_idx": int(idx),
                       "img_path": info["img_path"],
                       "time_s": t_a, **stats_a})
        rows_b.append({"row_idx": int(idx),
                       "img_path": info["img_path"],
                       "time_s": t_b, **stats_b})

        def _fmt(d):
            return (f"MPJPE={d['mpjpe']:6.1f}mm  PA={d['pa_mpjpe']:6.1f}mm  "
                    f"Pel={d['pelvis_mpjpe']:6.1f}mm  PelPA={d['pelvis_pa_mpjpe']:6.1f}mm  "
                    f"Reproj={d['reproj']:5.1f}px  PelReproj={d['pelvis_reproj']:5.1f}px  "
                    f"|β|={d['betas_norm']:4.2f}  |p|={d['pose_norm']:5.1f}  "
                    f"t={d['time_s']:4.1f}s")

        print(f"  [{k + 1:>3}/{len(sample_idx)}] row={idx}")
        print(f"    A (raw pose)        : {_fmt(stats_a)}")
        print(f"    B (VPoser)          : {_fmt(stats_b)}")

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    print(f"\n[5/5] Aggregating {len(rows_a)} rows...")

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)

    df_a.to_csv(os.path.join(args.out_dir, "method_A_raw_pose.csv"), index=False)
    df_b.to_csv(os.path.join(args.out_dir, "method_B_vposer.csv"), index=False)

    def _agg(df, label):
        return {
            "method": label,
            "n": len(df),
            "mpjpe_mm_mean": float(df["mpjpe"].mean()),
            "mpjpe_mm_std":  float(df["mpjpe"].std()),
            "pa_mpjpe_mm_mean": float(df["pa_mpjpe"].mean()),
            "pa_mpjpe_mm_std":  float(df["pa_mpjpe"].std()),
            "pelvis_mpjpe_mm_mean": float(df["pelvis_mpjpe"].mean()),
            "pelvis_mpjpe_mm_std":  float(df["pelvis_mpjpe"].std()),
            "pelvis_pa_mpjpe_mm_mean": float(df["pelvis_pa_mpjpe"].mean()),
            "pelvis_pa_mpjpe_mm_std":  float(df["pelvis_pa_mpjpe"].std()),
            "reproj_px_mean": float(df["reproj"].mean()),
            "reproj_px_std":  float(df["reproj"].std()),
            "pelvis_reproj_px_mean": float(df["pelvis_reproj"].mean()),
            "pelvis_reproj_px_std":  float(df["pelvis_reproj"].std()),
            "time_s_mean": float(df["time_s"].mean()),
            "betas_norm_mean": float(df["betas_norm"].mean()),
            "pose_norm_mean": float(df["pose_norm"].mean()),
            "z_norm_mean": float(df["z_norm"].mean()),
        }

    agg = [_agg(df_a, "A: 2D raw-pose fitter"),
           _agg(df_b, "B: SMPLify-X + VPoser")]
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    print("\n================ SUMMARY ================")
    print(df_agg.to_string(index=False))
    print("=========================================\n")

    delta = agg[1]["mpjpe_mm_mean"] - agg[0]["mpjpe_mm_mean"]
    pct = 100.0 * delta / max(agg[0]["mpjpe_mm_mean"], 1e-6)
    print(f"MPJPE: VPoser method is {delta:+.1f} mm vs raw-pose "
          f"({pct:+.1f}% relative)")
    print(f"PA-MPJPE: VPoser is {agg[1]['pa_mpjpe_mm_mean'] - agg[0]['pa_mpjpe_mm_mean']:+.1f} mm vs raw-pose")
    print(f"Pelvis-MPJPE: VPoser is {agg[1]['pelvis_mpjpe_mm_mean'] - agg[0]['pelvis_mpjpe_mm_mean']:+.1f} mm vs raw-pose")
    print(f"Pelvis-PA-MPJPE: VPoser is {agg[1]['pelvis_pa_mpjpe_mm_mean'] - agg[0]['pelvis_pa_mpjpe_mm_mean']:+.1f} mm vs raw-pose")
    print(f"Reproj: VPoser is {agg[1]['reproj_px_mean'] - agg[0]['reproj_px_mean']:+.1f} px vs raw-pose")
    print(f"Pelvis-Reproj: VPoser is {agg[1]['pelvis_reproj_px_mean'] - agg[0]['pelvis_reproj_px_mean']:+.1f} px vs raw-pose")
    print(f"Time:   VPoser is {agg[1]['time_s_mean'] / max(agg[0]['time_s_mean'], 1e-6):.2f}× slower")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"per_image_A": rows_a, "per_image_B": rows_b,
                   "aggregate": agg}, f, indent=2)
    print(f"Saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
