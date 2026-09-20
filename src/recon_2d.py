"""
2D keypoint-driven SMPL-X body mesh reconstruction.

Implements a lightweight optimization-based pipeline. Given a set of 2D
keypoints (image pixel coordinates) and a SMPL-X parametric body model,
iteratively refines the model parameters (global orientation, body pose,
shape, translation and a global scale) so that the projected 3D joints
match the 2D observations.

This is a simplified, single-image analogue of SMPLify-X, designed for
reproducibility on a CPU within minutes per image, suitable for an EI
conference paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import smplx
import torch


SMPLX_TO_2D_MAP = None


def _load_smplx_to_openpose_map() -> list[int]:
    """Return indices that map SMPL-X joints (144 joints total) to the 25
    OpenPose body keypoints. The list has length 25; entries are SMPL-X
    joint indices, with -1 for joints that don't have a direct mapping.
    """
    return [
        -1,  # 0 nose  -> approximated via head average later
        18, 19,  # 1, 2 eyes
        20, 21,  # 3, 4 ears
        -1, -1,  # 5, 6 shoulders approximated via clavicle 16/17
        16, 17,  # 7, 8  shoulders
        -1, -1,  # 9,10 elbows approximated
        18, 19,  # placeholder, overwritten below
    ]


SMPLX_BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# Mapping from OpenPose BODY_25 (25 joints) to a chosen set of SMPLX
# 22-body joints. -1 means no direct match (we approximate).
OPENPOSE25_TO_SMPLX_BODY = [
    -1,   # 0  nose        -> head (idx 15)
    15,   # 1  neck        -> neck
    16,   # 2  rShoulder   -> left_shoulder (mirrored dataset convention)
    17,   # 3  rElbow      -> right_shoulder
    18,   # 4  rWrist      -> right_elbow
    19,   # 5  lShoulder   -> left_elbow
    -1,   # 6  lElbow      -> left_wrist
    -1,   # 7  lWrist      -> use right_wrist placeholder
    13,   # 8  midHip      -> pelvis
    14,   # 9  rHip        -> left_hip
    1,    # 10 rKnee       -> right_hip
    2,    # 11 lKnee       -> right_knee
    4,    # 12 rAnkle      -> left_knee
    5,    # 13 lAnkle      -> left_ankle (placeholder)
    -1,   # 14 rFoot       -> use generic
    -1,   # 15 lFoot
    -1,   # 16 rEye
    -1,   # 17 lEye
    -1,   # 18 rEar
    -1,   # 19 lEar
    -1,   # 20 lBigToe
    -1,   # 21 lSmallToe
    -1,   # 22 lHeel
    -1,   # 23 rBigToe
    -1,   # 24 rSmallToe
]


@dataclass
class Camera:
    """A minimal weak-perspective camera.

    We represent projection as
        p_2d = s * R @ J_3d + t
    where s is a scalar scale, R is a 2x3 matrix projecting onto the
    image plane (we use the first two rows of a 3x3 rotation), and t is
    a 2D translation (image centre offset)."""
    scale: float = 1.0
    rot: Optional[np.ndarray] = None     # (2, 3)
    trans: Optional[np.ndarray] = None   # (2,)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D under weak perspective."""
        if self.rot is None or self.trans is None:
            raise RuntimeError("Camera must be initialised before calling project()")
        return self.scale * (points_3d @ self.rot.T) + self.trans


def build_smplx_model(model_dir: str, gender: str = "neutral") -> smplx.SMPLX:
    """Create a SMPL-X model on CPU."""
    return smplx.create(
        model_path=model_dir,
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
    )


def guess_camera_from_keypoints(keypoints_2d: np.ndarray,
                                keypoints_3d_init: np.ndarray,
                                focal_length: float = 1000.0,
                                image_size: tuple[int, int] = (720, 1280)) -> Camera:
    """Heuristically initialise a weak-perspective camera.

    The two arrays can have different sizes; only the overlapping prefix
    of valid 2D entries is used.
    """
    n = min(len(keypoints_2d), len(keypoints_3d_init))
    valid_2d = ~np.isnan(keypoints_2d[:n]).any(axis=1)
    if valid_2d.sum() < 2:
        return Camera(scale=1.0, rot=np.eye(3)[:2], trans=np.array(image_size) / 2)
    pts2d = keypoints_2d[:n][valid_2d]
    pts3d = keypoints_3d_init[:n][valid_2d]
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


def fit_smplx_to_2d(model: smplx.SMPLX,
                    keypoints_2d: np.ndarray,
                    num_steps: int = 200,
                    lr: float = 0.05,
                    lambda_shape: float = 1e-3,
                    lambda_pose: float = 1e-3,
                    focal_length: float = 1000.0,
                    image_size: tuple[int, int] = (720, 1280),
                    device: str = "cpu",
                    verbose: bool = False) -> dict:
    """Fit SMPL-X parameters to 2D keypoints via gradient descent.

    Parameters
    ----------
    model : smplx.SMPLX
        Body model used to compute joints and vertices.
    keypoints_2d : (K, 2) float
        Pixel coordinates of the detected keypoints (NaN where missing).
    num_steps : int
        Number of optimisation iterations.
    lr : float
        Learning rate.
    lambda_shape, lambda_pose : float
        L2 regulariser weights on shape and pose.
    focal_length, image_size : float, (h, w)
        Used to initialise the weak-perspective camera.

    Returns
    -------
    dict with keys: vertices, joints_3d, camera, betas, body_pose, global_orient, transl, loss_hist.
    """
    model = model.to(device)

    # Initial SMPL-X parameters
    betas = torch.zeros(1, model.num_betas, device=device, requires_grad=True)
    body_pose = torch.zeros(1, model.NUM_BODY_JOINTS * 3, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([betas, body_pose, global_orient, transl], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(num_steps // 3, 1),
                                                gamma=0.5)

    # Initial 3D joints for camera guess (T-pose) — use the full 127 SMPL-X
    # joints for the camera guess, then we restrict the reprojection
    # loss to the requested indices below.
    with torch.no_grad():
        out0 = model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                     transl=transl)
    joints0 = out0.joints[0].detach().cpu().numpy()[:, :3]
    cam = guess_camera_from_keypoints(keypoints_2d, joints0,
                                      focal_length=focal_length, image_size=image_size)

    # Indices we use as keypoints: take the first 22 valid 2D keypoints
    # and match them to the first 22 SMPL-X body joints.
    valid_mask = ~np.isnan(keypoints_2d).any(axis=1)
    n_use = min(22, int(valid_mask.sum()))
    if n_use < 6:
        raise RuntimeError(
            f"Only {n_use} valid 2D keypoints, need at least 6")
    joint_idx = np.where(valid_mask)[0][:n_use]
    kp_2d_t = torch.tensor(keypoints_2d[joint_idx],
                           dtype=torch.float32, device=device)
    joint_idx_t = torch.tensor(joint_idx, dtype=torch.long, device=device)

    # Store camera as tensors so they can be optimised too.
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
        out = model(betas=betas, body_pose=body_pose,
                    global_orient=global_orient, transl=transl)
        joints_3d = out.joints[0, joint_idx_t]  # (K, 3)
        # Full weak-perspective projection: scale + rotation (2x3) + translation
        proj_2d = (joints_3d @ rot_mat.T) * cam_scale + cam_trans
        reproj_loss = ((proj_2d - kp_2d_t) ** 2).sum(dim=-1).mean()

        shape_reg = (betas ** 2).mean()
        pose_reg = (body_pose ** 2).mean()
        orient_reg = (global_orient ** 2).mean()
        loss = reproj_loss + lambda_shape * shape_reg + lambda_pose * pose_reg

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(float(reproj_loss.detach().cpu()))
        if verbose and (step + 1) % max(num_steps // 5, 1) == 0:
            print(f"  step {step + 1}/{num_steps}  reproj={reproj_loss.item():.3f}")

    with torch.no_grad():
        out = model(betas=betas, body_pose=body_pose,
                    global_orient=global_orient, transl=transl)
    final_cam = Camera(scale=float(cam_scale.item()),
                       rot=cam.rot.copy(),
                       trans=cam_trans.detach().cpu().numpy().copy())
    return {
        "vertices": out.vertices[0].detach().cpu().numpy(),
        "joints_3d": out.joints[0].detach().cpu().numpy()[:, :3],
        "camera": final_cam,
        "betas": betas.detach().cpu().numpy().reshape(-1),
        "body_pose": body_pose.detach().cpu().numpy().reshape(-1),
        "global_orient": global_orient.detach().cpu().numpy().reshape(-1),
        "transl": transl.detach().cpu().numpy().reshape(-1),
        "loss_hist": loss_hist,
    }