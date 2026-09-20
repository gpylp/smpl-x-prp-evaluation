"""MediaPipe Pose 2D keypoint detector for SMPL-X 22-body-joint mapping.

MediaPipe Pose (BlazePose GHUM) outputs 33 landmarks. We map the most
reliable 22 of those landmarks to the SMPL-X 22-body-joint layout used
in this repository. The mapping follows the convention of OpenPose /
SMPLify-X, where ``PELVIS`` is the mid-hip midpoint and ``NECK`` is the
mid-shoulder midpoint.

Joint index mapping (MediaPipe -> SMPL-X 22-joint):

    SMPL-X idx   joint name   MediaPipe landmark idx
    --------------------------------------------------
    0            pelvis       midpoint(23, 24)        # mid-hip
    1            left_hip     23                      # left hip
    2            right_hip    24                      # right hip
    3            spine1       midpoint(11, 12)        # mid-shoulder
    4            left_knee    25                      # left knee
    5            right_knee   26                      # right knee
    6            spine2       midpoint(11, 12)        # = spine1
    7            left_ankle   27                      # left ankle
    8            right_ankle  28                      # right ankle
    9            spine3       midpoint(11, 12)        # = spine1
    10           left_foot    31                      # left foot index
    11           right_foot   32                      # right foot index
    12           neck         midpoint(11, 12)        # = spine1
    13           left_collar  11                      # left shoulder
    14           right_collar 12                      # right shoulder
    15           head         0                       # nose
    16           left_shoulder 11                     # same as collar
    17           right_shoulder 12                    # same as collar
    18           left_elbow    13                     # left elbow
    19           right_elbow   14                     # right elbow
    20           left_wrist    15                     # left wrist
    21           right_wrist   16                     # right wrist

Outputs are in image pixel coordinates (W, H).  When a landmark is
invisible (off-canvas or low confidence), the corresponding entry is
``np.nan`` and the caller may either skip or impute.

Usage
-----
    python src/mediapipe_2d.py --image path/to/img.png --out kp.npy
    python src/mediapipe_2d.py --image-dir dataset/agora/img/val --out-dir output_mp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# SMPL-X 22-body-joint layout (must match recon_2d.py SMPLX_BODY_JOINT_NAMES)
SMPLX22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# Map from SMPL-X idx (0..21) to MediaPipe landmark idx (0..32)
MP_INDEX = {
    0:  None,   # pelvis = mid-hip midpoint(23, 24)
    1:  23,     # left_hip
    2:  24,     # right_hip
    3:  None,   # spine1 = mid-shoulder midpoint(11, 12)
    4:  25,     # left_knee
    5:  26,     # right_knee
    6:  None,   # spine2 = mid-shoulder (same as spine1)
    7:  27,     # left_ankle
    8:  28,     # right_ankle
    9:  None,   # spine3 = mid-shoulder
    10: 31,     # left_foot (foot index)
    11: 32,     # right_foot
    12: None,   # neck = mid-shoulder
    13: 11,     # left_collar (left shoulder)
    14: 12,     # right_collar
    15: 0,      # head (nose)
    16: 11,     # left_shoulder
    17: 12,     # right_shoulder
    18: 13,     # left_elbow
    19: 14,     # right_elbow
    20: 15,     # left_wrist
    21: 16,     # right_wrist
}


def mediapipe_to_smplx22(landmarks_xy: np.ndarray, visibility: np.ndarray,
                          min_visibility: float = 0.5) -> np.ndarray:
    """Convert MediaPipe 33-landmark output to SMPL-X 22-body-joint layout.

    Parameters
    ----------
    landmarks_xy : (33, 2) array of pixel coordinates (W, H)
    visibility   : (33,)   array of MediaPipe visibility scores [0, 1]
    min_visibility : float, landmarks below this score are returned NaN

    Returns
    -------
    (22, 2) array of pixel coordinates. NaN indicates "not detected".
    """
    out = np.full((22, 2), np.nan, dtype=np.float32)
    # mid-shoulder (used for spine1, spine2, spine3, neck)
    if visibility[11] >= min_visibility and visibility[12] >= min_visibility:
        mid_shoulder = 0.5 * (landmarks_xy[11] + landmarks_xy[12])
    else:
        mid_shoulder = np.array([np.nan, np.nan])
    # mid-hip
    if visibility[23] >= min_visibility and visibility[24] >= min_visibility:
        mid_hip = 0.5 * (landmarks_xy[23] + landmarks_xy[24])
    else:
        mid_hip = np.array([np.nan, np.nan])

    for smplx_idx, mp_idx in MP_INDEX.items():
        if mp_idx is not None and visibility[mp_idx] >= min_visibility:
            out[smplx_idx] = landmarks_xy[mp_idx]
    if not np.isnan(mid_hip).any():
        out[0] = mid_hip
    if not np.isnan(mid_shoulder).any():
        out[3] = mid_shoulder
        out[6] = mid_shoulder
        out[9] = mid_shoulder
        out[12] = mid_shoulder
    return out


def detect_image(image_path: str,
                 min_visibility: float = 0.5,
                 model_complexity: int = 1) -> Optional[np.ndarray]:
    """Run MediaPipe Pose on a single image and return the SMPL-X
    22-joint layout, or ``None`` if no pose is detected.

    Parameters
    ----------
    image_path : path to a PNG/JPG image
    min_visibility : landmarks with visibility < this value are dropped
    model_complexity : 0 (lite), 1 (full), 2 (heavy)

    Returns
    -------
    (22, 2) ndarray of pixel coordinates, or None
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "MediaPipe not installed. Run: pip install mediapipe opencv-python"
        ) from e

    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=min_visibility,
    )
    res = pose.process(img_rgb)
    pose.close()
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    landmarks_xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    visibility = np.array([p.visibility for p in lm], dtype=np.float32)
    return mediapipe_to_smplx22(landmarks_xy, visibility, min_visibility)


def _walk_images(root: Path):
    exts = {".png", ".jpg", ".jpeg"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Single image path")
    p.add_argument("--image-dir", help="Directory tree to scan for images")
    p.add_argument("--out", help="Output .npy file (single image mode)")
    p.add_argument("--out-dir", default="output_mediapipe_2d",
                   help="Output directory (image-dir mode)")
    p.add_argument("--min-visibility", type=float, default=0.5)
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--max-images", type=int, default=0,
                   help="0 = no limit; otherwise stop after N images")
    args = p.parse_args()

    if args.image:
        kp = detect_image(args.image, args.min_visibility, args.model_complexity)
        if kp is None:
            print(f"[warn] no pose detected in {args.image}", file=sys.stderr)
            sys.exit(1)
        if args.out:
            np.save(args.out, kp)
            print(f"[done] saved {kp.shape} -> {args.out}")
        else:
            print(json.dumps(kp.tolist()))
        return

    if not args.image_dir:
        p.error("Either --image or --image-dir must be set.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list(_walk_images(Path(args.image_dir)))
    if args.max_images:
        paths = paths[: args.max_images]
    print(f"[init] scanning {len(paths)} images ...", flush=True)
    n_ok = 0
    for i, img_path in enumerate(paths):
        kp = detect_image(str(img_path), args.min_visibility, args.model_complexity)
        if kp is None:
            continue
        np.save(out_dir / (img_path.stem + ".npy"), kp)
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1:5d}/{len(paths)}] detected={n_ok}", flush=True)
    print(f"[done] {n_ok}/{len(paths)} images produced keypoints in {out_dir}")


if __name__ == "__main__":
    main()
