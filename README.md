# SMPL-X PRP Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![AGORA val](https://img.shields.io/badge/dataset-AGORA--val-orange.svg)]()

> **Companion code** for the dual-track submission:
> *The Visual Computer* (Springer, SCI Q4) and 《图学学报》(EI).
> A clean, CPU-only, 2D-driven SMPL-X reconstruction pipeline with the
> **Pelvis-Relative Protocol (PRP)** for fairer evaluation.

---

## ✨ What's inside

| Component | Description |
|---|---|
| **`src/recon_2d.py`**           | Lightweight 2D-driven SMPL-X fitter (Method A: raw pose). |
| **`src/compare_fitters.py`**    | SMPLify-X + VPoser fitter (Method B). |
| **`src/eval.py`**               | All 4 metrics: MPJPE, PA-MPJPE, **Pelvis-MPJPE, Pelvis-PA-MPJPE, Pelvis-Reproj**. |
| **`src/mesh_simplify.py`**      | Quadric Error Decimation (QED) wrapper. |
| **`src/run_full_eval.py`** 🆕   | Batch evaluator (200 / 500 / N images). |
| **`src/mediapipe_2d.py`** 🆕    | MediaPipe → SMPL-X 22-joint mapper. |
| **`src/plot_contamination.py`**| Figure 4 generator (global-DoF contamination). |
| **`paper_vc/`**                 | LaTeX source for The Visual Computer. |
| **`paper_jg/`**                 | LaTeX source for 《图学学报》. |

---

## 🚀 Quick start

### 1. Install dependencies

```bash
pip install torch numpy pandas scipy matplotlib open3d smplx mediapipe opencv-python
```

### 2. Download the SMPL-X neutral model

Place the requested files into `models/smplx/SMPLX_NEUTRAL.pkl` (see
[SMPL-X docs](https://smpl-x.is.tue.mpg.de/) for the access form).

### 3. Download the AGORA validation set

Request access at <https://agora.is.tue.mpg.de/>, then point the
scripts to `dataset/agora/camera_meta/val/`.

### 4. Run the two fitters on 16 AGORA val images (default)

```bash
python src/compare_fitters.py --num-images 16 --out-dir output_compare
python src/compare_visualize.py          # produces the montage figure
```

### 5. Run a full 200-image batch

```bash
python src/run_full_eval.py --num-images 200 --out-dir output_full_eval
```

---

## 📊 The Pelvis-Relative Protocol (PRP)

**Problem.** Standard MPJPE/PA-MPJPE saturate at the average pelvis distance
(~4.6 m on AGORA) for any 2D-driven fitter, because the global translation
is unrecoverable from 2D input alone. We call this *global-DoF contamination*.

**Solution.** Subtract the pelvis joint from prediction and ground truth
*before* computing distances:

```python
def pelvis_subtract(joints):
    return joints - joints[0:1]   # SMPL-X joint 0 == pelvis
```

The corresponding metric suite is **Pelvis-MPJPE, Pelvis-PA-MPJPE,
Pelvis-Reproj**, implemented in `src/eval.py`.

### Empirical evidence

On 16 AGORA val images the legacy and PRP suites diverge by ~7×:

| Method              | MPJPE (mm) | PA-MPJPE (mm) | Pelvis-MPJPE (mm) | Pelvis-PA-MPJPE (mm) |
|---------------------|------------|---------------|-------------------|----------------------|
| **M-A** (raw pose)  | 4639.5     | 4623.6        | **697.3**         | 602.8                |
| **M-B** (VPoser)    | 4642.5     | 4623.8        | 709.2             | **602.1**            |

Pelvis-Reproj is **constant** at 207.0 px across 18 hyper-parameter
settings — a hard lower bound of the 2D→3D inversion ambiguity.

---

## 📚 Citation

If you use this code in academic work, please cite the companion papers
(see `paper_vc/references.bib` and `paper_jg/references.bib`):

```bibtex
@article{guo2026prp,
  title  = {A Pelvis-Relative Re-evaluation Protocol for 2D-Driven Human Mesh Reconstruction},
  author = {Guo, Pengyu},
  journal= {The Visual Computer (under review)},
  year   = {2026}
}
```

---

## 🤝 Contributing

Bug reports and PRs are welcome. For major changes please open an
issue first.

## 📄 License

[MIT](LICENSE)
