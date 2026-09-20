"""VPoser v1.0 loader (SMPLify-X compatible).

The official VPoser v1.0 distribution (CVPR'19) ships with a plain
``.pt`` state-dict and an ``.ini``-style configuration.  This loader
bridges that to the modern ``human_body_prior`` API expected by the
rest of the codebase.

Usage
-----

    from vposer_loader import load_vposer_v1

    vposer, _ = load_vposer_v1(
        expr_dir=r"E:\\smpl_reconstruction\\models\\vposer"
                 r"\\v02_05\\snapshots\\vposer_v1_0\\vposer_v1_0",
        device="cpu",
    )
    z = vposer.encode(pose_aa_flat)        # N x latentD
    pose = vposer.decode(z, output_type='aa')  # N x 1 x J x 3
"""

from __future__ import annotations

import configparser
import importlib.util
import os
import sys
from pathlib import Path
from typing import Tuple

import torch


def _load_vposer_class(expr_dir: str):
    """Dynamically import ``vposer_smpl.py`` from the experiment dir and
    return its :class:`VPoser` symbol."""
    expr_dir = os.fspath(expr_dir)
    mod_path = os.path.join(expr_dir, "vposer_smpl.py")
    if not os.path.exists(mod_path):
        raise FileNotFoundError(f"VPoser model definition not found: {mod_path}")

    spec = importlib.util.spec_from_file_location("vposer_smpl", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vposer_smpl"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.VPoser


def _parse_ini_cfg(cfg_path: str) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    if "All" not in cfg:
        # Fall back to the only available section
        sections = cfg.sections()
        if not sections:
            raise ValueError(f"No sections in INI config {cfg_path}")
        section = sections[0]
    else:
        section = "All"
    return dict(cfg.items(section))


def load_vposer_v1(expr_dir: str,
                   checkpoint_name: str = "TR00_E096.pt",
                   device: str = "cpu") -> Tuple[torch.nn.Module, dict]:
    """Load a VPoser v1.0 model.

    :param expr_dir: Path that contains ``vposer_smpl.py`` and an
        ``.ini`` config (e.g. ``TR00_004_00_WO_accad.ini``).
    :param checkpoint_name: Name of the ``.pt`` file inside
        ``<expr_dir>/snapshots/``.
    :param device: Target device for the loaded model.
    :returns: ``(vposer, meta)`` where ``meta`` carries config info.
    """
    expr_dir = os.fspath(expr_dir)
    ini_candidates = sorted(Path(expr_dir).glob("TR00_*.ini"))
    if not ini_candidates:
        raise FileNotFoundError(
            f"No VPoser .ini config found under {expr_dir}")
    cfg = _parse_ini_cfg(str(ini_candidates[0]))

    # ``data_shape`` is stored as a string ``[1, 21, 3]`` in the INI.
    raw_shape = cfg.get("data_shape", "[1, 21, 3]")
    data_shape = [int(s.strip()) for s in raw_shape.strip("[]").split(",")]
    num_neurons = int(cfg.get("num_neurons", 512))
    latent_d = int(cfg.get("latentD", 32))
    use_cont_repr = cfg.get("use_cont_repr", "True").lower() in ("true", "1")

    VPoser = _load_vposer_class(expr_dir)
    model = VPoser(num_neurons=num_neurons,
                   latentD=latent_d,
                   data_shape=data_shape,
                   use_cont_repr=use_cont_repr)

    ckpt_path = os.path.join(expr_dir, "snapshots", checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VPoser checkpoint missing: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # VPoser v1.0 ships the state_dict directly; the newer framework
    # wraps it under "state_dict" with the "module." prefix.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[vposer] unexpected keys (ignored): {unexpected}")
    if missing:
        print(f"[vposer] missing keys (untrained): {missing}")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    meta = {
        "cfg_path": str(ini_candidates[0]),
        "checkpoint": ckpt_path,
        "data_shape": data_shape,
        "num_neurons": num_neurons,
        "latentD": latent_d,
        "use_cont_repr": use_cont_repr,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return model, meta
