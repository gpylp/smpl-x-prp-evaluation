"""
精简实验流程（聚焦网格轻量化）。

思路：
1. 用 SMPL-X 参数化模型采样多种姿态的网格
2. 在多档简化率（10%~100%）下做 Quadric 网格简化
3. 报告顶点数 / 面数 / 几何误差 / 文件大小
4. 画出 trade-off 曲线

不依赖 AGORA 的 SMPL-X GT 文件，仅用模型自带的 neutral 模板
+ 随机采样的 betas/body_pose 就能产生大量样本。
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from mesh_simplify import simplify_mesh, compute_geometric_error
from recon_2d import build_smplx_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--gender", default="neutral")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--num-samples", type=int, default=30,
                   help="随机姿态样本数")
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02],
                   help="网格简化目标面数比例")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-meshes", action="store_true",
                   help="是否保存样例网格到 output/meshes/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    mesh_dir = os.path.join(args.out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # 1. 加载模型
    print(f"[1/3] 加载 SMPL-X 模型 ({args.gender}) ...")
    model = build_smplx_model(args.model_dir, gender=args.gender).to(args.device)
    faces = np.asarray(model.faces, dtype=np.int64)
    n_orig_verts = int(model.v_template.shape[0])
    n_orig_faces = int(faces.shape[0])
    print(f"      模板: {n_orig_verts} 顶点 / {n_orig_faces} 面")

    # 2. 随机采样姿态，简化网格，记录指标
    print(f"[2/3] 采样 {args.num_samples} 个随机姿态，"
          f"在 {len(args.ratios)} 档简化率下评估 ...")

    rows = []
    for i in range(args.num_samples):
        # 随机参数
        betas = torch.randn(1, model.num_betas) * 1.5
        body_pose = torch.randn(1, model.NUM_BODY_JOINTS * 3) * 0.6
        global_orient = torch.randn(1, 3) * 0.4
        transl = torch.zeros(1, 3)
        with torch.no_grad():
            out = model(betas=betas, body_pose=body_pose,
                        global_orient=global_orient, transl=transl)
        verts = out.vertices[0].detach().cpu().numpy()

        # 各档简化
        ratio_stats = {}
        for ratio in args.ratios:
            target = max(4, int(round(n_orig_faces * ratio)))
            if ratio >= 1.0:
                target = n_orig_faces
            t0 = time.time()
            v_s, f_s = simplify_mesh(verts, faces, target)
            t_simp = time.time() - t0
            geom = compute_geometric_error(verts, faces, v_s, f_s)
            # 估算文件大小（.obj 字节近似）
            obj_size = int(v_s.shape[0] * 6 + f_s.shape[0] * 12)
            ratio_stats[f"{ratio:.2f}"] = {
                "target_faces": target,
                "actual_faces": int(f_s.shape[0]),
                "actual_verts": int(v_s.shape[0]),
                "rms_mm": geom["rms_mm"],
                "max_mm": geom["max_mm"],
                "mean_mm": geom["mean_mm"],
                "simplify_time_s": t_simp,
                "obj_bytes": obj_size,
            }

        # 保存首样例的网格（写 OBJ 格式）
        if args.save_meshes and i == 0:
            def write_obj(path, verts, faces):
                with open(path, "w") as f:
                    f.write("# SMPL-X mesh, OBJ format\n")
                    for v in verts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    for t in faces:
                        f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
            write_obj(os.path.join(mesh_dir, "full.obj"), verts, faces)
            for ratio in args.ratios:
                if ratio >= 1.0:
                    continue
                v_s, f_s = simplify_mesh(verts, faces,
                                         int(round(n_orig_faces * ratio)))
                write_obj(os.path.join(mesh_dir, f"simplified_{ratio:.2f}.obj"),
                          v_s, f_s)

        rows.append({"sample_id": i, "mesh_stats": ratio_stats})

    # 3. 汇总统计
    print("[3/3] 汇总统计 ...")
    agg = []
    for ratio in args.ratios:
        key = f"{ratio:.2f}"
        rms_vals = [r["mesh_stats"][key]["rms_mm"] for r in rows]
        max_vals = [r["mesh_stats"][key]["max_mm"] for r in rows]
        nf_vals = [r["mesh_stats"][key]["actual_faces"] for r in rows]
        nv_vals = [r["mesh_stats"][key]["actual_verts"] for r in rows]
        sz_vals = [r["mesh_stats"][key]["obj_bytes"] for r in rows]
        ts_vals = [r["mesh_stats"][key]["simplify_time_s"] for r in rows]
        agg.append({
            "ratio": key,
            "n_faces_mean": float(np.mean(nf_vals)),
            "n_verts_mean": float(np.mean(nv_vals)),
            "rms_mm_mean": float(np.mean(rms_vals)),
            "rms_mm_std": float(np.std(rms_vals)),
            "max_mm_mean": float(np.mean(max_vals)),
            "max_mm_std": float(np.std(max_vals)),
            "obj_kb_mean": float(np.mean(sz_vals) / 1024.0),
            "simplify_time_s_mean": float(np.mean(ts_vals)),
        })
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(os.path.join(args.out_dir, "simplify_summary.csv"), index=False)
    print("\n" + df_agg.to_string(index=False))

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({
            "n_samples": len(rows),
            "n_orig_verts": n_orig_verts,
            "n_orig_faces": n_orig_faces,
            "simplify_summary": agg,
            "per_sample": rows,
        }, f, indent=2)

    print(f"\n汇总表 -> {os.path.join(args.out_dir, 'simplify_summary.csv')}")
    print("完成。")


if __name__ == "__main__":
    main()