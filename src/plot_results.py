"""
生成网格简化实验的可视化图：
1. 网格面数 vs RMS 误差 trade-off 曲线
2. 简化前后网格对比（ASCII art 风格的端面轮廓 + 多视角渲染）
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_summary(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot_tradeoff(df: pd.DataFrame, out_path: str) -> None:
    """面数 vs RMS 误差 trade-off 曲线（主图）。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ratios = df["ratio"].astype(float).values
    n_faces = df["n_faces_mean"].values
    rms = df["rms_mm_mean"].values
    rms_std = df["rms_mm_std"].values

    ax.plot(n_faces, rms, "o-", color="#2E86AB", linewidth=2, markersize=8)
    ax.fill_between(
        n_faces,
        rms - rms_std,
        rms + rms_std,
        alpha=0.15, color="#2E86AB",
        label=r"$\pm 1$ std",
    )

    ax.set_xlabel("Number of Triangles", fontsize=12)
    ax.set_ylabel("RMS Vertex Error (mm)", fontsize=12)
    ax.set_title("Mesh Simplification Trade-off\n(Quadric Error Decimation on SMPL-X)",
                 fontsize=13, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # 在每个点上标简化率
    for x, y, r in zip(n_faces, rms, ratios):
        ax.annotate(f"{r:.0%}" if r < 1 else "Full",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_comparison_table(df: pd.DataFrame, out_path: str) -> None:
    """生成简化率 / 面数 / 误差 / 文件大小的汇总表格图。"""
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    columns = ["Ratio", "Faces", "Verts", "RMS (mm)", "Max (mm)", "Size (KB)"]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            f"{float(r['ratio']):.0%}" if float(r['ratio']) < 1 else "100%",
            f"{int(r['n_faces_mean']):,}",
            f"{int(r['n_verts_mean']):,}",
            f"{r['rms_mm_mean']:.1f} \u00b1 {r['rms_mm_std']:.1f}",
            f"{r['max_mm_mean']:.1f} \u00b1 {r['max_mm_std']:.1f}",
            f"{r['obj_kb_mean']:.0f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # 表头样式
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2E86AB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # 交替行颜色
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#EAF4F4")
            else:
                table[i, j].set_facecolor("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_error_distribution(per_sample_path: str, out_path: str,
                              ratios=(0.50, 0.25, 0.10, 0.05, 0.02)) -> None:
    """各简化档位的 RMS 误差分布箱线图。"""
    import json
    with open(per_sample_path) as f:
        data = json.load(f)

    samples = data["per_sample"]
    ratio_keys = [f"{r:.2f}" for r in ratios]
    dist_data = []
    labels = []
    for rk in ratio_keys:
        vals = [s["mesh_stats"][rk]["rms_mm"] for s in samples
                if rk in s["mesh_stats"]]
        if vals:
            dist_data.append(vals)
            labels.append(f"{float(rk):.0%}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(dist_data, labels=labels, patch_artist=True,
                    medianprops={"color": "#E74C3C", "linewidth": 2})
    colors = ["#AED6F1", "#85C1E9", "#5DADE2", "#3498DB", "#2E86AB"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Simplification Ratio", fontsize=12)
    ax.set_ylabel("RMS Vertex Error (mm)", fontsize=12)
    ax.set_title("RMS Error Distribution Across 100 Random Poses", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="output/simplify_summary.csv")
    parser.add_argument("--results-json", default="output/results.json")
    parser.add_argument("--out-dir", default="output/vis")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_summary(args.summary_csv)

    plot_tradeoff(df, os.path.join(args.out_dir, "tradeoff_curve.png"))
    plot_comparison_table(df, os.path.join(args.out_dir, "summary_table.png"))

    if os.path.exists(args.results_json):
        plot_error_distribution(
            args.results_json,
            os.path.join(args.out_dir, "error_boxplot.png"),
        )


if __name__ == "__main__":
    main()