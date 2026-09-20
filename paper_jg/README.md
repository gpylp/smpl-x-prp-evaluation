# 《图学学报》投稿——轻量化 SMPL-X 人体网格重建及骨盆相对评价协议

本目录包含中文 EI 期刊《图学学报》的投稿稿件及相关支撑材料。

## 📄 投稿稿件

| 文件 | 内容 |
|---|---|
| `main.tex`           | 中文全文（约 6-7 页），CTeX 模板 |
| `cover_letter.txt`   | 致编辑信 |
| `figures/`           | 6 幅配图（montage + 3 个 compare + 1 个 sample + contamination）|

## 🚀 编译

```bash
cd paper_jg
xelatex main.tex
xelatex main.tex   # 跑两遍以解析交叉引用
```

需要 TeXLive 完整版（含 `ctex` 宏包）。

## 📊 核心贡献（中文摘要要点）

1. **轻量化 SMPL-X 拟合器开源实现**：单文件、CPU-only、3.18 秒/帧。
2. **骨盆相对再评价协议（PRP）**：揭示 MPJPE 在世界坐标真值下的"全局自由度污染"现象。
3. **超参数不敏感性实证**：18 组配置的网格搜索表明该流程对正则权重不敏感。
4. **QED 网格简化**：10× 压缩仅带来 14.3 mm RMS 误差。

详见根目录 `README.md` 获取英文版完整说明和数据复现指令。
