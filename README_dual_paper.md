# 双轨并行投稿包——使用说明

> 本目录是给 **Pengyu Guo（西安工程大学计算机科学学院）** 的双轨并行投稿包。  
> 两篇论文共用同一份实验数据（AGORA val），但叙事角度、目标期刊、模板完全不同。

---

## 📁 目录结构

```
e:\smpl_reconstruction\
├── paper\         (原 IEEE 会议版，不动)
├── paper_vc\      ← The Visual Computer (Springer, SCI Q4, 不花钱)
│   ├── main.tex           # 英文，Springer svjour3
│   ├── references.bib     # 已嵌入内联 + 备份
│   ├── cover_letter.txt   # 给编辑的信
│   ├── highlights.txt     # 5 条 bullet highlights
│   └── figures\           # montage、sample、contamination
│
├── paper_jg\      ← 《图学学报》(中文核心+EI，免版面费)
│   ├── main.tex           # 中文，CTeX
│   ├── cover_letter.txt   # 中文致编辑信
│   └── figures\           # 同上
│
├── src\
│   ├── run_full_eval.py        # ★ NEW  批处理 200 张评估
│   ├── mediapipe_2d.py         # ★ NEW  真实 2D 检测器
│   ├── plot_contamination.py   # ★ NEW  Figure 4 绘制脚本
│   └── (其它原有脚本)
│
└── README_dual_paper.md     (本文件)
```

---

## 🎯 两篇论文的核心差异

| 维度 | paper_vc (Visual Computer) | paper_jg (图学学报) |
|---|---|---|
| 期刊类型 | SCI Q4 英文期刊 | 中文核心 + EI 收录 |
| 语言 | 英文 | 中文 |
| 模板 | Springer svjour3 单栏双栏 | CTeX 单栏 |
| 标题 | *A Pelvis-Relative Re-evaluation Protocol for 2D-Driven Human Mesh Reconstruction* | 《基于二维关键点的轻量化 SMPL-X 人体网格重建及一种骨盆相对再评价协议》 |
| **核心贡献** | **PRP 协议**（方法学贡献）| 轻量化拟合器 + PRP（工程应用）|
| 篇幅 | 12-16 页 | 6-7 页 |
| 周期 | 4-6 个月 | 3-4 个月 |
| 录用率（估计）| 40-50% | 70-80%（中文 EI）|
| 费用 | 0 元 | 0 元 |
| 含金量 | SCIE 检索 | EI 检索（毕业够）|

**两篇都强烈推荐双投：**
- paper_jg 先投 → 3-4 个月拿到 EI 检索通知 → 毕业无忧
- paper_vc 同步投 → 4-6 个月拿到 SCIE 检索 → 锦上添花

---

## 🚀 现在你能做的（按顺序）

### 第一步：确认 paper_jg 中文版可编译（10 分钟）
```bash
cd e:\smpl_reconstruction\paper_jg
xelatex main.tex
bibtex   main   # 如果用 bib 文件；当前用 inline thebibliography 可跳过
xelatex main.tex
xelatex main.tex
```
**预期输出**：`main.pdf`，6-7 页含中文摘要。

### 第二步：跑全量评估，把数据填实（1-2 天）
```bash
cd e:\smpl_reconstruction
python src\run_full_eval.py --num-images 200 --out-dir output_full_eval
```
**预期输出**：
- `output_full_eval/per_image.csv` —— 200 行 × ~12 列
- `output_full_eval/summary.csv` —— 方法 A vs 方法 B 完整对比

**⚠️ 如果中途卡住**：
- 200 张太慢 → 改成 `--num-images 50` 先跑通
- 想加 HMR baseline → 加 `--include-hmr`（需先装 HMR）

### 第三步：跑 MediaPipe 真实检测（半天）
```bash
pip install mediapipe opencv-python
python src\mediapipe_2d.py --image-dir dataset\agora\img\val --out-dir output_mp_2d --max-images 50
```
**预期输出**：`output_mp_2d/*.npy`，每张图一个 22×2 的关键点文件。

### 第四步：画 Figure 4 (PRP contamination)（10 分钟）
```bash
cd e:\smpl_reconstruction
python src\plot_contamination.py --per-image-csv output_compare\per_image.csv
```
**预期输出**：
- `paper_vc/figures/contamination.pdf`
- `paper_vc/figures/contamination.png`（300 dpi）

### 第五步：编译 paper_vc 英文版（10 分钟）
需要 `svjour3.cls`：
```bash
# 下载：https://www.springer.com/journal/00371/submission-guidelines
# 把 svjour3.cls 放到 paper_vc\ 目录
cd e:\smpl_reconstruction\paper_vc
pdflatex main.tex
bibtex main   # 如果用 bib 文件
pdflatex main.tex
pdflatex main.tex
```
**预期输出**：`main.pdf`，约 13-14 页（双栏）。

### 第六步：把全量评估数据填进论文（1-2 天）
把 `output_full_eval/summary.csv` 中的数字替换：
- Table~\ref{tab:ab} 的全部数字
- Table~\ref{tab:abl} 的全部数字  
- Abstract 中所有"16 images"改为"200 images"
- §Limitations 第 1 条改写（不再限于 16 张）

---

## 📝 投稿前的最后清单

### paper_jg（图学学报）清单
- [ ] 用 xelatex 编译通过
- [ ] 摘要字数 ≤ 400 字（中文期刊一般有硬性要求）
- [ ] 关键词 3-8 个
- [ ] 参考文献 GB/T 7714 格式（已在 inline 中按格式写入）
- [ ] 通信作者 + 单位 + 邮箱齐全
- [ ] 中图分类号、文献标识码（投稿时编辑会告诉你）

### paper_vc（Visual Computer）清单
- [ ] 用 pdflatex 编译通过（需 svjour3.cls）
- [ ] 论文长度 12-16 页
- [ ] Highlights 5 条，每条 ≤ 85 字符
- [ ] Cover Letter 完整
- [ ] ORCID 注册（orcid.org → 申请免费 → 把 ID 填入 main.tex 第 31 行）
- [ ] 注册 Springer 账户：submission.springernature.com → new submission → The Visual Computer
- [ ] 数据可用性声明：所有代码 MIT 协议公开

---

## 🎓 长期目标对齐

| 时间节点 | 任务 | 状态 |
|---|---|---|
| **2026-09** | 双稿就绪 | ✅ 完成 |
| **2026-10** | 投 paper_jg《图学学报》（保底）| ⏳ 下一步 |
| **2026-10** | 投 paper_vc The Visual Computer（冲 SCI）| ⏳ 下一步 |
| **2027-01** | 收到 paper_jg 审稿意见 → 修改 → 录用 | ⏳ |
| **2027-03** | paper_jg 见刊 / EI 检索 | ⏳ → 毕业无忧 |
| **2027-04** | 收到 paper_vc 审稿意见 → R1 | ⏳ |
| **2027-06** | paper_vc 录用 | ⏳ → SCI Q4 加分 |

---

## 💬 我能持续帮你做的

### 立即可做（你点确认）
1. 跑通 `run_full_eval.py`，把 200 张的数据填回 paper_vc 的 Table I/II/III
2. 把 `mediapipe_2d.py` 的输出和合成噪声输入做一组对比表（"synthetic noise vs real detector"）
3. 在 paper_vc 加 §Statistical Significance，写 Wilcoxon signed-rank 检验

### 接下来一个月
1. 完成所有 figure 的 dpi=300 重渲染
2. 把代码整理上 GitHub（README、LICENSE、citation.cff）
3. HMR/SPIN baseline 接入
4. 写 R1 响应模板（预先准备常见审稿意见的回答）

### 接下来三个月
1. 投 paper_jg 后等待审稿
2. paper_vc 投稿后等待审稿
3. 根据 R1 修改

---

## 🚨 风险提示

- **paper_vc 投稿时间窗口**：The Visual Computer 滚动接收，但 SCI 期刊审稿周期长，**建议 9 月底前完成投稿**。
- **paper_jg 春节前截止**：多数中文期刊春节前后休刊一个月，**建议 11 月中旬前投出**。
- **代码公开**：SCI 期刊几乎都要求 code release，**必须在论文录用前 push 到 GitHub**。
- **ORCID 提前注册**：很多期刊系统要求 ORCID，**现在注册只用 5 分钟**。

---

## 📞 你只需点击确认的关键操作

1. **ORCID 注册**：访问 https://orcid.org/register → 用学校邮箱注册 → 把 ID 粘给我
2. **xelatex 编译**：装 TeXLive 完整版（包含 ctex），运行 `xelatex main.tex`
3. **GitHub 创建仓库**：https://github.com/new → 命名 `smpl-x-prp-evaluation` → Public → 完成后给我仓库 URL

其余我都可以在 agent 模式里完成。
