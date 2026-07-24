# ANoCo（非官方复现）

[English](README.md) | **简体中文**

**训练无关、闭式解的异常检测** —— 对 CVPR 2026 论文 [*Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization*](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)
（Seo 等）的独立复现，并叠加了一组面向多样本（many-shot）工业质检的工程优化。

> ⚠️ **免责声明.** 本项目是根据论文完成的**非官方**复现，**未**获得论文作者或 Meta AI 的关联或背书。
> 截至撰写时作者尚未发布官方代码，本实现仅代表我们对该方法的理解。**不包含** DINOv3 权重与任何
> 专有数据集 —— 详见 [权重](#权重)。

## ANoCo 是什么？

与 PatchCore 类似，ANoCo 属于“特征记忆库 / 检索”范式的异常检测器 —— 但**打分内核不同**：

- **PatchCore** 用查询 patch 到最近正常 patch 的**距离**打分（独立相似度）。
- **ANoCo** 用把查询 patch 拉回正常流形所需的**特征漂移量**打分 —— 即*锚定图 Laplacian 能量
  最小化*的闭式解。漂移越大 ⇒ 越异常。无需训练、无需梯度：每个 patch 一次闭式求解。

流程（论文 §3.2–3.6）：

```
DINOv3 patch 特征
  → 锚定检索一致的正常邻居                 (§3.2, Eq. 1)
  → 双部图边权  w = cos · 范数兼容因子       (§3.3–3.4)
  → 锚定 Laplacian 闭式求解  f̃              (§3.5, Eq. 8–9)
  → 非一致性能量  E = ‖f̃ − f‖² · (1 − cos)   (§3.6, Eq. 10)
  → 异常热力图 / 图像级分数
```

## 亮点

- **忠实复现且经验证。** 闭式解与稠密线性求解 bit-exact 一致（`tests/`），MVTec-AD 指标贴合论文
  （见 [结果](#结果)）。
- **骨干无关的核心库**（`anoco/`）：只在 patch 特征上运算；DINOv3 是唯一的可选骨干依赖，隔离在
  `anoco/features/`。
- **工程优化**（超出论文范围，面向真实产线）：
  - **多层特征融合**（如 L11 + L17 拼接）—— 纹理 + 语义线索互补。
  - **聚类分层记忆库**（K-Means）+ **贪心 coreset**（PatchCore 式）—— 固定预算下更好的覆盖度。
  - **Excess 归一化** `max(0, E − μ − kσ)` —— 对标定 / 测试分布漂移鲁棒。
  - **Top-k 聚合**做图像级打分，比 max-pool 对孤立高分 patch 更不敏感。
  - **分块 + 稀疏匹配** 与 **top-k 检索** —— 结果不变，显存大幅降低（bit-exact）。
  - **阈值标定**到目标过杀（误报）率。
- **推理零测试时增强（TTA）** —— 所有增强都在参考侧预先计算。

## 安装

```bash
pip install -e .
# 可选依赖（MVTec demo 叠加图、safetensors 转换）：
pip install -e ".[demo]"
```

运行 DINOv3 骨干需要 CUDA GPU；合成数据的单元测试在 CPU 上即可运行。

## 权重

DINOv3 由 Meta 以 **DINOv3 License**（需申请授权）发布。本仓库**不分发**权重 —— 请自行获取并
遵守 Meta 的许可：

```bash
# 方式 A：官方权重（需先同意 Meta 的 DINOv3 许可）
python scripts/download_dinov3.py --backbone dinov3_vitl16

# 方式 B：把 Hugging Face 的 safetensors 转成 fb 格式
python scripts/convert_dinov3_safetensors.py \
    --src dinov3-vitl16-pretrain-lvd1689m.safetensors \
    --backbone dinov3_vitl16 --out weights/dinov3_vitl16.pt
```

在 `configs/*.yaml` 中通过 `dinov3_repo_dir` / `dinov3_weights` 指向 DINOv3 hub 仓库与权重，
或设置 `DINOV3_REPO` 环境变量。

## 快速开始

```bash
# 1) 单元测试（CPU，合成数据 —— 无需权重）
pytest tests/ -q

# 2) 复现 MVTec-AD 指标（公开数据，few-shot）
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
python scripts/demo_mvtec.py --categories screw metal_nut bottle --shots 1

# 3) 你自己的产线：先建库，再检测一个文件夹
python scripts/build_bank.py     --ok-dir path/to/ok  --config configs/production.yaml --out-dir banks/my_product
python scripts/run_inspector.py  --bank-dir banks/my_product --config configs/production.yaml \
                                 --image-dir path/to/images --save-overlays results/viz/
```

Python API：

```python
from anoco import ANoCo, ANoCoConfig

model = ANoCo(ANoCoConfig())
out = model.score_features(f_q, f_r, grid_hw=(48, 48))   # 骨干无关
print(out["score"], out["anomaly_map"].shape)
```

## 结果

### 复现保真度（像素级，`screw`）

核心方法贴合论文 —— MVTec-AD `screw`，DINOv3-L/16，单层，many-shot：
pixel-AUROC 97.7（论文 98.4）、pixel-PRO 91.8（93.5）、pixel-F1 49.1（53.2）。

### 工程优化有效吗？（公开数据消融）

在**全部 15 个 MVTec-AD 类别**上的图像级 AUROC，DINOv3-L/16，两列使用**完全相同**的 50 图
记忆库 + 贪心 coreset（10k）—— 从而隔离出 **多层融合（L11+L17）+ top-k 聚合** 相对单层基线的
贡献。复现命令：

```bash
# 基线
python scripts/demo_mvtec.py --data-root $MVTEC_ROOT --shots 0 --max-refs 50 \
    --coreset 10000 --coreset-method greedy --fp16 --layers 17 --agg max
# 优化后
python scripts/demo_mvtec.py --data-root $MVTEC_ROOT --shots 0 --max-refs 50 \
    --coreset 10000 --coreset-method greedy --fp16 --layers 11,17 --agg topk_mean --agg-k 5
```

| 类别 | 基线 (L17, max) | + 多层 + top-k | Δ |
|---|---|---|---|
| toothbrush | 87.8 | 100.0 | +12.2 |
| screw | 87.7 | 92.4 | +4.7 |
| capsule | 95.9 | 98.4 | +2.5 |
| cable | 98.3 | 99.7 | +1.4 |
| hazelnut | 99.6 | 100.0 | +0.4 |
| carpet | 99.6 | 99.9 | +0.3 |
| pill | 98.8 | 99.0 | +0.2 |
| bottle | 100.0 | 100.0 | 0 |
| grid | 100.0 | 100.0 | 0 |
| leather | 100.0 | 100.0 | 0 |
| metal_nut | 100.0 | 100.0 | 0 |
| tile | 100.0 | 100.0 | 0 |
| wood | 99.4 | 99.4 | 0 |
| zipper | 100.0 | 99.8 | −0.2 |
| transistor | 96.3 | 95.4 | −0.9 |
| **均值 (15)** | **97.6** | **98.9** | **+1.3** |

多层融合 + top-k 聚合的增益**恰好集中在单层基线最弱的类别**（toothbrush、screw、capsule）；
已到天花板（100）的类别保持不变。其余优化（**Excess 归一化**、**阈值标定**、**聚类分层建库**）
针对的是**工业运行点** —— 名义 vs 实际过杀的偏差、以及多模态 OK 的覆盖 —— 这些 MVTec 的标准
AUROC 口径体现不出来；详见 `docs/METHOD_NOTES.md`。

## 目录结构

```
anoco/                    核心库（骨干无关）
  retrieval.py            §3.2 锚定检索（+ 分块 / 稀疏 / top-k）
  graph.py                §3.3–3.4 双部图边权
  solver.py               §3.5 锚定 Laplacian 闭式解
  scoring.py              §3.6 非一致性能量 + 聚合
  membank.py              记忆库 + 贪心 coreset
  normalization.py        excess / z-score 归一化
  calibration.py          分数校准
  metrics.py              AUROC / AUPR / F1 / PRO（numpy + scipy）
  bank_builder.py         聚类分层建库 + 阈值标定
  inspector.py            生产推理封装
  features/dinov3.py      DINOv3 提取器（多层融合）
configs/                  L/16（论文）、B/16（速度）、production 模板
scripts/                  权重准备、MVTec demo、建库 / 推理
tests/                    4 组合成数据验证测试
docs/METHOD_NOTES.md      通用工程技术说明
```

## 引用

请引用原论文（[CVPR 2026，开放获取](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)）：

```bibtex
@InProceedings{Seo_2026_CVPR,
  author    = {Seo, Jungwook and Kim, Minjeong and Lee, Younkwan and Shin, Seungho and Baik, Sungyong},
  title     = {Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  pages     = {21336-21345}
}
```

本仓库是非官方复现；也欢迎引用其地址。

## 许可与声明

- **代码**：Apache-2.0（见 `LICENSE`）。Copyright 2026 Judy-Jim。
- **DINOv3 权重**：受 Meta 独立的 **DINOv3 License** 约束 —— 本仓库不包含。
- 与 ANoCo 作者及 Meta AI 无关联。“PatchCore”“DINOv3”“MVTec AD” 均归各自所有者所有。
