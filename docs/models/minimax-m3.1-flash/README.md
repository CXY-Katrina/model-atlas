---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: hf-config.json@main(2026-10-26, sha256 3bea745c…); vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879; vllm-blog 2026-10-23
Content-Status: verified
---

# MiniMax-M3.1-Flash 模型知识

MiniMax-M3.1-Flash 是 MiniMax-M3 家族的推理效率变体（2026-10-17 发布 checkpoint，
vLLM 2026-10-23 day-0 支持）。架构与 M3 相同（`model_type: minimax_m3_vl`，
`MiniMaxM3SparseForConditionalGeneration`），核心变化是**全层稀疏注意力 + 原生低比特量化**：
60 层全部启用 MiniMax Sparse Attention（MSA），KV cache 原生 NVFP4，MoE 路由专家权重 NVFP4、
其余线性层 MXFP8，并移除了 M3 的 EAGLE 式 MTP 模块（投机解码改由 DSpark drafter 承担，
draft head 随权重一起发布）。

## 成果导航

| 文件 | 内容 |
| --- | --- |
| [architecture.md](architecture.md) | 层族划分、数据流、缓存与推理阶段 |
| [formulas.md](formulas.md) | 逐算子公式、shape 与假设 |
| [symbols.md](symbols.md) | 符号、参数与数值事实表 |
| [implementation.md](implementation.md) | vLLM 源码调用链与量化/融合边界 |
| [evidence.md](evidence.md) | 节点/公式/参数到固定源码的映射 |
| [diagrams/](diagrams/) | Archify typed JSON 图示源（页面渲染 `public/models/minimax-m3.1-flash/diagrams/` 的交付 HTML） |

## 适用后端与阶段

- vLLM NVIDIA（SM100 MSA / Triton 回退）：主证据后端。
- vLLM AMD（ROCm, `amd/model.py`）：存在，本次未验证，不作事实展示。
- prefill / decode 均覆盖；DSpark 投机解码：draft head 随权重发布，vLLM 侧经
  `--speculative-config method=dspark` 挂接，本页仅记录接口关系。

## 验证状态

- HF config、vLLM 调用链、量化方案：已核实（见 evidence.md）。
- checkpoint shard 清单（safetensors index）：未获取，权重的分片位置标注"未验证"。
- 视觉塔：仅记录配置事实（CLIP、image_size 672），不展开内部结构。