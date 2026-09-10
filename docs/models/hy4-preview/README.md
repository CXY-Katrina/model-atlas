Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (Tencent Hunyuan V4 preview · HYV4ForCausalLM)
Source-Revisions: vllm@b2f685834a6456197e7033966fdef52a23f1abcd; vllm-ascend@ad86348b0cb2324d643df6a496f2d9c3879e4481; HF tencent/Hy4-preview config.json + model.safetensors.index.json @ 2026-09-09
Content-Status: verified

# hy4-preview 成果导航

Tencent HYV4(Hunyuan V4 preview)是一个 770B 总参数 / 约 49B 激活的稀疏 MoE 因果语言模型,
78 层全部使用 DSA(DeepSeek Sparse Attention,即 MLA + lightning indexer),并以 iHC
(independent Hyper-Connections,4 通道残差)取代单一 residual stream。官方 checkpoint:
[Hugging Face tencent/Hy4-preview](https://huggingface.co/tencent/Hy4-preview)。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| [architecture.md](./architecture.md) | 顶层拓扑、层族、iHC 通道模型、缓存与推理阶段 |
| [formulas.md](./formulas.md) | 逐算子公式(iHC / MLA / indexer / gated MLA / sink / MoE / MTP) |
| [symbols.md](./symbols.md) | 符号与形状约定(全部数值取自 checkpoint config.json) |
| [implementation.md](./implementation.md) | vLLM 源码调用链、融合边界、权重映射、后端差异 |
| [evidence.md](./evidence.md) | 节点 / 公式 / 权重 → 固定源码与配置证据映射 |
| [diagrams/](./diagrams/) | Archify typed JSON 图示源(交付 HTML 在 `public/models/hy4-preview/diagrams/`) |

## 变体与适用范围

- 本仓页面描述 **NVIDIA vLLM 路径**(vllm `vllm/models/hy_v4/nvidia/`,commit `b2f6858`)。
- vllm-ascend 文档已给出 Atlas 800I A3 的 W8A8 部署指引(约 762GB 权重),但 ascend 实现
  尚未合入 vllm-ascend 仓;页面只引用其部署事实,不描述未合入代码。
- checkpoint 为 BF16、131 个 safetensors 分片;`tie_word_embeddings=false`
  (`lm_head.weight` 独立存放);1 个 MTP draft 层(`model.mtp_layers.0`)在
  `HYV4ForCausalLM.load_weights` 中被丢弃,由独立的 `HYV4MTP` 类消费。

## 验证状态

`Content-Status: verified` — 同版双审(r3)通过。
