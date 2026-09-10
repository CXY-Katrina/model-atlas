# Feature: Add Hunyuan4 (Hy4-preview) to Model Atlas

Status: draft

## Goal

把腾讯 Hy4-preview（内部名 `hy_v4`，770B 总参 / 49B 激活 MoE）加入 Model Atlas，
与 MiniMax-M3 同等粒度：算子级交互数据流图 + 每节点 IO / KaTeX 公式 / 代码定位 /
checkpoint 权重绑定 + config.json 全景表。

## 结构要点（与 MiniMax-M3 的差异）

调研基础：vllm 主仓 `vllm/models/hy_v4/`（PR #54160，upstream/main `40824284bc`）
+ 官方 ModelScope config.json / safetensors index（Tencent-Hunyuan/Hy4-preview）。
每条事实带 file:line 出处，来源调研报告（贡献者本地存档）。

| 维度 | MiniMax-M3（现有） | Hunyuan4（新增） |
|---|---|---|
| 注意力 | GQA 64Q/4KV + QK-norm + 50% RoPE | **MLA**（q_lora 2048 / kv_lora 512，nope 192 + rope 64 interleaved，W_UK 吸收 decode） |
| 稀疏机制 | block 级：indexer 打分 → block max → top-16 blocks | **token 级**：lightning indexer FP8 打分 → top-2048 tokens；21 full 层算索引、57 shared 层读共享 buffer |
| 残差 | 普通 residual（fused add-norm） | **iHC 4 通道**：pre/post 双 sigmoid 门控（hc_fn 24576→8），层间 [T,4,6144]，末端 hc_head 合流 |
| 输出门控 | 无 | **gated MLA**：attn_out × σ(linear_gate(h)) elementwise |
| sink | 无 | learnable sink（每头 fp32 logit 折进 softmax 分母） |
| MoE | 128 experts top-4 + 1 shared，swigluoai | 256 experts sigmoid top-8 ×2.827 + 1 shared，SwiGLU clamp ±10；L0 dense |
| 层分布 | L0–2 dense + L3–59 sparse | 全 78 层 DSA；indexer full 层 {0,1,5,9,…,77} |
| MTP | num_mtp_modules=7（未展开） | 1 层 MTP（eh_proj + 单层 decoder 无 iHC，共享 embed/lm_head） |
| 多模态 | vision tower（已有 config 组） | 纯文本（代码 + config 双重确认无多模态） |

## Architecture plan

现状：`app/page.tsx`（861 行）把 M3 的图数据、公式表（LATEX_BY_ID / FORMULA_TERMS_BY_ID）、
代码面板（CODE_BY_ID）、config 表（CONFIG_GROUPS）、IO 绑定（INPUT_OVERRIDES / NEXT_BY_ID）
与通用渲染组件（GraphSurface / Op / Tensor）耦合在一个文件；`MODEL_REGISTRY` 已有
多模型占位但无切换机制。

两阶段贡献，各自可独立 review：

### Phase 1 — 渲染器与模型数据解耦（架构 PR）

1. 新建 `app/models/` 目录；把 M3 专属数据整体迁移到 `app/models/minimax-m3.tsx`，
   导出统一接口 `ModelPageData`（图节点表、公式表、代码面板、config 组、IO 绑定、
   registry 元信息）。
2. `page.tsx` 保留通用渲染与交互；顶部模型切换由 `MODEL_REGISTRY` 驱动
   （enabled 才可选，disabled 显示"待添加"），URL hash 同步（`#/hy4`）。
3. `model-data.tsx` 的 M3 checkpoint 数据随之迁入 `app/models/minimax-m3.tsx`。
4. 现有测试全部保持通过；为 registry 切换补最小测试。

### Phase 2 — Hy4 模型数据（内容 PR）

`app/models/hy4.tsx`，节点清单（decode 视角主图，prefill 差异在节点 formulaNote 标注）：

- iHC 侧：`hc-attn-pre`（flatten→RMS 统计→hc_fn→双 sigmoid→压缩）、`hc-attn-post`、
  `hc-mlp-pre/post`、`hc-head`（末端合流）
- attention 侧：`rmsnorm` → `fused-qkv-a`（6144→2624）→ `q-a-norm` → `q-b-proj`
  → `w-uk-absorb`（bmm）→ `kv-rmsnorm-rope-cache`（写 cache）→ indexer 链
  （`wk-weights-proj` → `k-norm` → `idx-rope` → `fp8-quant` → `idx-scores`
  → `topk-2048` → 共享 buffer）→ `sparse-attn`（+ learnable sink rescale）
  → `w-uv-upproj` → `gated-mla` → `o-proj`
- FFN 侧：`router`（fp32 sigmoid + bias 修正 + renorm ×2.827）→ routed experts
  （SwiGLU clamp ±10）+ shared expert → combine
- 输出侧：`final-norm` → `lm-head`（fp32）；MTP 侧栏（enorm/hnorm → eh_proj → mtp-block）
- config 全景表：官方 config.json 全字段（含 indexer_types / hc_* / learnable_sink 等）
- 每个 op 绑定真实权重（key/shape/dtype/shard/params，来源 safetensors index）

公式表来源：iHC 门控数学（hc.py:97-186）、MLA 吸收（mla_attention.py:933-1027）、
DSA 打分 score=(1/√32)(1/√128)Σ w_h⟨q,k⟩（attention.py:245-263）、sink
out·rL/(rL+e^(sink−rM))（flashmla_sparse.py:44-51）、路由（moe.py + fused_topk_bias_router）。

## Non-goals

- 不做 NPU（vllm-ascend）算子视图——本仓现有页面均为 vllm 主仓 CUDA/通用视角；
  NPU 差异可作为后续 feature（vllm-ascend PR #15377 数据已调研存档）。
- 不修改 M3 现有节点的任何事实内容（Phase 1 仅做物理搬迁）。

## Milestones

1. spec 合入 `.scratch/hunyuan4/spec.md`（本文件）
2. Phase 1 架构 PR（含迁移后全部测试通过）
3. Phase 2 Hy4 数据 PR（先 attention 链 + config 表，MoE/iHC/MTP 可拆子 PR）

## Comments
