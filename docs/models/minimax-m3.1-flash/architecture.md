---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: hf-config.json@main(2026-10-26); vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879
Content-Status: verified
---

# 架构

## 输入与输出

文本 token ids（可混入图像/视频占位 token，`image_token_index=200025`、`video_token_index=200026`）
经 Embedding（`V=200064`，`H=6144`，BF16，**不量化**）进入 60 层 decoder，
最后经 Final Gemma RMSNorm 与 LM head（不量化，与 embedding 不共享：
`tie_word_embeddings=false`）输出 `[B,S,V]` logits。
多模态输入由独立 vision tower（CLIP，image_size 672）+ patch-merge + projector
生成图像 embedding 后按占位 token 融合；vision 支路本文不展开。

## 层族划分（与 M3 的关键差异）

| 层 | attention | FFN | 说明 |
| --- | --- | --- | --- |
| L0–L2（3 层） | **MiniMax Sparse Attention** | Dense SwiGLU-OAI MLP（`dense_intermediate_size=12288`） | M3 此处为 dense GQA；M3.1-Flash 改为 sparse attention |
| L3–L59（57 层） | MiniMax Sparse Attention | Top-4 MoE（128 experts）+ 1 shared expert | 与 M3 相同 |

划分依据：`sparse_attention_freq` 全 1（60/60）；`moe_layer_freq` 前 3 为 0、其余为 1
（vLLM `_sparse_attention_layer_ids` / `_is_moe_layer`，nvidia/model.py L99–115）。

## 单层数据流

每层：`Xₗ → (residual 分离) → input Gemma RMSNorm → Attention 分支 →
fused add + post-attn Gemma RMSNorm → FFN 分支 → (deferred residual merge) → Xₗ₊₁`。

- Norm 均为 Gemma 风格 RMSNorm：`y = x/√(mean(x²)+ε)⊙(1+γ)`，`ε=1e−6`
  （nvidia/model.py L118–146，FlashInfer `gemma_rmsnorm` / `gemma_fused_add_rmsnorm`）。
- 残差合并延迟融合：FFN 输出不在本层 all-reduce/add，由下一层 input norm 的
  `fused_allreduce_gemma_rms_norm` 完成（DecoderLayer.forward L764–790，
  融合编排 L866–873）。

## Sparse Attention 分支（全部 60 层）

一次融合 GEMM 产生五路输出 `[q|k|v|index_q|index_k]`（`MinimaxM3QKVParallelLinearWithIndexer`）：

- 主分支：Q(64 heads)、K/V(各 4 heads) → per-head Q/K Gemma RMSNorm →
  Partial RoPE（前 64/128 维旋转，`rope_theta=1e7`）→ K/V 写入主 paged KV cache。
- Index 分支：Qidx（4 heads）、Kidx（1 head 共享）→ index Q/K Gemma RMSNorm →
  同一组 position 的 RoPE → Kidx 写入**独立 key-only index cache**。
- Lightning Indexer：每个 GQA group 的 index query 对完整 index-K 历史打分
  → 未来 key 置 −∞ → 每 128 token 的块取 max（`sparse_score_type=max`）
  → 每 query 每 group 选 Top-16 逻辑块（local block 强制保留，`sparse_local_block=1`；
  `sparse_init_block=0` 即首块无 sink 优先）→ 写入共享 `topk_indices_buffer`。
- 主 attention 只对选中块内 token 做精确 causal GQA（每 query ≤ 2048 keys）。

vLLM 中以上 norm/RoPE/cache 写入由单个融合 kernel
`fused_minimax_m3_qknorm_rope_kv_insert` 完成（nvidia/model.py L625–650），
indexer 与 attend 通过共享 buffer 交接（`_run_attention` L657–675），
cudagraph 安全（eager break 仅包住选块+attend）。

## 缓存

| cache | 内容 | dtype（M3.1-Flash 原生） | 生产者/消费者 |
| --- | --- | --- | --- |
| 主 paged KV | 每 token 4 KV heads × 128 维 ×(K,V) | **NVFP4**（QAT 原生；34,560 B/token，FP8 为 61,440 B/token） | 融合 insert kernel 写入；sparse attend kernel 按 block_indices+block_table 读取 |
| index K side cache | 每 token 1 个 128 维 index K（无 index V） | bf16 / fp8_e4m3 / nvfp4（博客推荐 nvfp4；`--attention-config indexer_kv_dtype`） | 同一融合 kernel 写入；indexer score kernel 读取 |

NVFP4 KV 仅 vLLM SM100 MSA 后端支持（common/sparse_attention.py
`select_main_backend_and_impl_cls`：非 MSA 或 AITER 路径直接报错）；
page size == sparse block size == 128。

## FFN 分支

- L0–L2 Dense MLP：`gate_up_proj`（MergedColumnParallel，6144→2×12288）→
  SwiGLU-OAI（clamp ±7；`min(g,7)·σ(1.702·min(g,7))·(clip(u,−7,7)+1.0)`）→
  `down_proj`（RowParallel，12288→6144）。
- L3–L59 MoE：FP32 router（`gate`，6144→128 logits）→ FusedMoE 内
  sigmoid → +correction bias 选 Top-4（bias 不进权重）→ 归一化 ×`routed_scaling_factor=2.0`
  → 专家加权求和；共享专家（3072 宽）始终激活；最终 routed+shared 相加。
- 专家权重 **NVFP4 packed**（`moe_quant_algo=NVFP4`）；router/embed/LM head/vision 不量化；
  其余线性层 **MXFP8**（weight_block [1,32]，group_size 16，activation dynamic）。

## 量化与 checkpoint

`quantization_config.quant_method=mxfp8`：权重按 [1,32] 块存储 scale
（checkpoint 键 `weight_scale_inv`，加载时映射为 `weight_scale`，nvidia/model.py L975–976）；
MoE 专家为 NVFP4 packed 格式。checkpoint 约 250.1 GB（博客；对比 M3-MXFP8 451.5 GB，44.6% 更小，
M3 BF16 约 869 GB，第三方估算）。参数量与 M3 相同（总 ≈428B，每 token 激活 ≈23B；
HF 模型卡明确 "Approximately 428B total; 23B activated per token"）。

## 与 M3 的差异清单（本页读者向）

1. L0–L2 由 dense GQA 变为 sparse attention（FFN 仍为 dense MLP）。
2. `rope_theta` 5e6 → 1e7。
3. 移除 MTP（7 个 `num_mtp_modules`）；投机解码改用 DSpark drafter
   （非因果 GQA + 4096 滑窗；HF 模型卡确认 draft head 随权重发布，
   NVIDIA 7 / AMD 3 投机 token 为部署配置）。
4. 原生 NVFP4 KV cache 与 NVFP4 MoE 权重；其余线性层 MXFP8。
5. vision `image_size` 2016 → 672。

## 已知限制

- AMD 后端（`amd/model.py`）本次未验证。
- checkpoint shard 文件清单未获取；页面权重条目的 shard 定位标注"未验证"。
- 视觉塔内部结构不展开。