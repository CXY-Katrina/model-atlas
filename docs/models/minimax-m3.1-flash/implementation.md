---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879
Content-Status: verified
---

# 实现（vLLM 调用链）

固定版本：`vllm-project/vllm @ 8fae0ff58bae8bde68b52bf3e547432250763e5c`（2026-10-23）。
M3.1-Flash 与 M3 共用同一 `model_type=minimax_m3_vl` 与同一套实现；
M3.1-Flash 的差异由 HF config（全层 sparse、量化配置）与部署参数（NVFP4 KV、DSpark）驱动，
不存在独立的 "m3.1" 模型文件。

## 注册与入口

- `vllm/model_executor/models/registry.py`：
  `MiniMaxM3SparseForCausalLM` / `MiniMaxM3SparseForConditionalGeneration`
  → `vllm.models.minimax_m3`；`MiniMaxM3MTP` 同样注册（M3.1-Flash 无 MTP 配置，不实例化）。
- `vllm/models/minimax_m3/__init__.py`：按平台分流——NVIDIA → `nvidia/model.py`，
  ROCm → `amd/model.py`（本页证据为 NVIDIA 路径）。

## 层选择

`nvidia/model.py`：
- `_sparse_attention_layer_ids(config)`（L99–107）：`sparse_attention_freq[i] != 0`
  → M3.1-Flash 返回 {0..59}，**全部层**使用 `MiniMaxM3SparseAttention`。
- `_is_moe_layer(config, layer_id)`（L110–115）：`moe_layer_freq[layer_id] != 0`
  → L0–2 为 dense MLP，L3–59 为 MoE。
- `MiniMaxM3DecoderLayer.__init__`（L678–762）据此组装 attention 与 FFN。

## SparseAttention.forward 调用链（nvidia/model.py L582–675）

1. `qkv, _ = self.qkv_proj(hidden_states)` — 五路融合 GEMM
   （`MinimaxM3QKVParallelLinearWithIndexer`，L444–455）。
2. `ops.fused_minimax_m3_qknorm_rope_kv_insert(...)`（L625–650）——单 kernel 完成：
   主 Q/K 与 index Q/K 的 per-head Gemma norm、partial RoPE（共享 cos_sin_cache）、
   K/V 量化写入主 paged KV cache、Kidx 写入 index key-only cache；
   NVFP4 KV 时使用 `kv_k_scale/kv_v_scale`（L648–649，源自 `set_default_quant_scales`）。
   Q 与 index_q 被 de-interleave 到独立 buffer；index_q dtype 对齐 index cache。
3. `_run_attention`（L657–675，eager break）：先 `self.indexer(index_query)`
   把 Top-16 块写入共享 `topk_indices_buffer`，再 `self.impl.forward(...)`
   读取该 buffer 执行稀疏 attend；最后 `o_proj`（RowParallel，reduce_results=False，
   all-reduce 与 post norm 融合）。

## Indexer（common/indexer.py）

- `MiniMaxM3IndexerCache`（L102–）：独立注册的 key-only paged side cache
  （独立 prefix，独立 slot_mapping）。
- `MiniMaxM3Indexer`（L578–）：持有 score/top-k 参数
  （topk_blocks=16、sparse_block_size=128、init_blocks=0、local_blocks=1、score_type=max），
  forward 写 `topk_indices_buffer[token, index_head, topk]`（int32）。
- `select_indexer_impl_cls`（L495–）：按 `indexer_kv_dtype` 选择实现；
  支持 bf16 / fp8_e4m3 / nvfp4。博客推荐 nvfp4（`--attention-config '{"indexer_kv_dtype":"nvfp4"}'`）；
  main 源码 L533–535 对 nvfp4 仍 `raise NotImplementedError`（CuteDSL indexer 待合并），
  prebuilt image `vllm/vllm-openai:minimax-m31-flash` 已支持。

## 主 attend 后端选择（common/sparse_attention.py）

`select_main_backend_and_impl_cls`：
- Blackwell SM100 且 topk_blocks ∈ {4,8,16,32} 且非 fp8_e5m2 → MSA kernel；
- KV dtype 为 NVFP4 时**只允许** MSA 后端（否则 ValueError）→
  `MiniMaxM3SparseMSANvfp4Backend`（nvidia/sparse_attention_msa.py L76–98，
  `supported_kv_cache_dtypes=["nvfp4"]`）。
- ROCm → AITER sparse PA；其余 → Triton。
- MSA decode 内部可选 CUTLASS sparse decode（`msa_cutlass_sparse_decode.py`）。

## MoE（nvidia/model.py L195–289）

`MiniMaxM3MoE.forward`：`gate`（GateLinear，FP32 in/out）产生 router_logits →
`FusedMoEFactory(...)` 内部完成 sigmoid、correction-bias Top-4、归一化、
`routed_scaling_factor=2.0`、专家计算与 shared expert 融合。
`activation="swigluoai_uninterleave"`（w13 packed 为 [all gates; all ups]）。

## 权重加载（nvidia/model.py L944–1039）

- 五路融合映射：`.q_proj/.k_proj/.v_proj/.index_q_proj/.index_k_proj → .qkv_proj`
  （stacked_params_mapping L952–961）。
- 量化 scale：`weight_scale_inv` → `weight_scale`（L975–976，适配 MXFP8 checkpoint 命名）。
- MoE 专家：`w1=gate / w2=down / w3=up`（get_expert_mapping L934–942）。
- `mtp.` 前缀权重跳过（L970–971）——M3.1-Flash 本就无 MTP 权重。

## 融合边界汇总

| 融合 kernel | 覆盖数学步骤 | 证据 |
| --- | --- | --- |
| fused_minimax_m3_qknorm_rope_kv_insert | Q/K/index QK norm + partial RoPE + K/V/Kidx 量化写入 cache | L625–650 |
| fused_allreduce_gemma_rms_norm | TP all-reduce + residual add + Gemma RMSNorm | L770–787 |
| FusedMoE | sigmoid + bias Top-4 + 归一化 ×2.0 + 专家 + shared 合并 | L251–272 |
| MSA / CUTLASS sparse decode | block 内 QK、scale、mask、softmax、PV | nvidia/sparse_attention_msa.py |

## DSpark 投机解码（外部组件）

M3.1-Flash 无 MTP 模块；vLLM 部署以 `--speculative-config '{"method":"dspark", ...}'`
启用 DSpark drafter（非因果 GQA + 4096 token 滑窗；NVIDIA 7 / AMD 3 投机 token，
FlashInfer PrimiTS / TRITON_ATTN 后端）。draft head 随 checkpoint 一起发布
（HF 模型卡 "the release includes the draft head"），vLLM 以独立 speculative config 挂接，
主模型 decode 的 `decode_query_len` 相应变为 `1 + num_speculative_tokens`
（common/sparse_attention.py 的 decode metadata 注释）。