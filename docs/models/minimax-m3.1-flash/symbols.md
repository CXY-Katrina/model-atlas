---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: hf-config.json@main(2026-10-26); vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879
Content-Status: verified
---

# 符号与参数

## 形状轴

| 符号 | 含义 | 取值/来源 |
| --- | --- | --- |
| B | batch 内请求数 | runtime |
| S / Nq | 本轮 query token 数 | runtime |
| T | 含历史的 KV 总长度 | runtime |
| TP | tensor parallel size | 部署配置（博客推荐 TP4） |

## 结构参数（text_config）

| 符号 | 配置键 | 值 | 备注 |
| --- | --- | --- | --- |
| H | hidden_size | 6144 | |
| L | num_hidden_layers | 60 | |
| Nₕ | num_attention_heads | 64 | query heads |
| Nₖᵥ | num_key_value_heads | 4 | GQA 比 16:1 |
| Dₕ | head_dim | 128 | |
| V | vocab_size | 200064 | |
| S_max | max_position_embeddings | 1048576 | 1M 上下文 |
| ε | rms_norm_eps | 1e−6 | Gemma RMSNorm |
| θ_base | rope_theta | **10000000** | M3 为 5e6 |
| Dᵣ | rotary_dim | 64 | partial_rotary_factor 0.5 × Dₕ |
| H_expert | intermediate_size | 3072 | 单 routed expert 宽度 |
| H_dense | dense_intermediate_size | 12288 | L0–2 dense MLP 宽度 |
| H_shared | shared_intermediate_size | 3072 | shared expert 宽度 |
| E | num_local_experts | 128 | routed experts |
| K | num_experts_per_tok | 4 | 每 token 激活专家 |
| E_shared | n_shared_experts | 1 | |
| s_route | routed_scaling_factor | 2.0 | |
| α / β / c | swiglu_alpha / swiglu_beta / swiglu_limit | 1.702 / 1.0(默认) / 7.0 | β 缺省，vLLM 默认 1.0 |
| — | moe_layer_freq | L0–2: 0，L3–59: 1 | dense MLP ↔ MoE 分界 |
| — | qk_norm_type | per_head | |
| — | scoring_func / use_routing_bias | sigmoid / true | |

## Sparse attention 参数（text_config.sparse_attention_config）

| 符号 | 配置键 | 值 | 备注 |
| --- | --- | --- | --- |
| D_idx | sparse_index_dim | 128 | index head 维度 |
| N_idx | sparse_num_index_heads | 4 | == Nₖᵥ，按 GQA group 对齐 |
| K_block | sparse_topk_blocks | 16 | 每 query 每组选中块数 |
| B_block | sparse_block_size | 128 | tokens/block == KV page size |
| B_init | sparse_init_block | 0 | 无 sink 块优先 |
| B_local | sparse_local_block | 1 | 当前块强制入选 |
| — | sparse_score_type | max | 块归约方式 |
| — | sparse_attention_freq | 全 1（60 项） | **M3.1-Flash：L0–2 也是 sparse** |
| — | sparse_disable_index_value | 全 1（60 项） | index V/O 投影从不创建 |

## 量化（quantization_config）

| 项 | 值 | 作用范围 |
| --- | --- | --- |
| quant_method | mxfp8 | 默认线性层权重（[1,32] 块 scale，group_size 16，activation dynamic） |
| moe_quant_algo / format | NVFP4 / nvfp4-pack-quantized | routed experts（及 fused shared 见博客 MegaMoE） |
| ignored_layers | router gates ×57、lm_head、embed_tokens、multi_modal_projector、patch_merge_mlp、vision_tower | 不量化 |
| KV cache dtype | nvfp4（部署 `--kv-cache-dtype nvfp4`） | 主 KV + index cache（博客推荐 `indexer_kv_dtype=nvfp4`）；34,560 B/token（FP8: 61,440） |

## 数值事实

| 事实 | 值 | 来源 |
| --- | --- | --- |
| 总参数 / 每 token 激活 | ≈428B / ≈23B | HF 模型卡（"Approximately 428B total; 23B activated per token"） |
| checkpoint 大小 | ≈250.1 GB（NVFP4 专家 + MXFP8 其他）；较 M3-MXFP8 451.5 GB 小 44.6% | vLLM 博客 2026-10-23 |
| 4×B300 · 256K 序列并发 | 76（M3 FP8 KV 为 28，≈2.7×） | vLLM 博客 2026-10-23 |
| 每 query 可见 keys 上限 | 2048 = 16 块 × 128 | 结构推算（博客 MSA primer 一致） |
| decode indexer 扫描量 | 128 B/token/层（fp8 index cache；bf16 为 256 B；dense FP8 K/V 为 1024 B，1/8） | vLLM 博客 MSA primer |
| vision tower | CLIP，Hᵥ=1280，32 层，patch 14，image_size **672** | HF config（M3 为 2016） |
| 图像 token | image_seq_length 576；patch_merge 2×2 | HF config |

## 页面符号约定

- `Xₗ / U / Xₗ₊₁`：层输入 / attention 后 residual / 层输出。
- `X̂ / Û`：input norm / post-attn norm 输出。
- `Q̃ K̃ → Qʳ Kʳ`：norm 后 → partial RoPE 后。
- `Qidx / Kidx`：index 分支投影；`𝒦idx`：key-only index cache。
- `𝒮`：选中块集合；`K_sel ≤ 2048`：候选 token 数。
- 与 M3 页面相同符号含义一致；不同数值（θ_base、层划分）以本表为准。