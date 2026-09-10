Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (HYV4ForCausalLM)
Source-Revisions: HF config.json @ 2026-09-09
Content-Status: verified

# hy4-preview 符号表

全部数值取自官方 checkpoint config.json(tencent/Hy4-preview),非 vllm 配置类默认值。

## 轴与形状

| 符号 | 含义 | 值 / 来源 |
| --- | --- | --- |
| B | batch(请求)数 | runtime |
| S | 序列长度(参考记法,与 B 组成 `[B,S,·]`);页面运行时常量用 T 表示 | runtime |
| T | 本轮 token 数(query 数与写入 cache 的 token 数;缓存/候选相关 shape 中同一 token 轴) | runtime |
| H | hidden_size | 6144 |
| L | num_hidden_layers | 78 |
| V | vocab_size | 120832 |
| S_max | max_position_embeddings | 1048576 |
| C_hc | hc_mult(iHC 通道数) | 4 |
| R_q | q_lora_rank | 2048 |
| C_kv | kv_lora_rank | 512 |
| D_nope | qk_nope_head_dim | 192 |
| D_rope | qk_rope_head_dim | 64 |
| D_qk | qk_head_dim = D_nope + D_rope | 256 |
| D_v | v_head_dim | 256 |
| N_h | num_attention_heads | 64 |
| H_ffn | intermediate_size(L0 dense) | 18432 |
| H_exp | moe_intermediate_size | 2048 |
| E | n_routed_experts | 256 |
| E_sh | n_shared_experts | 1 |
| K | num_experts_per_tok | 8 |
| N_idx | index_n_heads | 32 |
| D_idx | index_head_dim | 128 |
| T_idx | index_topk | 2048 |
| g_fp8 | FP8 per-token-group 量化组大小(indexer 的 q 量化与 key-only cache 的 scale 分组共用) | 128(数值与 D_idx 相同,语义无关) |
| N_kv | num_key_value_heads(runtime MLA 的 KV 头组数) | 8 |
| L_mtp | num_nextn_predict_layers(MTP draft 层数) | 1 |
| ε_rms | rms_norm_eps | 1e−5 |
| ε_hc | hc_eps | 1e−6 |
| θ | rope_theta | 1e7 |
| c | swiglu_limit(仅 routed experts) | 10.0 |
| s_route | routed_scaling_factor | 2.827 |
| m_hc | hc_magnitude(post 门幅度) | 2.0 |

## 公式符号

| 符号 | 含义 |
| --- | --- |
| x / y | iHC 边界的输入 / 输出 |
| pre_i, post_i | iHC 第 i 通道的 pre/post sigmoid 门(i=0..3) |
| hc_fn, hc_head_fn | iHC 门投影(24576→8 / 24576→4,FP32) |
| hc_scale, hc_base | 门仿射参数(scale[2] / base[8];head 为 scale[1] / base[4]) |
| s_0, s_1 | hc_scale 的两个分量:pre 门 / post 门的缩放(初始 0.01) |
| q̃ / k̃ | q_lora / kv_lora RMSNorm 输出 |
| Q_h, K_h, V_h | 第 h 个 head 的 query(nope+rope)、key(nope)、value |
| k_pe | MQA 共享的 64 维 rope key |
| S_h, P, O | 第 h head 的 attention score / 概率 / 上下文(S_h 与序列长度 S 不同义) |
| sink_h | 第 h head 的可学习 attention sink(FP32,checkpoint 初始 0) |
| lse | attention 的 log-sum-exp(内核内部量) |
| Qidx_{t,h} | indexer 第 h head 的 query |
| k_j | indexer 第 j 个 token 的 key(LayerNorm 后,FP8 cache) |
| w_{t,h} | indexer head 权重(wk_weights_proj 的 32 通道输出) |
| q_scale | q 的 FP8 per-token-group 量化 scale(ue8m0) |
| I_t | token t 的 top-2048 索引集合(写共享 buffer);候选不足时该行以 −1(no-token sentinel)填充 |
| r^{rt}, s^{rt} | router logits(FP32） / 其 sigmoid 分数 |
| r_hc | iHC 的 rsqrt 因子 `1/√(mean(x²)+ε)`(与 router logits 无关) |
| s^{idx}_j | indexer 第 j 个候选的打分(与 router 分数无关) |
| expert_bias | e_score_correction_bias,FP32[256],只影响选择 |
| ŵ_e | expert e 的混合权重(归一化 × 2.827) |
| E_e, E_shared | 路由 expert / shared expert 函数 |
| g, u | expert SwiGLU 的 gate / up 分支 |
| Û | post_attention_layernorm 输出 |
| h_e, h_p | MTP 的 embed 分支 / 主干 hidden 分支(enorm/hnorm 后) |
| TP | tensor parallel size(只影响分片,不改变数学);页面权重 shape 为 checkpoint 全局 shape |

### 同名不同义对照

| 记法 | 含义 A | 含义 B |
| --- | --- | --- |
| S | 序列长度(`[B,S,·]`,轴表) | attention score 记作 `S_h`(带 head 下标,公式符号表) |
| r | iHC 的 rsqrt 因子 `r_hc` | router logits `r^{rt}` |
| s | router σ 分数 `s^{rt}` | indexer 打分 `s^{idx}_j` |
| 128 | `D_idx`(index 头维) | `g_fp8`(FP8 量化组大小,g_fp8 仅为记法:源码为 `quant_block_size`) |

## checkpoint 权重命名约定(节选)

| checkpoint key | 运行时归宿 |
| --- | --- |
| `model.layers.{l}.q_a_proj` | `fused_qkv_a_proj` shard 0 |
| `model.layers.{l}.kv_a_proj_with_mqa` | `fused_qkv_a_proj` shard 1 |
| `model.layers.{l}.indexer.wk` | `wk_weights_proj` shard 0(weights_proj 为 shard 1) |
| `model.layers.{l}.mlp.gate.e_score_correction_bias` | `expert_bias` |
| `model.layers.{l}.hc_attn_layer.hc_pre.hc_fn` | `hc_fn.weight` |
| `model.hc_head.hc_head_fn` | `hc_head_fn.weight` |
| `model.layers.{l}.mlp.experts.{gate_up_proj,down_proj}` | 融合 all-experts 格式,gate_up 按 dim −2 切成 w1/w3 |
