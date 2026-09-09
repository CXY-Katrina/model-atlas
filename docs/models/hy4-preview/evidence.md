Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (HYV4ForCausalLM)
Source-Revisions: vllm@b2f685834a; vllm-ascend@ad86348b0; HF config.json @ 2026-09-09
Content-Status: verified

# hy4-preview 证据映射

固定源:`.scratch/add-hy4/sources/`(hash 见 `sources/MANIFEST.md`)。
`model.py`/`attention.py`/`hc.py`/`moe.py`/`mtp.py` 均 vllm `b2f685834a`;
config = HF `tencent/Hy4-preview` `config.json`(2026-09-09 抓取);
index = 同仓 `model.safetensors.index.json`(131 shards)。

| 页面/节点/公式 id | 后端与阶段 | 语义/shape | 源文件@SHA:符号 | 配置或权重证据 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| F-IHC-PRE / 节点 ihc-pre | NVIDIA·所有层×2 | [T,4,6144]→[T,6144]+门[T,4] | attention 无;hc.py@b2f6858:HYV4HCPreLayer.forward | config: enable_ihc,hc_mult=4,hc_eps=1e−6,hc_magnitude=2.0;权重 hc_attn_layer/hc_mlp_layer.hc_pre.hc_fn(78×2 处) | 已核验 |
| F-IHC-POST / ihc-post | NVIDIA·所有层×2 | [T,6144]×4 通道散回 | hc.py:HYV4HCPostLayer.forward | 权重索引无 post 参数(无参数模块) | 已核验 |
| F-IHC-HEAD / ihc-head | NVIDIA·模型出口 | [T,4,6144]→[T,6144] | hc.py:HYV4HCHeadLayer.forward;model.py L389 | 权重 model.hc_head.hc_head_fn/base/scale | 已核验 |
| 层调度(dense/moe) | NVIDIA | L0 dense,L1–77 MoE | model.py L130–143 | config: mlp_layer_types;权重 L0 mlp.{gate,up,down}_proj、L1–77 mlp.experts.* | 已核验 |
| F-A-PROJ / attn-proj | NVIDIA·所有层 | 6144→[2048,576] | attention.py L364–371,forward L669–683 | config: q_lora_rank=2048,kv_lora_rank=512,qk_rope_head_dim=64;权重 q_a_proj、kv_a_proj_with_mqa | 已核验 |
| F-A-UP / attn-kvb | NVIDIA·所有层 | 512→64×448 | attention.py L402–408 | config: qk_nope=192,v_head=256 | 已核验 |
| norm1 / norm2(节点 norm1、norm2) | NVIDIA·所有层×2 | RMSNorm ε=1e−5,γ [6144] | model.py L129/L144(HYV4DecoderLayer.__init__) | config: rms_norm_eps=1e−5;权重 input_layernorm.weight、post_attention_layernorm.weight(78×2 处) | 已核验 |
| qan / kvn(节点 qan、kvn) | NVIDIA·所有层 | q̃ [T,2048]、k̃ [T,512] RMSNorm | attention.py L167、L385/L401 | config: q_lora_rank=2048,kv_lora_rank=512;权重 self_attn.q_a_layernorm.weight、kv_a_layernorm.weight | 已核验 |
| MLA cache(节点 cache) | NVIDIA·所有层 | 每 token 512+64 压缩布局;dtype=cache_config 决定,默认 BF16,量化请求时后端升格 fp8_ds_mla | mla_attention.py L349–362(仅 quantized 升格);flashmla_sparse.py L261(use_fp8_kv_cache 判定)、supported dtypes auto/bfloat16 | config: kv_lora_rank=512,qk_rope_head_dim=64;部署选项 --kv-cache-dtype | 已核验 |
| F-A-ROPE / attn-rope | NVIDIA·所有层 | rope 64 维,interleaved | attention.py L416–435,forward L688–690 | config: rope_theta 1e7,max_position 1048576 | 已核验 |
| F-IDX / idx 链 | NVIDIA·21 个 full 层 | top-2048 索引 | attention.py L125–263(compute_skip_topk_layers L54) | config: indexer_types 78 项(full@{0,1,5,…,77}),index_n_heads=32,index_head_dim=128,index_topk=2048;权重仅这 21 层有 indexer.* | 已核验 |
| shared 复用 / idx-reuse | NVIDIA·57 层 | 读共享 buffer | attention.py L320–331,L756–757 | 权重索引:57 层无 indexer.* | 已核验 |
| F-A-SCORE+SINK / attn-core | NVIDIA·所有层 | 候选 2048 上精确 softmax | flashmla_sparse.py@b2f6858:sink 折叠;attention.py L482–518 | config: learnable_sink=true(init 0.0);权重 learnable_sink_param | 已核验 |
| F-A-GATE / attn-gate | NVIDIA·所有层 | attn⊙σ(gate),elementwise | attention.py L455–476,L710–731 | config: gated_mla=true,gating_type=elementwise;权重 linear_gate [64×256,6144] | 已核验 |
| o_proj | NVIDIA·所有层 | 16384→6144 | attention.py L409–415 | 权重 o_proj [6144,16384] | 已核验 |
| F-MOE router / moe-router | NVIDIA·L1–77 | 6144→256 FP32 | moe.py L122–129 | config: n_routed_experts=256;权重 mlp.gate.weight(F32)、e_score_correction_bias | 已核验 |
| F-MOE experts / moe-experts | NVIDIA·L1–77 | top-8,sigmoid,×2.827,clamp 10 | moe.py L144–170(FusedMoEFactory) | config: num_experts_per_tok=8,norm_topk_prob,routed_scaling_factor=2.827,swiglu_limit=10.0,n_group/topk_group=1;权重 experts.gate_up_proj/down_proj(融合) | 已核验 |
| F-MOE shared / moe-shared | NVIDIA·L1–77 | 1×SwiGLU 2048 无 clamp | moe.py L131–142(HYV4FeedForward) | config: n_shared_experts=1;权重 shared_experts.{gate,up,down}_proj | 已核验 |
| F-FFN / ffn 链 | NVIDIA·L0 | SwiGLU 18432 | moe.py L23–66 | config: intermediate_size=18432;权重 L0 mlp.* | 已核验 |
| F-MTP | NVIDIA·draft | enorm/hnorm/eh_proj,无 iHC | mtp.py L324–390;model.py L704–708(drop) | config: num_nextn_predict_layers=1;权重 model.mtp_layers.0.*(无 hc_*) | 已核验 |
| lm_head FP32 | NVIDIA·输出 | logits FP32 | model.py HYV4ForCausalLM.compute_logits | config: enable_lm_head_fp32=true;权重 lm_head.weight(shard 00053,独立) | 已核验 |
| 部署事实(W8A8/A3) | ascend 教程 | 762GB,Atlas 800I A3 | vllm-ascend@ad86348b0:Hy4-preview.md | `.github/vllm-main-verified.commit`=b2f6858 | 已核验 |

## 反向警示(不能出现的内容)

- vllm 配置类默认值(hidden 2816、34 层、index_n_heads 16、max_position 262144)
  是占位默认,**不是** checkpoint 值;页面/图示不得引用。
- indexer 的 k_norm 是 **LayerNorm**(ε=1e−6),主干的 q_a/kv_a norm 是 **RMSNorm**
  (ε=1e−5);不得写成同一种。
- attention scale 是 1/√256(qk_head_dim),indexer 是 1/√128;swiglu clamp 只在
  routed experts;dense/shared 不截断。
- checkpoint 中 q_a_proj 与 kv_a_proj_with_mqa 是两个张量(运行时融合);
  experts 是融合 all-experts 格式(非 per-expert 分文件)。
