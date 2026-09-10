Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (HYV4ForCausalLM)
Source-Revisions: vllm@b2f685834a6456197e7033966fdef52a23f1abcd; vllm-ascend@ad86348b0
Content-Status: verified

# hy4-preview 源码实现注记(vLLM NVIDIA 路径)

源码根:`vllm/models/hy_v4/`(commit `b2f6858`);文件行号以该 commit 为准。

## 调用链

```
HYV4ForCausalLM.forward
└─ HYV4Model.forward(model.py L356–393)
   ├─ embed_tokens → hidden [T,6144](residual=None)
   ├─ 78 × HYV4DecoderLayer.forward(L152–209)
   │  └─ enable_ihc → _forward_ihc:
   │     hc_attn_layer.prepare_input → pre → input_layernorm → self_attn
   │     → hc_attn_layer.post → hc_mlp_layer.prepare_input → pre
   │     → post_attention_layernorm → mlp → hc_mlp_layer.post
   │     返回 (hidden_states, None):residual 携带在 4 通道 hidden 内
   ├─ hc_head(hidden)(L389):4 通道合并
   └─ model.norm(RMSNorm ε=1e−5)
compute_logits:lm_head 以 FP32 计算(head_dtype),enable_lm_head_fp32=true
```

## attention 融合与边界(attention.py)

- `fused_qkv_a_proj = MergedColumnParallelLinear(6144, [2048, 576])`(L364–371):
  checkpoint 的 `q_a_proj` 与 `kv_a_proj_with_mqa` 在 load 时经 stacked mapping
  (model.py L623–624)合并为一次 GEMM,forward 后按 `[2048 | 576]` 切分。
- 两条 RoPE 均为 interleaved(`get_rope(..., is_neox_style=False)`,L416–421 /
  L430–435)。注释明确:checkpoint 的 indexer q_pe/k_pe 是 Megatron/PTM 布局,
  用 NeoX 会破坏 top-k 选择。
- `_indexer_and_attn`(L736–765)是 `eager_break_during_capture` 的单一 eager 段:
  full 层先跑 `self.indexer(...)`(把 top-k 写入共享 `topk_indices_buffer`),
  随后 `self.mla_attn(...)` 读取同一 buffer;shared 层(`skip_topk=True`)跳过
  indexer 直接读最近前驱 full 层写入的索引。
- 稀疏层不做 dense 回退:L337–353 直接探测 sparse MLA backend,失败即抛错。
- sink:`learnable_sink_param [num_local_heads] FP32`(L482–498)作为 `sinks`
  kwarg 传给 `MLAAttention`;`_resolve_sink_backend` 绑定 `HYV4FlashMLASparseImpl`
  (flashmla_sparse.py),内核在 softmax 分母中折叠
  `out *= exp(lse)/(exp(lse)+exp(sink))`;同时 `_force_sparse_mqa`(L635–656)
  强制 prefill 也走稀疏路径,避免"decode 有 sink、prefill 没有"的架构不一致。
- gated MLA(elementwise,L710–731):`attn_out = attn_out * sigmoid(linear_gate(hidden))`;
  elementwise 模式 gate 每 head 256 通道,与 `v_head_dim` 对齐。可选 HPC 单内核
  融合(`hpc_gated_mla_gemm`)。

## indexer(attention.py L125–263)

- `wq_b = ReplicatedLinear(2048, 32×128)`,从 q̃(q_lora RMSNorm 输出)投影;
  `wk_weights_proj = MergedColumnParallelLinear(6144, [128, 32], disable_tp)`,
  一次 GEMM 同时产生 indexer key(128)与 per-head 权重(32);FP8 wk 权重在
  load 时上转 BF16 以维持融合。
- `k_norm = LayerNorm(128, eps=1e−6)` —— 主干的两个 norm 是 RMSNorm,indexer 的
  key norm 是 LayerNorm(带均值减除),两者不可混淆。
- q 经 `per_token_group_quant_fp8`(group 128,ue8m0)量化;权重按
  `w · q_scale · softmax_scale · N_idx^{-1/2}` 折叠;`softmax_scale = 128^{-1/2}`。
- FP8 k-cache:`DeepseekV32IndexerCache`,cache 行宽
  `head_dim + head_dim/128×4`(128 值 + 每 128 元素 4 字节 scale)。
- `compute_skip_topk_layers`(L54)由 `indexer_types` 推导 shared 层集合:
  57 层跳过 indexer 计算;MTP 层(layer_id ≥ 78)始终建 full indexer(L324–331)。

## MoE(moe.py)

- router:`GateLinear(6144, 256, fp32)`;`expert_bias [256] fp32` 来自
  checkpoint `gate.e_score_correction_bias`(load 时改名,model.py L584–587)。
- `FusedMoEFactory(scoring_func="sigmoid", num_expert_group=1, topk_group=1,
  renormalize=norm_topk_prob, routed_scaling_factor=2.827,
  e_score_correction_bias=expert_bias, shared_experts=…, swiglu_limit=10.0)`
  (moe.py L152–170)。swiglu clamp 只作用于 routed experts:
  gate `clamp(max=10)`、up `clamp(±10)`、`silu(g)·u`;dense FFN 与 shared expert
  不截断(`HYV4FeedForward`,moe.py L23–66)。
- checkpoint experts 为融合格式:`experts.gate_up_proj [E,4096,6144]` 按 dim −2
  切成 w1/w3;`experts.down_proj [E,6144,2048]`(model.py L508–526)。

## iHC(hc.py)

- `HYV4HCPreLayer`(L27–139):flatten → fp32 → `rsqrt(mean+1e−5)`(注意这里是
  `layernorm_epsilon=1e−5`)→ `hc_fn`(ReplicatedLinear 24576→8,fp32)→
  pre/post 门(pre = σ(·scale₀+base₀)+1e−6;post = 2·σ(·scale₁+base₁)+1e−6)→
  通道加权和。HPC 单内核替换(`HpcIHCPre/Post/Head`)仅当安装 hpc 包且
  `VLLM_ENABLE_HPC_OPS=1` 时启用,数学不变。
- `HYV4HCPostLayer`(L142–186):fp32 下 `post·x + residual`,无参数。
- `HYV4HCHeadLayer`(L189–277):同 pre 的 RMS→投影→σ 门,把 4 通道合回 1 通道。
- `hc_base[:4]` 初始化 −log(hc_mult−1) = −log 3;`hc_scale` 初始 0.01。

## MTP(mtp.py)

- `HYV4MultiTokenPredictorLayer`(L324–390):`enorm/hnorm = RMSNorm(6144)`,
  `eh_proj = Linear(12288→6144)`,block 为 `HYV4DecoderLayer` 且
  `mtp_config.enable_ihc = False`(L351,checkpoint 无 MTP iHC 权重),full indexer;
  输出经 `final_layernorm`。embed/lm_head 与主干共享。
- `HYV4ForCausalLM.load_weights` 以 `drop_prefixes {"model.mtp_layers."}` 丢弃
  draft 权重(model.py L704–708);draft 由 `HYV4MTP` 单独加载。

## 后端与部署差异

- NVIDIA:vllm `hy_v4/nvidia/` 全部路径;FP8 indexer cache + FlashMLA sparse。
- Ascend:vllm-ascend `docs/source/tutorials/models/Hy4-preview.md` 提供 Atlas 800I A3
  的 W8A8 部署(约 762GB,专用 docker 镜像 `quay.io/ascend/vllm-ascend:hy4-a3`),
  对应 vllm 主仓 commit `b2f6858`;ascend 侧模型代码尚未合入其仓库。
- sink/FP8 行为依赖后端能力;不支持 sink 的后端会禁用 sink 并告警(见 architecture.md)。
