Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (HYV4ForCausalLM)
Source-Revisions: vllm@b2f685834a; HF config.json @ 2026-09-09
Content-Status: verified

# hy4-preview 结构

## 输入与输出

- 输入:`input_ids [B,S]`(+ `positions [Nq]`、attention metadata 由 runtime 提供)。
- 输出:final RMSNorm 后的 hidden → `lm_head`(FP32 计算,`enable_lm_head_fp32=true`)
  得到 `[B,S,V]`,V=120832。

## 顶层拓扑

```
embed_tokens [V,6144]
→ 78 × HYV4DecoderLayer(L0 dense FFN,L1–L77 MoE;全部 iHC + DSA)
→ hc_head(4 通道合并)
→ model.norm(RMSNorm ε=1e−5)
→ lm_head(FP32 logits)
侧路:MTP ×1(enorm/hnorm → eh_proj → 无 iHC 的 decoder block → final_layernorm,共享 embed/lm_head)
```

## iHC:4 通道残差(hc_mult=4)

每个 decoder layer 的 hidden 以 `[num_tokens, 4, 6144]` 携带。每个子块(attn、MLP)各有一个
`HYV4HCLayer` 边界:

1. `prepare_input`:把 `[T,6144]`(层首)或 `[T,4,6144]`(子块间)规范化为 3D;
   层首将单通道 broadcast 到 4 通道,之后整个网络不再有单一 residual。
2. `hc_pre`(`HYV4HCPreLayer`,有参数):对展平的 `[T,24576]` 做 RMS 式归一化
   (ε=1e−5,`rms_norm_eps`),经 `hc_fn`(24576→8,FP32)+ `hc_scale[2]`/`hc_base[8]`
   产生 pre/post 各 4 个 sigmoid 门;`y = Σ_i pre_i·x_i` 把 4 通道减成单通道交给子块,
   同时输出 post 门。
3. 子块输出经 `hc_post`(`HYV4HCPostLayer`,无参数)散回通道:
   `y[n,i,d] = post[n,i]·x[n,d] + residual[n,i,d]`。
4. 78 层之后 `hc_head`(有参数)按同样"RMS→hc_head_fn(24576→4)→σ 门加权求和"
   把 4 通道合并回 `[T,6144]`,再进 final norm。

门偏置初始化 `hc_base[:4] = −log(3)`(hc_mult−1 的对数),即训练起点近似均匀门。
checkpoint 只保存 pre 权重;post 无参数(权重索引证实)。

## 层族(78 层)

| 族 | 层号 | FFN | indexer |
| --- | --- | --- | --- |
| dense | L0 | SwiGLU FFN,intermediate 18432 | full |
| MoE·full | L1, L5, L9, …, L77(共 20 层) | MoE | full(自己算 top-2048) |
| MoE·shared | 其余 57 层 | MoE | shared(复用最近前驱 full 层的 top-k 索引) |

`indexer_types` 共 78 项,full 恰好在 `{0,1,5,9,…,77}` 共 21 层(含 L0);权重索引显示这 21 层才有
`indexer.*` 权重。所有 78 层的 attention 类型均为 `deepseek_sparse_attention`;
稀疏路径不做 dense 回退(vllm 直接 fail fast)。

## Attention 子块(全部层同构,DSA = MLA + lightning indexer)

```
hidden [T,6144]
├─ fused_qkv_a_proj(6144 → [2048 | 576])   ← checkpoint 的 q_a_proj + kv_a_proj_with_mqa 运行时融合
│   ├─ q_a(2048) → RMSNorm → q_b_proj → 64 heads ×(192 nope + 64 rope)
│   │                └→ indexer.wq_b(2048 → 32×128)          [仅 full 层]
│   └─ (kv_lora 512 | k_pe 64) → kv_a RMSNorm(512)
├─ kv_b_proj(512 → 64 ×(192 nope K + 256 V))
├─ RoPE(interleaved,θ=1e7):q 的 rope 64 维 + k_pe
├─ [full 层] lightning indexer:FP8 q(ue8m0, group 128)、FP8 k cache、
│   k_norm 为 LayerNorm(128, ε=1e−6)、top-2048 token 索引写入共享 buffer
├─ [shared 层] 直接读 buffer 中最近前驱 full 层的索引
├─ FlashMLA sparse attention:候选 2048 token 上精确 softmax,scale 1/√256,
│   per-head learnable sink:`out *= exp(lse)/(exp(lse)+exp(sink))`
├─ gated MLA(elementwise):`attn_out ⊙ σ(linear_gate(hidden))`,gate 64×256
└─ o_proj(16384 → 6144)
```

KV cache:MLA 压缩 cache(`kv_lora 512 + rope 64`/token);indexer 另有独立 FP8
key-only side cache(`DeepseekV32IndexerCache`);top-k 索引写入模型级共享
`topk_indices_buffer [max_num_batched_tokens, 2048] int32`。

## MoE 子块(L1–L77)

```
Û [T,6144]
├─ gate(6144 → 256,FP32)→ sigmoid 分数 + expert_bias → 组内 Top-8(group=1)
│   → renormalize(norm_topk_prob)→ × routed_scaling_factor 2.827
├─ 256 routed experts:clamped SwiGLU(moe_intermediate 2048;gate 截 max=10,up 截 ±10)
├─ 1 shared expert:普通 SwiGLU FFN(2048,不截断,所有 token 必经)
└─ routed(已加权归并)+ shared 相加
```

## 已知限制

- 页面描述 vllm 的执行图;`o_proj` 前的 gate 投影在 elementwise 模式下输出宽度为
  `64×256`(每 head 256 通道),与 v_head_dim 对齐后逐元素相乘。
- sink 依赖 sink-capable sparse backend;若平台后端不支持,sink 会被禁用并告警
  (权重仍加载)——这是部署差异而非架构差异。
