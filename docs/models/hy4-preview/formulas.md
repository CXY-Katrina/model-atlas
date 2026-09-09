Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: hy4-preview (HYV4ForCausalLM)
Source-Revisions: vllm@b2f685834a; HF config.json @ 2026-09-09
Content-Status: verified

# hy4-preview 公式

符号见 [symbols.md](./symbols.md);公式 id 与页面节点一一对应,证据见 [evidence.md](./evidence.md)。

## F-IHC · iHC 边界

### F-IHC-PRE(hc_pre,attn/MLP 子块各一)

```
x_flat = flatten(x) ∈ ℝ^{T×24576},在 FP32 计算
rsqrt = 1/√(mean(x_flat²) + ε_rms)          (ε_rms = 1e−5)
mixes = hc_fn(x_flat) · rsqrt ∈ ℝ^{T×8}
pre_i  = σ(mixes_i · hc_scale[0] + hc_base_i) + ε_hc       i = 0..3
post_i = 2·σ(mixes_{4+i} · hc_scale[1] + hc_base_{4+i}) + ε_hc
y = Σ_i pre_i · x_i ∈ ℝ^{T×6144}            (4 通道减为单通道)
```

- 定义域:`x ∈ [T,4,6144]`;输出 `y ∈ [T,6144]` 与 post 门 `[T,4]`。
- `ε_hc = 1e−6`;`hc_scale` 初始化 0.01;`hc_base[:4]` 初始化 −log 3,
  `hc_base[4:8]` 初始化 0。
- hc_magnitude = 2 乘在 post 门上(等价于把子块输出放大后写入残差通道)。

### F-IHC-POST(hc_post,无参数)

```
y[n,i,d] = post[n,i] · x[n,d] + residual[n,i,d]     (FP32 计算后转回原 dtype)
```

### F-IHC-HEAD(hc_head,模型出口)

```
x_flat = flatten(x) ∈ ℝ^{T×24576}(FP32)
rsqrt = 1/√(mean(x_flat²) + ε_rms)
mixes = hc_head_fn(x_flat) · rsqrt ∈ ℝ^{T×4}
pre_i = σ(mixes_i · hc_head_scale + hc_head_base_i) + ε_hc
y = Σ_i pre_i · x_i ∈ ℝ^{T×6144}
```

## F-ATTN · MLA 主干

### F-A-PROJ(融合 down 投影)

```
[q_a | kv_a | k_pe] = hidden · fused_qkv_a_projᵀ     输出切分 [2048 | 512 | 64]
q̃ = RMSNorm(q_a; ε=1e−5)
Q = q̃ · q_b_projᵀ ∈ ℝ^{T×64×256}(每 head 192 nope + 64 rope)
k̃ = RMSNorm(kv_a; ε=1e−5)   (kv_lora 压缩向量)
```

### F-A-UP(kv_b 投影)

```
K_nope = W_{kv_b,nope} · k̃ ∈ ℝ^{T×64×192}
V      = W_{kv_b,v}    · k̃ ∈ ℝ^{T×64×256}
```

### F-A-ROPE(interleaved RoPE,θ=1e7)

```
RoPE(q[...,192:256], k_pe; positions)   仅旋转 rope 段
```
checkpoint 的 q_pe/k_pe 为 interleaved(Megatron/PTM)布局,必须用
`is_neox_style=False`;NeoX 会破坏 indexer 的相对位置依赖。

### F-A-SCORE + F-A-SINK(稀疏注意力 + sink)

```
S_h = Q_h · [K_h,nope ‖ k_pe]ᵀ / √256          仅在 top-2048 候选 token 上
P   = softmax(S + mask_causal)
O_h = P · V_h
sink_h 为 per-head FP32 参数(checkpoint 初始 0)
out_h = O_h · exp(lse) / (exp(lse) + exp(sink_h))
```
注意 scale 是 qk_head_dim=256(nope192+rope64),不是 v_head_dim。

### F-A-GATE(gated MLA,elementwise)

```
g = hidden · W_gateᵀ ∈ ℝ^{T×64×256}
Y_attn = O ⊙ σ(g)                   (每 head 的 256 通道逐元素)
Y = Y_attn · o_projᵀ ∈ ℝ^{T×6144}
```

## F-IDX · lightning indexer(仅 21 个 full 层执行)

```
Qidx = q̃·wq_b → 32 heads × 128(其中 rope 64 维同样 interleaved 旋转;q̃ 已是 q_a RMSNorm 输出,不再二次归一化)
[k | w] = hidden · wk_weights_projᵀ               融合 GEMM,输出 [128 | 32]
k = LayerNorm(k; ε=1e−6)                          注意是 LayerNorm 不是 RMSNorm
s_{t,j} = Σ_h w_{t,h} · ⟨Qidx_{t,h}, k_j⟩ / √128 / √32
w'_{t,h} = w_{t,h} · q_scale_{t,h}                (FP8 per-token-group 128, ue8m0)
I_t = TopK_2048(s_t)                              causal 内;写入共享 buffer
```
FP8:q 在投影后做 per-token-group 量化(group 128,ue8m0 scale);k 以 FP8 存入独立
indexer k-cache(插 cache 时融合量化)。shared 层不执行本节任何计算。

## F-MOE · MoE(L1–L77)

```
r = Û · W_gateᵀ ∈ ℝ^{T×256}(FP32)
s = σ(r);选择 = TopK_8(s + expert_bias)     (n_group=1, topk_group=1)
ŵ_e = 2.827 · s_e / Σ_{j∈Top8} s_j          (bias 只影响选择,不进权重)
E_e(Û) = W_down,e [ clamp(g,10)⊙σ(clamp(g,10)) ⊙ clamp(u,−10,10) ]   g=gate,u=up
Y_moe = Σ_{e∈Top8} ŵ_e·E_e(Û) + E_shared(Û)     shared 不截断、不路由
```

## F-FFN · dense FFN(L0)

```
Y_ffn = W_down [ SwiGLU(W_gate·Û, W_up·Û) ]     intermediate 18432,无 clamp
```

## F-MTP · MTP draft 层(×1)

```
h_e = enorm(embed_t+1);h_p = hnorm(hidden_t)
h = eh_proj([h_e ‖ h_p]) ∈ ℝ^{T×6144}
h = MTPBlock(h)(无 iHC 的 decoder block;full indexer;MoE)
h = final_layernorm(h) → 与主干共享 lm_head
```
checkpoint 中 `model.mtp_layers.0.*` 存在但没有 hc 权重;`HYV4ForCausalLM.load_weights`
将其丢弃,由 `HYV4MTP` 推理类加载。
