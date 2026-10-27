---
Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m3.1-flash
Source-Revisions: hf-config.json@main(2026-10-26); vllm@2f1cb151fd9b036422cc27f0ad82f5f66edec879
Content-Status: verified
---

# 公式

符号定义见 [symbols.md](symbols.md)；公式 id 与 evidence.md / 页面详情一致。
数学解释与 vLLM 实际执行的区别：融合 kernel（QK norm + RoPE + cache insert、
FusedMoE 内部 routing）在图中按数学步骤拆解，不代表存在独立运行时 OP。

## F-NORM · Gemma RMSNorm

```
RMS(x) = sqrt( (1/H) Σⱼ xⱼ² + ε )
yᵢ = xᵢ / RMS(x) · (1 + γᵢ)
```
`H=6144`，`ε=1e−6`；per-head 变体归一化轴为 `Dₕ=128`（Q/K norm）或 `D_idx=128`（index Q/K norm）。
证据：nvidia/model.py L118–146（FlashInfer `gemma_rmsnorm`）。

## F-PROJ5 · 五路融合投影（全部 60 层）

```
Z = X̂ · [Wqᵀ | Wkᵀ | Wvᵀ | Wqidxᵀ | Wkidxᵀ]        Z ∈ [Nq, 9856]
split(Z) → Q[8192] · K[512] · V[512] · Qidx[512] · Kidx[128]   （末维通道数）
```
`8192=64·128`，`512=4·128`，`128=1·128`；TP 下 Q/K/V/Qidx 按 head 切分，Kidx 单头。
证据：nvidia/model.py L444–455（`MinimaxM3QKVParallelLinearWithIndexer`）。

## F-QKNORM · per-head Q/K Gemma RMSNorm

```
Q̃ₕ,ᵢ = Qₕ,ᵢ / sqrt((1/Dₕ)Σⱼ Qₕ,ⱼ² + ε) · (1 + γQ,ᵢ)      （K、index Q/K 同理）
```

## F-ROPE · Partial RoPE

```
(Qrot, Qpass) = split(Q̃; Dᵣ, Dₕ−Dᵣ) = split(Q̃; 64, 64)
Qʳ = concat(RoPE(Qrot, p; θ=1e7), Qpass)
```
`partial_rotary_factor=0.5`，旋转角基底 `θ_base=1e7`（**M3 为 5e6**）；
index 分支与主分支共用同一 `rotary_emb`（nvidia/model.py L473–488）。

## F-IDXSCORE · Indexer token 打分与块归约

```
Sʳᵢⱼ = ⟨ Qidxʳᵢ, Kidxⱼ ⟩                       （缩放由 kernel 内部应用，见下方说明）
S̃ʳᵢⱼ = Sʳᵢⱼ  (j ≤ pᵢ) ;  −∞ (j > pᵢ)           （未来 key 屏蔽）
Bʳᵢᵤ = max_{j ∈ [128u, 128(u+1))} S̃ʳᵢⱼ          （block max，B_block=128）
```
说明：vLLM indexer impl 内部按 `scale = head_dim^-0.5` 配置构造（common/indexer.py
MiniMaxM3Indexer L578+ 的 `scale` 参数来自 SparseAttention 的 `self.scaling`）。
公式写点积形式；kernel 内部应用 scale，但 Top-K 块选择对正缩放不变，
故数学语义等价于"点积相似度 + 块内 max"，缩放不改变选择结果。

## F-TOPK · Top-16 块选择

```
Bᵤ ← +∞  (u ∈ local(i))        local_blocks=1：当前 query 所在块强制入选
𝒮ʳᵢ = TopK₁₆(Bʳᵢ)              K_block=16；无效槽记 −1
```
每 query 可见 keys ≤ 16×128 = 2048。证据：common/indexer.py index_topk；
参数 init_blocks=0（无 sink 块）。

## F-ATTN · 候选块内精确 causal GQA（数学分解，非独立 OP）

```
Aₕᵢⱼ = Qʳₕᵢ · Kʳ⌊h/16⌋ⱼ / √Dₕ        j ∈ 𝒮⌊h/16⌋ᵢ 展开后的 token 集合
Ã = A + Mcausal + Mpad                （块选中 ≠ 块内全可见）
P = softmax(Ã, dim=-1)
Oₕᵢ = Σⱼ Pₕᵢⱼ · V⌊h/16⌋ⱼ
Yattn = RowParallel( concat_h(Oₕ) · Woᵀ )    Wo: [6144, 8192]
```
实际执行为融合 sparse-attention kernel（Triton 或 SM100 MSA/CUTLASS decode），
不物化 A/P 中间张量；K/V 由 kernel 按 `block_indices + block_table` 直接寻址。

## F-MLP · Dense SwiGLU-OAI MLP（L0–L2）

```
G = Û Wgateᵀ,  U = Û Wupᵀ              （MergedColumnParallel: [B,S,2·12288] packed）
Z = min(G,7) ⊙ σ(1.702·min(G,7)) ⊙ (clip(U,−7,7) + 1.0)
Yffn = Z Wdownᵀ
```
`α=1.702`、`β=1.0`（M3.1-Flash config 缺省，vLLM `MiniMaxM3TextConfig` 默认 1.0）、`c=7.0`。
证据：nvidia/model.py L149–192 + activation.py `SiluAndMulWithClamp`。

## F-MOE · Sigmoid Top-4 MoE + shared（L3–L59）

```
r = Û Wrouterᵀ  ∈ [Nq,128]  (FP32)
s = σ(r)
𝓔 = TopK₄(s + b)                        b = e_score_correction_bias（只用于选择）
ŵₑ = 2.0 · sₑ / Σ_{j∈𝓔} sⱼ              混合权重不含 b；routed_scaling_factor=2.0
Yrouted = Σ_{e∈𝓔} ŵₑ · Eₑ(Û)            Eₑ: SwiGLU-OAI(w1,w3) → w2，宽 3072
Ymoe = Yrouted + Eshared(Û)             shared 宽 3072，恒激活
```
证据：nvidia/model.py L195–289（MiniMaxM3MoE.forward、FusedMoEFactory 参数）。

## F-SLOT · KV slot 解析（runtime）

```
slot(r,p) = block_table[r, ⌊p/128⌋] · 128 + (p mod 128)
```
主 KV 与 index K 各有 slot_mapping；paged kernel 依 block_table 把逻辑块译成物理页。

## 量化表示（checkpoint ↔ 运行时）

- 线性层（除 router/embed/LM head/vision/projector）：MXFP8，权重按 [1,32] 块存 scale，
  activation 运行时动态量化；checkpoint 键 `weight_scale_inv` → 加载时改名 `weight_scale`。
- MoE 专家：NVFP4 packed（`nvfp4-pack-quantized`，group_size 16）。
- KV cache：NVFP4（E2M1 + 块 scale），index K cache 默认 bf16，可配 fp8_e4m3 / nvfp4。