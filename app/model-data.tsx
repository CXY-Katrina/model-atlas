"use client";

import { useState } from "react";

type Scope = "language" | "vision";
type Tab = "io" | "formula" | "code" | "weights";
type Tone = "norm" | "projection" | "attention" | "index" | "moe" | "vision" | "output";

export type Weight = {
  key: string;
  shape: string;
  dtype: "BF16" | "F32";
  shard: string;
  runtime?: string;
  note?: string;
  params?: string;
};

export type Node = {
  id: string;
  tone: Tone;
  kicker: string;
  title: string;
  summary: string;
  input: string;
  inputShape: string;
  output: string;
  outputShape: string;
  formula: string;
  formulaNote: string;
  runtime: string;
  source: string;
  sourceUrl: string;
  code: string;
  weights: Weight[];
};

export const MODEL = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/nvidia/model.py";
export const VISION = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/common/vision_tower.py";
const INDEXER = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/common/indexer.py";
const SPARSE = "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/common/ops/sparse_attn.py";

export function layerShard(layer: number) {
  if (layer === 0) return "model-00001-of-00059.safetensors";
  if (layer < 3) return "model-00002-of-00059.safetensors";
  return `model-${String(layer).padStart(5, "0")}-of-00059.safetensors`;
}

function key(layer: number, suffix: string) {
  return `language_model.model.layers.${layer}.${suffix}`;
}

export function denseNodes(layer: number): Node[] {
  const shard = layerShard(layer);
  return [
    {
      id: "dense-norm", tone: "norm", kicker: `L${layer} · PRE-NORM`, title: "Gemma RMSNorm",
      summary: "先保留 residual，再用 1+γ 的 Gemma 风格 RMSNorm 规范化本层输入。",
      input: "Xₗ", inputShape: "[B,S,6144]", output: "X̂ₗ", outputShape: "[B,S,6144]",
      formula: "X̂ = X / √(mean(X²)+ε) ⊙ (1+γ)", formulaNote: "ε=1e-6；vLLM 可把上一层 all-reduce 与这里融合。",
      runtime: "MiniMAXGemmaRMSNorm · input_layernorm", source: "nvidia/model.py · MiniMaxM3DecoderLayer.forward", sourceUrl: MODEL,
      code: `residual = hidden_states\nhidden_states = self.input_layernorm(hidden_states)\nhidden_states = self.self_attn(positions, hidden_states)`,
      weights: [{ key: key(layer,"input_layernorm.weight"), shape:"[6144]", dtype:"BF16", shard, params:"6,144" }],
    },
    {
      id: "dense-qkv", tone: "projection", kicker: "64Q : 4KV · ONE GEMM", title: "Q / K / V 投影",
      summary: "checkpoint 保存三块矩阵；vLLM loader 将它们拼成 QKVParallelLinear。",
      input: "X̂", inputShape: "[B,S,6144]", output: "Q | K | V", outputShape: "[B,S,8192 | 512 | 512]",
      formula: "Q=X̂Wqᵀ, K=X̂Wkᵀ, V=X̂Wvᵀ", formulaNote: "head_dim=128；每 16 个 Query heads 共享一组 K/V。",
      runtime: "QKVParallelLinear · qkv_proj", source: "nvidia/model.py · MiniMaxM3Attention", sourceUrl: MODEL,
      code: `self.qkv_proj = QKVParallelLinear(6144, 128, 64, 4)\nqkv, _ = self.qkv_proj(hidden_states)`,
      weights: [
        { key:key(layer,"self_attn.q_proj.weight"), shape:"[8192,6144]", dtype:"BF16", shard, runtime:"qkv_proj · q shard", params:"50.33M" },
        { key:key(layer,"self_attn.k_proj.weight"), shape:"[512,6144]", dtype:"BF16", shard, runtime:"qkv_proj · k shard", params:"3.15M" },
        { key:key(layer,"self_attn.v_proj.weight"), shape:"[512,6144]", dtype:"BF16", shard, runtime:"qkv_proj · v shard", params:"3.15M" },
      ],
    },
    {
      id: "dense-attn", tone: "attention", kicker: "QK NORM · 50% ROPE · FULL GQA", title: "完整历史因果注意力",
      summary: "Q/K 按 head 归一化，仅前 64/128 维应用 RoPE；前三层读取全部可见 KV。",
      input: "Q,K,V + positions", inputShape: "Q [B,64,S,128] · K/V [B,4,T,128]", output: "attention heads", outputShape: "[B,S,8192]",
      formula: "Oₕ=softmax(Q̃ₕK̃⌊h/16⌋ᵀ/√128+Mcausal)V⌊h/16⌋", formulaNote: "S=本轮 query 长度；T=包含历史 cache 的 KV 总长度。",
      runtime: "fused_qknorm_rope_kv_insert → Attention", source: "nvidia/model.py · MiniMaxM3Attention.forward", sourceUrl: MODEL,
      code: `ops.fused_minimax_m3_qknorm_rope_kv_insert(qkv, ...)\nq, k, v = qkv.split([8192, 512, 512], dim=-1)\nattn_output = self.attn(q, k, v)`,
      weights: [
        { key:key(layer,"self_attn.q_norm.weight"), shape:"[128]", dtype:"BF16", shard, params:"128" },
        { key:key(layer,"self_attn.k_norm.weight"), shape:"[128]", dtype:"BF16", shard, params:"128" },
      ],
    },
    {
      id: "dense-o", tone: "projection", kicker: "HEADS → HIDDEN", title: "O Projection + residual",
      summary: "64 个 head 拼成 8192 通道，再投回模型隐藏宽度 6144。",
      input: "concat(O₀…O₆₃)", inputShape: "[B,S,8192]", output: "Yattn", outputShape: "[B,S,6144]",
      formula: "Yattn=concat(O)Woᵀ+residual", formulaNote: "后接 post_attention_layernorm，再进入 FFN。",
      runtime: "RowParallelLinear · o_proj", source: "nvidia/model.py · MiniMaxM3Attention", sourceUrl: MODEL,
      code: `output, _ = self.o_proj(attn_output)\nhidden_states, residual = post_attention_layernorm(output, residual)`,
      weights: [{ key:key(layer,"self_attn.o_proj.weight"), shape:"[6144,8192]", dtype:"BF16", shard, params:"50.33M" }],
    },
    {
      id: "dense-mlp", tone: "moe", kicker: "DENSE FFN · WIDTH 12288", title: "SwiGLU-OAI MLP",
      summary: "仅 L0–L2 使用 dense FFN；gate/up 在运行时合并，down 投回 6144。",
      input: "normalized hidden", inputShape: "[B,S,6144]", output: "Xₗ₊₁", outputShape: "[B,S,6144]",
      formula: "Wdown[clamp(g)·σ(1.702·clamp(g))·clamp(u+1)]", formulaNote: "g=Wgate x，u=Wup x；clamp limit=7.0。",
      runtime: "MiniMaxM3MLP · gate_up_proj / down_proj", source: "nvidia/model.py · MiniMaxM3MLP", sourceUrl: MODEL,
      code: `gate_up, _ = self.gate_up_proj(x)\nx = self.act_fn(gate_up)\nx, _ = self.down_proj(x)`,
      weights: [
        { key:key(layer,"mlp.gate_proj.weight"), shape:"[12288,6144]", dtype:"BF16", shard, runtime:"gate_up_proj · gate", params:"75.50M" },
        { key:key(layer,"mlp.up_proj.weight"), shape:"[12288,6144]", dtype:"BF16", shard, runtime:"gate_up_proj · up", params:"75.50M" },
        { key:key(layer,"mlp.down_proj.weight"), shape:"[6144,12288]", dtype:"BF16", shard, params:"75.50M" },
      ],
    },
  ];
}

export function sparseNodes(layer: number): Node[] {
  const shard = layerShard(layer);
  return [
    {
      id:"sparse-proj", tone:"projection", kicker:`L${layer} · FIVE-WAY PROJECTION`, title:"主分支 + Index 分支融合投影",
      summary:"一次 GEMM 同时产生 Q/K/V 与 Qidx/Kidx；前者生成内容，后者只选 KV block。",
      input:"RMSNorm(Xₗ)", inputShape:"[B,S,6144]", output:"Q|K|V|Qidx|Kidx", outputShape:"8192 | 512 | 512 | 512 | 128",
      formula:"[Q,K,V,Qidx,Kidx]=X̂·Wpackedᵀ", formulaNote:"共 9856 输出通道；M3 禁用 index V/O。",
      runtime:"MinimaxM3QKVParallelLinearWithIndexer", source:"nvidia/model.py · MiniMaxM3SparseAttention", sourceUrl:MODEL,
      code:`self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(\n    6144, 128, 64, 4, total_idx_heads=4, idx_head_dim=128\n)\nqkv, _ = self.qkv_proj(hidden_states)`,
      weights:[
        {key:key(layer,"self_attn.q_proj.weight"),shape:"[8192,6144]",dtype:"BF16",shard,runtime:"qkv_proj · q",params:"50.33M"},
        {key:key(layer,"self_attn.k_proj.weight"),shape:"[512,6144]",dtype:"BF16",shard,runtime:"qkv_proj · k",params:"3.15M"},
        {key:key(layer,"self_attn.v_proj.weight"),shape:"[512,6144]",dtype:"BF16",shard,runtime:"qkv_proj · v",params:"3.15M"},
        {key:key(layer,"self_attn.index_q_proj.weight"),shape:"[512,6144]",dtype:"BF16",shard,runtime:"qkv_proj · index q",params:"3.15M"},
        {key:key(layer,"self_attn.index_k_proj.weight"),shape:"[128,6144]",dtype:"BF16",shard,runtime:"qkv_proj · index k",params:"0.79M"},
      ],
    },
    {
      id:"indexer", tone:"index", kicker:"INDEX BRANCH · 4 HEADS", title:"Token score → block max",
      summary:"每个 GQA group 有独立 Qidx，Kidx 跨组共享；先遮住未来 token，再对每 128 token 取 max。",
      input:"Qidx,Kidx cache", inputShape:"[B,S,4,128] · [B,T,1,128]", output:"block scores", outputShape:"[B,4,S,⌈T/128⌉]",
      formula:"Mᵢ,bʳ=maxⱼ∈block(b),j≤i ⟨RMS(Qidxᵢʳ),RMS(Kidxⱼ)⟩/√128", formulaNote:"未来 token 先置 −∞，所以尾块内部也不会偷看未来。",
      runtime:"MiniMaxM3Indexer · key-only paged cache", source:"common/indexer.py · MiniMaxM3Indexer", sourceUrl:INDEXER,
      code:`self.indexer = MiniMaxM3Indexer(\n    topk_blocks=16, sparse_block_size=128,\n    num_index_heads=4, index_head_dim=128,\n    init_blocks=0, local_blocks=1, score_type="max"\n)`,
      weights:[
        {key:key(layer,"self_attn.index_q_norm.weight"),shape:"[128]",dtype:"BF16",shard,params:"128"},
        {key:key(layer,"self_attn.index_k_norm.weight"),shape:"[128]",dtype:"BF16",shard,params:"128"},
      ],
    },
    {
      id:"topk", tone:"index", kicker:"LOGICAL BLOCK ROUTING", title:"Priority-aware Top-16",
      summary:"每个 index head 选 16 个逻辑块；local block 强制保留，首块没有 sink 优先级。",
      input:"block scores", inputShape:"[B,4,S,Nblocks]", output:"logical block ids", outputShape:"[B,S,4,16]",
      formula:"TopK₁₆(M+10²⁹·Ilocal+10³⁰·Iinit), init_blocks=0", formulaNote:"可见块 ≤16 时会全选；block_table 再把逻辑 id 翻译为物理 page。",
      runtime:"index_topk → topk_indices_buffer", source:"common/indexer.py · index_topk", sourceUrl:INDEXER,
      code:`self.indexer(index_query)\n# writes [token, index_head, topk] logical block ids\n# into the shared topk_indices_buffer`, weights:[],
    },
    {
      id:"sparse-attn", tone:"attention", kicker:"MAIN BRANCH · EXACT ATTENTION", title:"Selected pages 上的精确 GQA",
      summary:"主 Q/K 做 per-head norm 与 50% RoPE；attention 只读取 Top-16 pages，但块内仍做精确 token softmax。",
      input:"Q + selected K/V pages", inputShape:"Q [B,64,S,128] · ≤16×128 KV/group", output:"attention heads", outputShape:"[B,S,8192]",
      formula:"Oₕ=softmax(QₕK𝒮ᵀ/√128+Mcausal+Mpad)V𝒮", formulaNote:"稀疏的是访问块集合，不是块内数学。page size 与 sparse block size 都是 128。",
      runtime:"MiniMaxM3SparseImpl.forward", source:"common/ops/sparse_attn.py", sourceUrl:SPARSE,
      code:`self.indexer(index_query)\nattn_output = self.impl.forward(\n    self, query, self.kv_cache, output\n)\noutput, _ = self.o_proj(attn_output)`,
      weights:[
        {key:key(layer,"self_attn.q_norm.weight"),shape:"[128]",dtype:"BF16",shard,params:"128"},
        {key:key(layer,"self_attn.k_norm.weight"),shape:"[128]",dtype:"BF16",shard,params:"128"},
        {key:key(layer,"self_attn.o_proj.weight"),shape:"[6144,8192]",dtype:"BF16",shard,params:"50.33M"},
      ],
    },
    {
      id:"router", tone:"moe", kicker:"FP32 ROUTER · 128 → TOP-4", title:"Sigmoid + correction bias",
      summary:"每 token 计算 128 个 FP32 logits；校正 bias 决定入选专家，原始 sigmoid 分数用于混合权重。",
      input:"post-attn hidden", inputShape:"[B,S,6144]", output:"expert ids + weights", outputShape:"2 × [B,S,4]",
      formula:"𝓔=TopK₄(σ(XWrouterᵀ)+b), ŵₑ=σ(rₑ)/Σⱼ∈𝓔σ(rⱼ)", formulaNote:"routed_scaling_factor=2.0；router 与 bias 都是 F32。",
      runtime:"GateLinear + FusedMoE routing", source:"nvidia/model.py · MiniMaxM3MoE", sourceUrl:MODEL,
      code:`router_logits, _ = self.gate(hidden_states)\nfinal_hidden_states = self.experts(\n    hidden_states=hidden_states, router_logits=router_logits\n)`,
      weights:[
        {key:key(layer,"block_sparse_moe.gate.weight"),shape:"[128,6144]",dtype:"F32",shard,params:"0.79M"},
        {key:key(layer,"block_sparse_moe.e_score_correction_bias"),shape:"[128]",dtype:"F32",shard,params:"128"},
      ],
    },
    {
      id:"experts", tone:"moe", kicker:"ROUTED LANE · 128 EXPERTS", title:"每 token 执行 4 个 routed experts",
      summary:"每专家三块矩阵；以下是 expert 0 的真实键，结构同构重复 128 次。vLLM 把 w1/w3 打包为 w13。",
      input:"4 token groups", inputShape:"4 × [tokensₑ,6144]", output:"4 expert outputs", outputShape:"4 × [tokensₑ,6144]",
      formula:"Eₑ(x)=W2ₑ[SwiGLU-OAI(W1ₑx,W3ₑx)]", formulaNote:"单专家 56.62M；128 个 routed experts 合计约 7.248B 参数/层。",
      runtime:"FusedMoE · experts.w13 / experts.w2", source:"nvidia/model.py · FusedMoEFactory", sourceUrl:MODEL,
      code:`FusedMoEFactory(\n  num_experts=128, top_k=4, hidden_size=6144,\n  intermediate_size=3072, activation="swigluoai_uninterleave"\n)`,
      weights:[
        {key:key(layer,"block_sparse_moe.experts.0.w1.weight"),shape:"[3072,6144]",dtype:"BF16",shard,runtime:"experts.w13 · gate",note:"×128",params:"18.87M"},
        {key:key(layer,"block_sparse_moe.experts.0.w3.weight"),shape:"[3072,6144]",dtype:"BF16",shard,runtime:"experts.w13 · up",note:"×128",params:"18.87M"},
        {key:key(layer,"block_sparse_moe.experts.0.w2.weight"),shape:"[6144,3072]",dtype:"BF16",shard,runtime:"experts.w2",note:"×128",params:"18.87M"},
      ],
    },
    {
      id:"shared", tone:"moe", kicker:"SHARED LANE · ALWAYS ON", title:"1 个 shared expert",
      summary:"所有 token 都执行，不参与 Top-K 竞争；形状与单个 routed expert 相同。",
      input:"all tokens", inputShape:"[B,S,6144]", output:"shared output", outputShape:"[B,S,6144]",
      formula:"Yshared=Wdown[SwiGLU-OAI(Wgate x,Wup x)]", formulaNote:"约 56.62M 参数/层，始终属于激活路径。",
      runtime:"MiniMaxM3MLP · shared_experts", source:"nvidia/model.py · MiniMaxM3MoE", sourceUrl:MODEL,
      code:`self.shared_experts = MiniMaxM3MLP(\n    intermediate_size=3072, reduce_results=False\n)`,
      weights:[
        {key:key(layer,"block_sparse_moe.shared_experts.gate_proj.weight"),shape:"[3072,6144]",dtype:"BF16",shard,runtime:"shared gate_up · gate",params:"18.87M"},
        {key:key(layer,"block_sparse_moe.shared_experts.up_proj.weight"),shape:"[3072,6144]",dtype:"BF16",shard,runtime:"shared gate_up · up",params:"18.87M"},
        {key:key(layer,"block_sparse_moe.shared_experts.down_proj.weight"),shape:"[6144,3072]",dtype:"BF16",shard,params:"18.87M"},
      ],
    },
    {
      id:"combine", tone:"output", kicker:"MOE OUTPUT", title:"Weighted combine + residual",
      summary:"4 个 routed 输出加权并乘 2，再与 shared expert 和 decoder residual 汇合。",
      input:"4 routed + shared + residual", inputShape:"逻辑上 6 × [B,S,6144]", output:"Xₗ₊₁", outputShape:"[B,S,6144]",
      formula:"Xₗ₊₁=residual+2·Σₑ∈𝓔 ŵₑEₑ(X)+Eshared(X)", formulaNote:"实现按 token 分派/合并，不会物化 6 份完整张量。",
      runtime:"FusedMoE output combine", source:"nvidia/model.py · MiniMaxM3MoE.forward", sourceUrl:MODEL,
      code:`return final_hidden_states.view(num_tokens, hidden_dim)`, weights:[],
    },
  ];
}

export const visionNodes: Node[] = [
  {
    id:"patch",tone:"vision",kicker:"VISION INPUT",title:"Conv3D Patch Embedding",
    summary:"把图像/视频切成 2×14×14 时空 patch，并投影到视觉宽度 1280。",
    input:"flat pixel patches",inputShape:"[Nᵥ,3×2×14×14]=[Nᵥ,1176]",output:"visual tokens",outputShape:"[Nᵥ,1280]",
    formula:"Xᵥ=Conv3D(pixels), kernel=stride=(2,14,14)",formulaNote:"672×672 单图示例 Nᵥ=(672/14)²=2304；动态分辨率会改变 Nᵥ。",
    runtime:"MiniMaxVLPatchEmbed.patch_embedding",source:"common/vision_tower.py · MiniMaxVLPatchEmbed",sourceUrl:VISION,
    code:`nn.Conv3d(3, 1280, kernel_size=(2,14,14), stride=(2,14,14), bias=False)`,
    weights:[{key:"vision_tower.vision_model.embeddings.patch_embedding.weight",shape:"[1280,3,2,14,14]",dtype:"F32",shard:"model-00059-of-00059.safetensors",params:"1.51M"}],
  },
  {
    id:"vit",tone:"vision",kicker:"ENCODER ×32",title:"Full attention Vision Transformer",
    summary:"16-head full attention + partial 3D RoPE，再经过 1280→5120→1280 GELU MLP；每段带残差。",
    input:"xₗ+(t,h,w)",inputShape:"[Nᵥ,1280] · 16 heads × 80",output:"xₗ₊₁",outputShape:"[Nᵥ,1280]",
    formula:"u=x+Attn(LN₁(x),RoPEₜₕ𝓌); x′=u+W₂GELU(W₁LN₂(u))",formulaNote:"80 维 head 中 t/h/w 各旋转 26 维，共 78 维，剩余 2 维直通。",
    runtime:"MiniMaxVLEncoderLayer ×32",source:"common/vision_tower.py · MiniMaxVLEncoderLayer",sourceUrl:VISION,
    code:`x = x + self.self_attn(self.layer_norm1(x), ...)\nresidual = x\nx, _ = self.fc1(self.layer_norm2(x))\nx, _ = self.fc2(self.act(x))\nreturn residual + x`,
    weights:[
      {key:"vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",shape:"[1280,1280]",dtype:"BF16",shard:"model-00059-of-00059.safetensors",runtime:"qkv_proj · q",note:"×32",params:"1.64M"},
      {key:"vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight",shape:"[1280,1280]",dtype:"BF16",shard:"model-00059-of-00059.safetensors",runtime:"qkv_proj · k",note:"×32",params:"1.64M"},
      {key:"vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight",shape:"[1280,1280]",dtype:"BF16",shard:"model-00059-of-00059.safetensors",runtime:"qkv_proj · v",note:"×32",params:"1.64M"},
      {key:"vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",shape:"[5120,1280]",dtype:"BF16",shard:"model-00059-of-00059.safetensors",runtime:"encoder.layers.0.fc1",note:"×32",params:"6.55M"},
      {key:"vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight",shape:"[1280,5120]",dtype:"BF16",shard:"model-00059-of-00059.safetensors",runtime:"encoder.layers.0.fc2",note:"×32",params:"6.55M"},
    ],
  },
  {
    id:"projector",tone:"projection",kicker:"WIDTH ALIGNMENT",title:"Multi-modal projector",
    summary:"两层 GELU MLP 把视觉宽度 1280 对齐到语言 hidden_size=6144。",
    input:"ViT output",inputShape:"[Nᵥ,1280]",output:"language-width tokens",outputShape:"[Nᵥ,6144]",
    formula:"Z=W₂·GELU(W₁Xᵥ+b₁)+b₂",formulaNote:"projector 中间宽度使用 text hidden size 6144。",
    runtime:"MiniMaxVLMultiModalProjector",source:"common/vision_tower.py · projector",sourceUrl:VISION,
    code:`x, _ = self.linear_1(x)  # 1280 → 6144\nx = self.act(x)\nx, _ = self.linear_2(x)  # 6144 → 6144`,
    weights:[
      {key:"multi_modal_projector.linear_1.weight",shape:"[6144,1280]",dtype:"BF16",shard:"model-00026-of-00059.safetensors",params:"7.86M"},
      {key:"multi_modal_projector.linear_2.weight",shape:"[6144,6144]",dtype:"BF16",shard:"model-00026-of-00059.safetensors",params:"37.75M"},
    ],
  },
  {
    id:"merge",tone:"vision",kicker:"TOKEN COMPRESSION · 2×2",title:"Spatial Patch Merger",
    summary:"每 2×2 个相邻 token 拼成 24576 通道，再压回 6144；视觉 token 数缩小 4 倍。",
    input:"2×2 grouped tokens",inputShape:"[Nᵥ/4,24576]",output:"merged visual tokens",outputShape:"[Nᵥ/4,6144]",
    formula:"Yᵥ=W₂·GELU(W₁·concat₂×₂(Z))",formulaNote:"672×672 示例从 2304 patch 压缩为 576 个视觉 token。",
    runtime:"MiniMaxVLPatchMerger · patch_merge_mlp",source:"common/vision_tower.py · MiniMaxVLPatchMerger",sourceUrl:VISION,
    code:`x = x.reshape(x.shape[0] // 4, -1)\nx, _ = self.linear_1(x)  # 24576 → 6144\nx, _ = self.linear_2(self.act(x))`,
    weights:[
      {key:"patch_merge_mlp.linear_1.weight",shape:"[6144,24576]",dtype:"BF16",shard:"model-00026-of-00059.safetensors",params:"150.99M"},
      {key:"patch_merge_mlp.linear_2.weight",shape:"[6144,6144]",dtype:"BF16",shard:"model-00026-of-00059.safetensors",params:"37.75M"},
    ],
  },
  {
    id:"fusion",tone:"output",kicker:"MODAL FUSION",title:"替换视觉占位 token",
    summary:"视觉向量替换 image/video placeholder 的文本 embedding，随后共享同一 60 层 decoder。",
    input:"text embeddings + Yᵥ",inputShape:"最后一维均为 6144",output:"fused embeddings",outputShape:"[B,S,6144]",
    formula:"X₀[visual_positions]←Yᵥ",formulaNote:"后续 attention/MoE 不再区分模态。",
    runtime:"merge_multimodal_embeddings",source:"nvidia/model.py · multimodal wrapper",sourceUrl:MODEL,
    code:`inputs_embeds = merge_multimodal_embeddings(\n  input_ids, inputs_embeds, multimodal_embeddings,\n  [image_token_id, video_token_id]\n)`,weights:[],
  },
];

function FlowNode({node,active,onClick}:{node:Node;active:boolean;onClick:()=>void}) {
  return <button className={`flow-node ${node.tone} ${active?"active":""}`} onClick={onClick} aria-pressed={active}>
    <span>{node.kicker}</span><div><b>{node.title}</b><small>{node.summary}</small></div><code>{node.outputShape}</code>
  </button>;
}

function Arrow({label}:{label?:string}) {
  return <div className="flow-arrow"><i/><span>↓</span>{label&&<code>{label}</code>}</div>;
}

function LinearFlow({nodes,active,onSelect}:{nodes:Node[];active:string;onSelect:(id:string)=>void}) {
  return <div className="linear-flow">{nodes.map((node,index)=><div key={node.id}><FlowNode node={node} active={active===node.id} onClick={()=>onSelect(node.id)}/>{index<nodes.length-1&&<Arrow label={node.outputShape}/>}</div>)}</div>;
}

function SparseFlow({nodes,active,onSelect}:{nodes:Node[];active:string;onSelect:(id:string)=>void}) {
  const get=(id:string)=>nodes.find(node=>node.id===id)!;
  const item=(id:string)=><FlowNode node={get(id)} active={active===id} onClick={()=>onSelect(id)}/>;
  return <div className="linear-flow sparse-flow">
    {item("sparse-proj")}
    <div className="split"><span>同一次投影，分成选择路径与内容路径</span><i>↙</i><i>↘</i></div>
    <div className="branch-grid">
      <section className="branch-lane"><header><b>INDEX BRANCH</b><span>决定读哪些块</span></header>{item("indexer")}<Arrow label="block scores"/>{item("topk")}</section>
      <section className="branch-lane main-lane"><header><b>MAIN BRANCH</b><span>生成语义输出</span></header><div className="cache-note"><b>Q/K/V + paged KV cache</b><code>Q [64×128] · K/V [4×128]</code></div><Arrow label="Top-16 控制 page 访问"/>{item("sparse-attn")}</section>
    </div>
    <div className="converge"><span>↓</span><code>O projection + residual + post norm</code></div>
    {item("router")}
    <div className="split moe-split"><span>同一输入并行进入 routed 与 shared 路径</span><i>↙</i><i>↘</i></div>
    <div className="branch-grid"><section className="branch-lane"><header><b>ROUTED ×4</b><span>128 选 4</span></header>{item("experts")}</section><section className="branch-lane"><header><b>SHARED ×1</b><span>始终执行</span></header>{item("shared")}</section></div>
    <div className="converge"><span>↓</span><code>weighted sum + shared + residual</code></div>{item("combine")}
  </div>;
}

function Inspector({node,tab,setTab}:{node:Node;tab:Tab;setTab:(tab:Tab)=>void}) {
  const tabs:[Tab,string][]=[["io","I/O Shape"],["formula","数学公式"],["code","代码"],["weights","模型权重"]];
  return <aside className="inspector-card">
    <header><span>{node.kicker}</span><h3>{node.title}</h3><p>{node.summary}</p><code>{node.runtime}</code></header>
    <div className="inspector-tabs" role="tablist">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div>
    <div className="inspector-body">
      {tab==="io"&&<div className="io-proof"><div><span>INPUT</span><code>{node.input}</code><b>{node.inputShape}</b></div><i>→</i><div><span>OUTPUT</span><code>{node.output}</code><b>{node.outputShape}</b></div></div>}
      {tab==="formula"&&<div className="formula-proof"><span>COMPUTE</span><code>{node.formula}</code><p>{node.formulaNote}</p></div>}
      {tab==="code"&&<div className="code-proof"><a href={node.sourceUrl} target="_blank" rel="noreferrer"><span>SOURCE</span><b>{node.source}</b><i>↗</i></a><pre><code>{node.code}</code></pre></div>}
      {tab==="weights"&&<div className="weight-list">{node.weights.length?node.weights.map(weight=><article key={weight.key}><div className="weight-key"><span>CHECKPOINT KEY</span><code>{weight.key}</code></div><div className="weight-meta"><b>{weight.dtype}</b><code>{weight.shape}</code>{weight.params&&<span>{weight.params} params</span>}{weight.note&&<em>{weight.note}</em>}</div><div className="weight-shard"><span>{weight.shard}</span>{weight.runtime&&<code>→ {weight.runtime}</code>}</div></article>):<div className="no-weight"><b>无可训练权重</b><p>这是缓存、mask、路由选择或张量合并操作；行为来自配置与运行时状态。</p></div>}</div>}
    </div><footer><i/> safetensors header + config + vLLM main</footer>
  </aside>;
}

function LayerRibbon({value,onChange}:{value:number;onChange:(layer:number)=>void}) {
  return <div className="layer-ribbon">{Array.from({length:60},(_,layer)=><button key={layer} className={`${layer<3?"dense":"sparse"} ${value===layer?"active":""}`} onClick={()=>onChange(layer)} aria-label={`Layer ${layer}`} aria-pressed={value===layer}><span>{layer}</span></button>)}</div>;
}

function Overview({chooseLayer,chooseVision}:{chooseLayer:(layer:number)=>void;chooseVision:()=>void}) {
  return <section className="panel overview-card" id="architecture">
    <div className="panel-heading"><div><span className="eyebrow">END-TO-END FORWARD</span><h2>先看模型骨架，再下钻一层</h2><p>两种输入在 embedding 处汇合；60 层语言主干只有两种层型。</p></div><div className="verified-pill"><i/> 59 shards aligned</div></div>
    <div className="overview-flow">
      <div className="input-branches"><button onClick={()=>chooseLayer(0)}><span>TEXT</span><b>input_ids → token embeddings</b><code>[B,S] → [B,S,6144]</code></button><button className="vision-input" onClick={chooseVision}><span>IMAGE / VIDEO</span><b>Vision Tower ×32</b><code>pixels → [Nᵥ/4,6144]</code></button></div>
      <div className="merge-arrows"><span>↘</span><code>replace visual placeholders</code><span>↙</span></div>
      <button className="overview-stage fusion-stage" onClick={()=>chooseLayer(0)}><span>FUSION</span><b>Unified token embeddings</b><code>[B,S,6144]</code></button><Arrow label="shared hidden width H=6144"/>
      <div className="decoder-stack"><button className="overview-stage dense-stage" onClick={()=>chooseLayer(0)}><span>L0–L2 · ×3</span><b>Dense GQA + Dense MLP</b><code>full causal history · ≈333.46M params/layer</code></button><div className="stack-arrow">↓ <code>[B,S,6144]</code></div><button className="overview-stage sparse-stage" onClick={()=>chooseLayer(3)}><span>L3–L59 · ×57</span><b>MSA + Top-4 MoE + Shared Expert</b><code>Top-16 blocks · ≈7.416B params/layer</code></button></div>
      <Arrow label="X₆₀ [B,S,6144]"/><button className="overview-stage output-stage" onClick={()=>chooseLayer(59)}><span>OUTPUT</span><b>final RMSNorm → lm_head</b><code>[B,S,6144] → [B,S,200064]</code></button>
    </div>
  </section>;
}

const ledger=[
  ["Dense layer · attention","106.95M","1.45%","amber","Q/K/V/O 主矩阵"],
  ["Dense layer · FFN","226.49M","3.05%","amber","12288-wide gate/up/down"],
  ["Sparse layer · attention + index","110.89M","1.50%","green","主 GQA + Index projections"],
  ["Sparse layer · 128 routed experts","7.248B","97.73%","violet","56.62M × 128 experts"],
  ["Sparse layer · shared + router","57.41M",".77%","rose","shared expert + FP32 router"],
];

export default function Home(){
  const [scope,setScope]=useState<Scope>("language");
  const [layer,setLayer]=useState(3);
  const [selected,setSelected]=useState("indexer");
  const [tab,setTab]=useState<Tab>("weights");
  const [dark,setDark]=useState(false);
  const nodes=scope==="vision"?visionNodes:layer<3?denseNodes(layer):sparseNodes(layer);
  const activeNode=nodes.find(node=>node.id===selected)??nodes[0];
  const activeId=activeNode.id;
  const chooseLayer=(next:number)=>{setScope("language");setLayer(next);setSelected(next<3?"dense-qkv":"indexer");setTab("weights")};
  const chooseVision=()=>{setScope("vision");setSelected("patch");setTab("weights")};

  return <main className={`app ${dark?"dark":""}`}><div className="ambient one"/><div className="ambient two"/><div className="shell">
    <header className="topbar"><a className="brand" href="#architecture"><span className="brand-mark"><i/><i/><i/></span><div><b>MiniMax-M3</b><small>ARCHITECTURE TRACE</small></div></a><div className="top-facts"><span><i/> facts aligned</span><b>428B total</b><b>23B active</b><b>1M context</b></div><button className="theme-toggle" onClick={()=>setDark(value=>!value)} aria-label="切换明暗主题"><span>{dark?"☀":"☾"}</span>{dark?"浅色":"深色"}</button></header>
    <section className="hero"><div><span>MODEL ARCHITECTURE · CODE · WEIGHTS</span><h1>把 MiniMax-M3<br/>从一张图追到一个张量</h1><p>结构不是摘要。点击任一模块，核对输入输出 shape、数学公式、vLLM 实现，以及官方 BF16 checkpoint 中真实存在的权重键。</p></div><div className="checkpoint-card"><span>OFFICIAL CHECKPOINT</span><strong>869.16 <small>GB</small></strong><p>59 safetensors shards</p><div><span>hidden</span><b>6144</b><span>Q : KV</span><b>64 : 4</b><span>layers</span><b>3 + 57</b></div></div></section>
    <Overview chooseLayer={chooseLayer} chooseVision={chooseVision}/>
    <div className="workspace" id="trace">
      <aside className="control-card"><div className="control-heading"><span className="eyebrow">TRACE CONTROL</span><h2>选择要下钻的结构</h2></div><fieldset><legend>模型分支</legend><div className="scope-tabs"><button className={scope==="language"?"active":""} onClick={()=>chooseLayer(layer)}>语言主干</button><button className={scope==="vision"?"active":""} onClick={chooseVision}>视觉塔</button></div></fieldset>{scope==="language"&&<fieldset><legend>Decoder layer</legend><div className="layer-picker"><strong>L{layer}</strong><span className={layer<3?"dense":"sparse"}>{layer<3?"Dense GQA + MLP":"MSA + SparseMoE"}</span></div><input className="layer-range" type="range" min="0" max="59" value={layer} onChange={event=>chooseLayer(Number(event.target.value))}/><div className="range-ends"><span>L0</span><span>L59</span></div></fieldset>}<fieldset><legend>本层证据</legend><div className="evidence-list"><div><i/>输入输出 shape</div><div><i/>数学公式</div><div><i/>vLLM 运行时代码</div><div><i/>checkpoint key / dtype / shard</div></div></fieldset><div className="reading-note"><b>读图方法</b><p>沿连线看数据流，再点击节点。权重页先显示 checkpoint 名字，箭头后是 vLLM 融合后的运行时参数。</p></div></aside>
      <div className="results">
        <section className="panel trace-panel"><div className="panel-heading"><div><span className="eyebrow">{scope==="vision"?"VISION PIPELINE":layer<3?`DECODER L${layer} · DENSE`:`DECODER L${layer} · MSA + MOE`}</span><h2>{scope==="vision"?"Pixels → language-width visual tokens":layer<3?"完整历史注意力与 Dense FFN":"索引选块、精确注意力与 Top-4 MoE"}</h2><p>{scope==="vision"?"以 672×672 单图展示 shape；Nᵥ 会随动态分辨率变化。":`当前层主要权重位于 ${layerShard(layer)}。`}</p></div><div className={`layer-badge ${scope==="vision"?"vision":layer<3?"dense":"sparse"}`}>{scope==="vision"?"ViT ×32":`L${layer}`}</div></div>{scope==="language"&&<LayerRibbon value={layer} onChange={chooseLayer}/>}<div className="explorer-grid"><div className="graph-canvas">{scope==="vision"||layer<3?<LinearFlow nodes={nodes} active={activeId} onSelect={setSelected}/>:<SparseFlow nodes={nodes} active={activeId} onSelect={setSelected}/>}</div><Inspector node={activeNode} tab={tab} setTab={setTab}/></div></section>
        <section className="panel ledger-card"><div className="panel-heading"><div><span className="eyebrow">PARAMETER LEDGER</span><h2>为什么 57 个 MoE 层主导 428B</h2><p>条形按单个 sparse layer ≈7.416B 归一；norm/bias 对总量影响很小。</p></div><span className="unit-pill">parameters / layer</span></div><div className="ledger-grid"><div className="ledger-bars">{ledger.map(([label,value,width,tone,note])=><div className="ledger-row" key={label}><div><span>{label}</span><b>{value}</b></div><div className="bar-track"><i className={tone} style={{width}}/></div><small>{note}</small></div>)}</div><div className="ledger-summary"><article><span>DENSE LAYER ×3</span><b>≈333.46M</b><p>attention 106.95M + dense FFN 226.49M</p></article><article><span>SPARSE LAYER ×57</span><b>≈7.416B</b><p>128 routed experts 占约 97.7%</p></article><article><span>VISION</span><b>≈19.66M / layer</b><p>32 层约 629M，另有约 234M bridge</p></article></div></div></section>
        <section className="panel mapping-card"><div className="panel-heading"><div><span className="eyebrow">CHECKPOINT → RUNTIME</span><h2>同一组权重，为什么代码里名字不一样</h2><p>checkpoint 保存分离矩阵；vLLM 加载时为了减少 GEMM 次数进行打包。</p></div></div><div className="mapping-table"><div><span>CHECKPOINT</span><code>q_proj · k_proj · v_proj</code><i>→</i><b>QKVParallelLinear.qkv_proj</b><small>Dense 与 Vision attention</small></div><div><span>CHECKPOINT</span><code>q/k/v + index_q/index_k</code><i>→</i><b>MinimaxM3QKV…WithIndexer</b><small>MSA 一次 GEMM 输出五路</small></div><div><span>CHECKPOINT</span><code>gate_proj · up_proj</code><i>→</i><b>gate_up_proj</b><small>Dense MLP 与 shared expert</small></div><div><span>CHECKPOINT</span><code>expert.*.w1 · w3 · w2</code><i>→</i><b>FusedMoE w13 · w2</b><small>128 routed experts</small></div></div><div className="mapping-warning"><b>不要混淆：</b><span>checkpoint 键来自官方 weight index 与分片 header；运行时名来自 vLLM main。两者通过 loader 映射相连。</span></div></section>
      </div>
    </div>
    <footer className="sources-panel"><div><span className="eyebrow">PRIMARY SOURCES · VERIFIED 2026-08-16</span><h2>事实来源与解释边界</h2><p>参数、dtype、shape 和 shard 来自发布配置、权重索引及 safetensors header；运行路径以 vLLM main 为准；MSA 数学语义以技术报告为准。</p></div><div className="source-links"><a href="https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json" target="_blank" rel="noreferrer"><b>config.json</b><span>维度、稀疏与 MoE 配置</span><i>↗</i></a><a href="https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/model.safetensors.index.json" target="_blank" rel="noreferrer"><b>weight index</b><span>checkpoint key → shard</span><i>↗</i></a><a href={MODEL} target="_blank" rel="noreferrer"><b>vLLM MiniMax-M3</b><span>decoder、MSA、MoE、loader</span><i>↗</i></a><a href={VISION} target="_blank" rel="noreferrer"><b>vLLM vision tower</b><span>ViT、projector、merger</span><i>↗</i></a><a href="https://arxiv.org/abs/2606.13392" target="_blank" rel="noreferrer"><b>MSA report</b><span>Index branch 与 sparse attention</span><i>↗</i></a><a href="https://github.com/MiniMax-AI/MiniMax-M3" target="_blank" rel="noreferrer"><b>Official repository</b><span>428B / 23B / 1M context</span><i>↗</i></a></div><small>页面 shape 是逻辑布局；tensor parallel、batch packing、cache dtype 与 kernel layout 会改变物理张量。参数账本用于解释结构，不等同于某个并行配置下的单卡显存。</small></footer>
  </div></main>;
}
