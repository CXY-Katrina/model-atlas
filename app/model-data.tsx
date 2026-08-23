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
      formula: "Wdown[min(g,7)·σ(1.702·min(g,7))·(clip(u,−7,7)+1)]", formulaNote: "g=Wgate x，u=Wup x；self.act_fn = SiluAndMulWithClamp(limit=7.0, alpha=1.702, beta=1.0)。",
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
      input:"GemmaRMSNorm(Xₗ)", inputShape:"[B,S,6144]", output:"Q|K|V|Qidx|Kidx", outputShape:"8192 | 512 | 512 | 512 | 128",
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
