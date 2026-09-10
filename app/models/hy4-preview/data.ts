import type { Node } from "../types";
import { CODE_URL, ATTENTION_URL, HC_URL, MOE_URL } from "./sources";

// checkpoint:HF tencent/Hy4-preview,131 个 safetensors 分片(lm_head.weight 在 00053)。
export const SHARD = "131 分片 · tencent/Hy4-preview";

function hcPreNode(scope: string, zh: string): Node {
  return {
    id: `hcpre-${scope}`, tone: "norm", kicker: `iHC PRE · ${zh}`, title: "hc_pre:4 通道归约",
    summary: "把 [T,4,6144] 的 4 通道残差归约为单通道交给子块,同时产生 post 门;FP32 计算。",
    input: "4 通道 hidden", inputShape: "[T,4,6144]",
    output: "归约 hidden + post 门", outputShape: "[T,6144] + [T,4]",
    formula: "y=Σᵢ σ(mixes·s₀+b₀)ᵢ·xᵢ",
    formulaNote: "flatten 后 RMS(ε=1e−5)·hc_fn(24576→8) 得 8 个 logits;pre/post 各 4 个 σ 门。hc_post 无参数:y=post·x+residual。",
    runtime: "HYV4HCPreLayer.forward(可选 HPC 单内核)",
    source: "nvidia/hc.py · HYV4HCPreLayer.forward", sourceUrl: HC_URL,
    code: "x_flat = x.flatten(1).float()\nrsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)\nmixes = self.hc_fn(x_flat)[0] * rsqrt\npre = torch.sigmoid(mixes[..., :hc] * s0 + b0) + eps\npost = 2.0 * torch.sigmoid(mixes[..., hc:] * s1 + b1) + eps\ny = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)",
    weights: [
      { key: `model.layers.{l}.hc_${scope}_layer.hc_pre.hc_fn.weight`, shape: "[8, 24576]", dtype: "F32", shard: SHARD, params: "196,608" },
      { key: `model.layers.{l}.hc_${scope}_layer.hc_pre.hc_base`, shape: "[8]", dtype: "F32", shard: SHARD, params: "8" },
      { key: `model.layers.{l}.hc_${scope}_layer.hc_pre.hc_scale`, shape: "[2]", dtype: "F32", shard: SHARD, params: "2" },
    ],
  };
}

export function coreNodes(): Node[] {
  return [
    {
      id: "input", tone: "output", kicker: "DECODER LAYER INPUT", title: "hidden_states",
      summary: "L0 层首来自 embedding;层间是上一层的 4 通道输出。iHC 下 residual=None。",
      input: "Xₗ", inputShape: "[T,6144] 或 [T,4,6144]",
      output: "prepare_input 后", outputShape: "[T,4,6144]",
      formula: "residual ≡ 4 通道 hidden 本身", formulaNote: "层首将 [T,6144] 广播为 4 通道;此后不再有单一 residual 流。",
      runtime: "HYV4HCLayer.prepare_input", source: "nvidia/model.py · _forward_ihc", sourceUrl: CODE_URL,
      code: "hidden_states = self.hc_attn_layer.prepare_input(hidden_states)\nhidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)",
      weights: [],
    },
    hcPreNode("attn", "注意力子块前"),
    {
      id: "norm1", tone: "norm", kicker: "PRE-NORM", title: "input_layernorm",
      summary: "对归约后的单通道 hidden 做 RMSNorm,输出唯一的归一化张量。",
      input: "归约 hidden", inputShape: "[T,6144]", output: "X̂", outputShape: "[T,6144]",
      formula: "X̂=x/RMS(x)·γ", formulaNote: "ε=1e−5;注意 hc_pre 内部的 RMS 与这里是两次独立归一化。",
      runtime: "RMSNorm.forward", source: "nvidia/model.py · HYV4DecoderLayer.__init__", sourceUrl: CODE_URL,
      code: "hidden_states = self.input_layernorm(hidden_states)",
      weights: [{ key: "model.layers.{l}.input_layernorm.weight", shape: "[6144]", dtype: "BF16", shard: SHARD, params: "6,144" }],
    },
    {
      id: "hcpost1", tone: "output", kicker: "iHC POST · 注意力后", title: "hc_post:散回 4 通道",
      summary: "无参数模块:子块输出乘 post 门后加回 4 通道残差。",
      input: "attn 输出 + 残差 + post 门", inputShape: "[T,6144]+[T,4,6144]+[T,4]",
      output: "更新后的 4 通道 hidden", outputShape: "[T,4,6144]",
      formula: "y[n,i,d]=post·x+residual", formulaNote: "post = 2·σ(·)+1e−6,幅度 hc_magnitude=2.0;FP32 计算后转回 dtype。",
      runtime: "HYV4HCPostLayer.forward", source: "nvidia/hc.py · HYV4HCPostLayer.forward", sourceUrl: HC_URL,
      code: "post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)\ny = post_gated + residual",
      weights: [],
    },
    hcPreNode("mlp", "FFN 子块前"),
    {
      id: "norm2", tone: "norm", kicker: "POST-ATTN NORM", title: "post_attention_layernorm",
      summary: "注意力子块之后、FFN 子块之前的 RMSNorm。",
      input: "归约 hidden", inputShape: "[T,6144]", output: "Û", outputShape: "[T,6144]",
      formula: "Û=x/RMS(x)·γpost", formulaNote: "ε=1e−5。",
      runtime: "RMSNorm.forward", source: "nvidia/model.py · HYV4DecoderLayer.__init__", sourceUrl: CODE_URL,
      code: "hidden_states = self.post_attention_layernorm(hidden_states)",
      weights: [{ key: "model.layers.{l}.post_attention_layernorm.weight", shape: "[6144]", dtype: "BF16", shard: SHARD, params: "6,144" }],
    },
    {
      id: "hcpost2", tone: "output", kicker: "iHC POST · FFN 后", title: "hc_post:散回 4 通道",
      summary: "FFN 子块输出乘 post 门后加回 4 通道残差,作为层输出。",
      input: "FFN 输出 + 残差 + post 门", inputShape: "[T,6144]+[T,4,6144]+[T,4]",
      output: "层输出 hidden", outputShape: "[T,4,6144]",
      formula: "y[n,i,d]=post·x+residual", formulaNote: "层间 residual=None:4 通道张量本身就是残差载体。",
      runtime: "HYV4HCPostLayer.forward", source: "nvidia/hc.py · HYV4HCPostLayer.forward", sourceUrl: HC_URL,
      code: "hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)",
      weights: [],
    },
  ];
}

export function attentionNodes(mode: "full" | "shared"): Node[] {
  const nodes: Node[] = [
    {
      id: "aproj", tone: "projection", kicker: "FUSED DOWN PROJ · 6144→[2048|576]", title: "fused_qkv_a_proj",
      summary: "一次 GEMM 同时产生 q_a(2048) 与 (kv 512 | k_pe 64);checkpoint 保存两块矩阵,load 时堆叠。",
      input: "X̂", inputShape: "[T,6144]", output: "q_a | kv | k_pe", outputShape: "2048 | 512 | 64",
      formula: "[q_a‖kv_a‖k_pe]=X̂·Wᵀ", formulaNote: "q_a_proj 与 kv_a_proj_with_mqa 经 stacked_params_mapping 合并为 shard 0/1。",
      runtime: "MergedColumnParallelLinear · fused_qkv_a_proj", source: "nvidia/attention.py · HYV4MLAAttention", sourceUrl: ATTENTION_URL,
      code: "qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]\nq_c, kv_lora = qkv_lora.split([2048, 576], dim=-1)",
      weights: [
        { key: "model.layers.{l}.self_attn.q_a_proj.weight", shape: "[2048,6144]", dtype: "BF16", shard: SHARD, runtime: "fused_qkv_a_proj · shard 0", params: "12.58M" },
        { key: "model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight", shape: "[576,6144]", dtype: "BF16", shard: SHARD, runtime: "fused_qkv_a_proj · shard 1", params: "3.54M" },
      ],
    },
    {
      id: "qan", tone: "norm", kicker: "Q LORA NORM", title: "q_a_layernorm",
      summary: "对 q_lora 向量做 RMSNorm;q̃ 同时是 indexer 查询的来源。",
      input: "q_a", inputShape: "[T,2048]", output: "q̃", outputShape: "[T,2048]",
      formula: "q̃=q/RMS(q)·γ", formulaNote: "ε=1e−5。",
      runtime: "RMSNorm.forward", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "q_c = self.q_a_layernorm(q_c)",
      weights: [{ key: "model.layers.{l}.self_attn.q_a_layernorm.weight", shape: "[2048]", dtype: "BF16", shard: SHARD, params: "2,048" }],
    },
    {
      id: "qb", tone: "projection", kicker: "UP PROJ · 2048→64×256", title: "q_b_proj",
      summary: "把 q̃ 投到 64 个 head,每 head 192 nope + 64 rope 维。",
      input: "q̃", inputShape: "[T,2048]", output: "Q", outputShape: "[T,64,256]",
      formula: "Q=q̃·W_qbᵀ", formulaNote: "每 head 前 192 维 nope,后 64 维 rope(参与旋转)。",
      runtime: "ColumnParallelLinear · q_b_proj", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "q = self.q_b_proj(q_c)[0].view(-1, 64, 256)",
      weights: [{ key: "model.layers.{l}.self_attn.q_b_proj.weight", shape: "[16384,2048]", dtype: "BF16", shard: SHARD, params: "33.55M" }],
    },
    {
      id: "kvn", tone: "norm", kicker: "KV LORA NORM", title: "kv_a_layernorm",
      summary: "对 512 维压缩向量 k̃ 做 RMSNorm;k̃ 进 cache,k_pe 单独旋转。",
      input: "kv_lora", inputShape: "[T,512]", output: "k̃ | k_pe", outputShape: "512 | 64",
      formula: "k̃=kv/RMS(kv)·γ", formulaNote: "ε=1e−5;k_pe 64 维不经过本 norm。",
      runtime: "RMSNorm.forward", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "kv_c, k_pe = kv_lora.split([512, 64], dim=-1)\nkv_c_normed = self.kv_a_layernorm(kv_c)",
      weights: [{ key: "model.layers.{l}.self_attn.kv_a_layernorm.weight", shape: "[512]", dtype: "BF16", shard: SHARD, params: "512" }],
    },
    {
      id: "kvb", tone: "projection", kicker: "KV UP PROJ · 512→64×448", title: "kv_b_proj",
      summary: "从 k̃ 恢复每头的 K_nope(192)与 V(256);KV cache 只存压缩向量。默认(sink 生效)走吸收式:W_UK 折入 q、W_UV 作用于 latent 输出,K_nope/V 不物化,仅 dense/masked-MHA prefill 分支显式物化。",
      input: "k̃(cache 历史)", inputShape: "[T,512]", output: "K_nope | V", outputShape: "[T,64,192] | [T,64,256]",
      formula: "[K_nope‖V]=k̃·W_kvbᵀ", formulaNote: "稀疏内核在候选 token 上消费投影结果。",
      runtime: "ColumnParallelLinear · kv_b_proj", source: "nvidia/attention.py · __init__", sourceUrl: ATTENTION_URL,
      code: "self.kv_b_proj = ColumnParallelLinear(512, 64 * (192 + 256), ...)",
      weights: [{ key: "model.layers.{l}.self_attn.kv_b_proj.weight", shape: "[28672,512]", dtype: "BF16", shard: SHARD, params: "14.68M" }],
    },
    {
      id: "rope", tone: "attention", kicker: "ROPE · θ=1e7 · INTERLEAVED", title: "RoPE(q rope 段 + k_pe)",
      summary: "只旋转每 head 的后 64 维与 MQA 共享的 k_pe;checkpoint 为 Megatron/PTM 交错布局。",
      input: "Q 的 rope 段 + k_pe + positions", inputShape: "[T,64,64]+[T,1,64]+[T]",
      output: "旋转后 Q rope 段 / 旋转后 k_pe", outputShape: "[T,64,64] + [T,1,64](Q 整体记作 Qᵣ)",
      formula: "q' = R_p q", formulaNote: "is_neox_style=False;旋转后的 k_pe 由 cache 写入路径消费。用 NeoX 布局会破坏 indexer 的相对位置依赖。",
      runtime: "get_rope(is_neox_style=False)", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "q[..., 192:], k_pe = self.rotary_emb(positions, q[..., 192:], k_pe)",
      weights: [],
    },
    {
      id: "cache", tone: "index", kicker: "COMPRESSED KV CACHE", title: "MLA 压缩 KV cache",
      summary: "每 token 只存 512 维 k̃ + 64 维 k_pe;K_nope/V 在注意力侧由 kv_b_proj 恢复。",
      input: "k̃ + 旋转后 k_pe + slot_mapping", inputShape: "[T,512]+[T,1,64]",
      output: "候选历史", outputShape: "[T,576] · 默认 BF16(k_pe 为 RoPE 后)",
      formula: "cache[slot]←(k̃,k_pe)", formulaNote: "语义压缩维度 512+64 不随 dtype 变;BF16 物理 576 元素/head,量化部署升格 fp8_ds_mla(656 B/token 自定义布局)。",
      runtime: "MLAAttention · FlashMLA sparse", source: "nvidia/flashmla_sparse.py · backend", sourceUrl: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla_sparse.py",
      code: "# 压缩布局每 token 512+64;dtype 由 cache_config 决定(默认 BF16,\n# 量化请求时后端升格 fp8_ds_mla)。\nkv_cache[slot] = concat(kv_c_normed, k_pe)",
      weights: [],
    },
  ];
  if (mode === "full") {
    nodes.push(
      {
        id: "iwqb", tone: "index", kicker: "INDEXER · Q 侧", title: "wq_b 索引查询",
        summary: "从 q̃ 投出 32 个索引头,每头 128 维(64 nope + 64 rope)。",
        input: "q̃", inputShape: "[T,2048]", output: "Qidx", outputShape: "[T,32,128]",
        formula: "Qidx=q̃·W_wqbᵀ", formulaNote: "ReplicatedLinear,不做 TP 切分;indexer 内另有独立 interleaved RoPE 实例,其 q_pe/k_pe 各 64 维旋转后才打分。",
        runtime: "Indexer.prepare_inputs", source: "nvidia/attention.py · Indexer", sourceUrl: ATTENTION_URL,
        code: "q, _ = self.wq_b(qr)\nq = q.view(-1, 32, 128)",
        weights: [{ key: "model.layers.{l}.self_attn.indexer.wq_b.weight", shape: "[4096,2048]", dtype: "BF16", shard: SHARD, params: "8.39M" }],
      },
      {
        id: "iwk", tone: "index", kicker: "INDEXER · K+权重 · 融合 GEMM", title: "wk_weights_proj",
        summary: "一次 GEMM 同时产生索引 key(128)与 32 个 head 权重 w。",
        input: "hidden", inputShape: "[T,6144]", output: "k | w", outputShape: "[T,128] | [T,32]",
        formula: "[k‖w]=hidden·Wᵀ", formulaNote: "wk 若以 FP8 存储,load 时上转 BF16 以维持融合;本 checkpoint 为 BF16,两块权重直接并入 wk_weights_proj(shard 0/1)。",
        runtime: "MergedColumnParallelLinear · disable_tp", source: "nvidia/attention.py · Indexer.__init__", sourceUrl: ATTENTION_URL,
        code: "kw, _ = self.wk_weights_proj(hidden_states)\nk = kw[:, :128]; weights = kw[:, 128:]",
        weights: [
          { key: "model.layers.{l}.self_attn.indexer.wk.weight", shape: "[128,6144]", dtype: "BF16", shard: SHARD, runtime: "wk_weights_proj · shard 0", params: "0.79M" },
          { key: "model.layers.{l}.self_attn.indexer.weights_proj.weight", shape: "[32,6144]", dtype: "BF16", shard: SHARD, runtime: "wk_weights_proj · shard 1", params: "0.20M" },
        ],
      },
      {
        id: "ikn", tone: "norm", kicker: "INDEXER · KEY NORM", title: "k_norm = LayerNorm",
        summary: "索引 key 的归一化是 LayerNorm(带均值减除),与主干的 RMSNorm 不同。",
        input: "k", inputShape: "[T,128]", output: "LayerNorm 后 k", outputShape: "[T,128]",
        formula: "k=(k−μ)/√(σ²+1e−6)·γ+β", formulaNote: "ε=1e−6;γ、β 各 128;norm 后 k 的 rope 64 维由 indexer 专属 interleaved RoPE 旋转。",
        runtime: "nn.LayerNorm(128, eps=1e-6)", source: "nvidia/attention.py · Indexer.__init__", sourceUrl: ATTENTION_URL,
        code: "self.k_norm = LayerNorm(self.head_dim, eps=1e-6)\nk = self.k_norm(k)",
        weights: [
          { key: "model.layers.{l}.self_attn.indexer.k_norm.weight", shape: "[128]", dtype: "BF16", shard: SHARD, params: "128" },
          { key: "model.layers.{l}.self_attn.indexer.k_norm.bias", shape: "[128]", dtype: "BF16", shard: SHARD, params: "128" },
        ],
      },
      {
        id: "iquant", tone: "index", kicker: "FP8 · UE8M0", title: "q 量化 + 权重折叠",
        summary: "q 按 128 元素一组做 FP8 量化;head 权重折叠 q_scale、1/√128、1/√32。",
        input: "Qidx + w", inputShape: "[T,32,128]+[T,32]",
        output: "FP8 q + 折叠后 w", outputShape: "fp8 + [T,32]",
        formula: "w'=w·q_scale/√128/√32", formulaNote: "k 的量化融合在插 cache 时完成。",
        runtime: "per_token_group_quant_fp8(ue8m0)", source: "nvidia/attention.py · prepare_inputs", sourceUrl: ATTENTION_URL,
        code: "q_fp8, q_scale = per_token_group_quant_fp8(q, 128, use_ue8m0=True)\nweights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * 32**-0.5",
        weights: [],
      },
      {
        id: "iscore", tone: "index", kicker: "INDEXER · 打分", title: "加权打分 + Top-2048",
        summary: "q 与全部历史 k 打分(经 w 加权),causal 内选 top-2048 写入共享 buffer。",
        input: "FP8 q + 历史 k + w'", inputShape: "[T,32,128]+[T,128]+[T,32]",
        output: "top-2048 索引", outputShape: "[T,2048] int32",
        formula: "I=TopK₂₀₄₈(Σₕ w'⟨q,k⟩)", formulaNote: "写进模型级共享 topk_indices_buffer,由同层注意力与后续 shared 层消费;候选不足 T_idx 的槽位以 −1(no-token sentinel)填充,消费端按 −1 判空。",
        runtime: "SparseAttnIndexer", source: "nvidia/attention.py · Indexer.forward", sourceUrl: ATTENTION_URL,
        code: "hidden_states, q_quant, k, weights = self.prepare_inputs(...)\nreturn self.indexer_op(hidden_states, q_quant, k, weights)",
        weights: [],
      },
    );
  } else {
    nodes.push({
      id: "ishared", tone: "index", kicker: "SHARED INDEXER · 57 层", title: "复用最近 full 层索引",
      summary: "shared 层不建 indexer:直接读取共享 buffer 中最近前驱 full 层写入的 top-2048 索引。",
      input: "topk_indices_buffer", inputShape: "[T,2048] int32",
      output: "候选 token 集合", outputShape: "每 query 2048",
      formula: "I_t = buffer[nearest_full(t)]", formulaNote: "checkpoint 权重索引证实:57 个 shared 层没有 indexer.* 权重;读取的 −1 槽位视为空位。",
      runtime: "skip_topk=True 路径", source: "nvidia/attention.py · _indexer_and_attn", sourceUrl: ATTENTION_URL,
      code: "if self.indexer is not None and self.is_sparse and not self.skip_topk:\n    self.indexer(...)  # shared 层跳过\nout.copy_(self.mla_attn(q, kv_c_normed, k_pe, ...))",
      weights: [],
    });
  }
  nodes.push(
    {
      id: "qk", tone: "attention", kicker: "SPARSE MLA · 候选 2048", title: "QKᵀ/√256 + causal",
      summary: "稀疏的是访问集合:每 query 只读 top-2048 候选,候选内仍是精确 softmax。",
      input: "Q + 候选 K", inputShape: "[T,64,256]+候选",
      output: "scores", outputShape: "[64,≤2048]/query",
      formula: "S=QKᵀ/√256+M", formulaNote: "scale 用 qk_head_dim=256(192+64),不是 512 或 128。",
      runtime: "FlashMLA sparse backend", source: "nvidia/flashmla_sparse.py", sourceUrl: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla_sparse.py",
      code: "attn_out = self.mla_attn(q, kv_c_normed, k_pe, output_shape=...)",
      weights: [],
    },
    {
      id: "sink", tone: "attention", kicker: "LEARNABLE SINK · 每 head", title: "softmax + sink 折扣",
      summary: "每 head 一个 FP32 sink(checkpoint 初始 0),折进 softmax 分母。",
      input: "scores + sink_h", inputShape: "[64,≤2048]+[64]",
      output: "P", outputShape: "[64,≤2048]",
      formula: "out×exp(lse)/(exp(lse)+exp(sink))", formulaNote: "稀疏路径不做 dense 回退;后端不支持 sink 时该参数降为 BF16 并告警(checkpoint 为 FP32,初始 0)。",
      runtime: "HYV4FlashMLASparseImpl(sinks kwarg)", source: "nvidia/flashmla_sparse.py · HYV4FlashMLASparseImpl", sourceUrl: "https://github.com/vllm-project/vllm/blob/main/vllm/models/hy_v4/nvidia/flashmla_sparse.py",
      code: "out *= exp(lse) / (exp(lse) + exp(sink))",
      weights: [{ key: "model.layers.{l}.self_attn.learnable_sink_param", shape: "[64]", dtype: "F32", shard: SHARD, params: "64" }],
    },
    {
      id: "pv", tone: "attention", kicker: "CONTEXT · 同一候选集", title: "P·V",
      summary: "按同一候选顺序聚合 V;V 由 kv_b_proj 从压缩 cache 恢复。",
      input: "P + V", inputShape: "[64,≤2048]+[T,64,256]",
      output: "heads", outputShape: "[T,64,256]",
      formula: "O=P·V", formulaNote: "v_head_dim=256,拼成 16384 通道。",
      runtime: "FlashMLA sparse kernel", source: "nvidia/attention.py · _indexer_and_attn", sourceUrl: ATTENTION_URL,
      code: "out.copy_(self.mla_attn(q, kv_c_normed, k_pe, output_shape=out.shape))",
      weights: [],
    },
    {
      id: "gate", tone: "moe", kicker: "GATED MLA · ELEMENTWISE", title: "⊙ σ(linear_gate(hidden))",
      summary: "门从子块输入 hidden 投影(64×256),与注意力输出逐元素相乘。",
      input: "heads + hidden", inputShape: "[T,64,256]+[T,6144]",
      output: "门控后 heads", outputShape: "[T,16384]",
      formula: "Y=O⊙σ(g)", formulaNote: "elementwise 门控发生在 o_proj 之前。",
      runtime: "ColumnParallelLinear · linear_gate", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "gate_score = self.linear_gate(hidden_states)[0]\nattn_out = attn_out * torch.sigmoid(gate_score)",
      weights: [{ key: "model.layers.{l}.self_attn.linear_gate.weight", shape: "[16384,6144]", dtype: "BF16", shard: SHARD, params: "100.66M" }],
    },
    {
      id: "oproj", tone: "projection", kicker: "OUTPUT PROJ · 16384→6144", title: "o_proj",
      summary: "把 64×256 的门控 heads 投回隐藏宽度。",
      input: "门控后 heads", inputShape: "[T,16384]", output: "Y_attn", outputShape: "[T,6144]",
      formula: "Y_attn=Y·W_oᵀ", formulaNote: "交回 hc_post 散回 4 通道。",
      runtime: "RowParallelLinear · o_proj", source: "nvidia/attention.py · forward", sourceUrl: ATTENTION_URL,
      code: "out, _ = self.o_proj(attn_out)\nreturn out",
      weights: [{ key: "model.layers.{l}.self_attn.o_proj.weight", shape: "[6144,16384]", dtype: "BF16", shard: SHARD, params: "100.66M" }],
    },
  );
  return nodes;
}

export function denseFfnNodes(): Node[] {
  return [
    {
      id: "gateup", tone: "moe", kicker: "DENSE FFN · L0 · H_ffn=18432", title: "gate/up 投影",
      summary: "MergedColumnParallelLinear 融合 gate 与 up 两次投影;L0 唯一的 dense FFN。",
      input: "Û", inputShape: "[T,6144]", output: "packed gate_up", outputShape: "[T,36864]",
      formula: "[g‖u]=Û·Wᵀ", formulaNote: "dense FFN 不做 swiglu clamp。",
      runtime: "HYV4FeedForward · gate_up_proj", source: "nvidia/moe.py · HYV4FeedForward", sourceUrl: MOE_URL,
      code: "gate_up, _ = self.gate_up_proj(x)\nout = self.act_fn(gate_up)",
      weights: [
        { key: "model.layers.0.mlp.gate_proj.weight", shape: "[18432,6144]", dtype: "BF16", shard: SHARD, runtime: "gate_up_proj · gate", params: "113.25M" },
        { key: "model.layers.0.mlp.up_proj.weight", shape: "[18432,6144]", dtype: "BF16", shard: SHARD, runtime: "gate_up_proj · up", params: "113.25M" },
      ],
    },
    {
      id: "act", tone: "moe", kicker: "SILU · 无 CLAMP", title: "SwiGLU(dense 不截断)",
      summary: "silu(g)⊙u;与 routed experts 不同,这里没有 ±10 截断。",
      input: "g | u", inputShape: "2 × [T,18432]", output: "激活", outputShape: "[T,18432]",
      formula: "z=silu(g)⊙u", formulaNote: "SiluAndMul;packed 前半 gate 后半 up。",
      runtime: "SiluAndMul", source: "nvidia/moe.py · HYV4FeedForward.forward", sourceUrl: MOE_URL,
      code: "out = self.act_fn(gate_up)  # SiluAndMul",
      weights: [],
    },
    {
      id: "down", tone: "moe", kicker: "ROW PARALLEL · 18432→6144", title: "down 投影",
      summary: "投回隐藏宽度,输出 Y_ffn。",
      input: "激活", inputShape: "[T,18432]", output: "Y_ffn", outputShape: "[T,6144]",
      formula: "Y=z·W_downᵀ", formulaNote: "交回 hc_post。",
      runtime: "RowParallelLinear · down_proj", source: "nvidia/moe.py · HYV4FeedForward", sourceUrl: MOE_URL,
      code: "out, _ = self.down_proj(out)\nreturn out",
      weights: [{ key: "model.layers.0.mlp.down_proj.weight", shape: "[6144,18432]", dtype: "BF16", shard: SHARD, params: "113.25M" }],
    },
  ];
}

export function moeNodes(): Node[] {
  return [
    {
      id: "router", tone: "moe", kicker: "FP32 ROUTER · 256 LOGITS", title: "gate 路由打分",
      summary: "每 token 计算 256 个 FP32 logits;sigmoid 打分,expert_bias 只影响选择。",
      input: "Û", inputShape: "[T,6144]", output: "router_logits", outputShape: "[T,256]",
      formula: "r=Û·W_gᵀ", formulaNote: "GateLinear FP32;e_score_correction_bias 在 checkpoint 里叫 gate.e_score_correction_bias。",
      runtime: "GateLinear · fp32", source: "nvidia/moe.py · HYV4MoEFused", sourceUrl: MOE_URL,
      code: "router_logits, _ = self.gate(hidden_states)",
      weights: [
        { key: "model.layers.{l}.mlp.gate.weight", shape: "[256,6144]", dtype: "F32", shard: SHARD, params: "1.57M" },
        { key: "model.layers.{l}.mlp.gate.e_score_correction_bias", shape: "[256]", dtype: "F32", shard: SHARD, runtime: "expert_bias", params: "256" },
      ],
    },
    {
      id: "select", tone: "index", kicker: "TOP-8 · SIGMOID", title: "σ + Top-8 + renormalize",
      summary: "σ(r) 打分,按 s+bias 选 Top-8;混合权重用未加 bias 的分数归一化后 ×2.827。",
      input: "router_logits", inputShape: "[T,256]", output: "专家集合 + ŵ", outputShape: "8 / token",
      formula: "ŵ=2.827·s/Σs", formulaNote: "n_group=topk_group=1,等价朴素 Top-8;norm_topk_prob=true。",
      runtime: "FusedMoEFactory(sigmoid, renormalize)", source: "nvidia/moe.py · __init__", sourceUrl: MOE_URL,
      code: "FusedMoEFactory(num_experts=256, top_k=8, scoring_func='sigmoid',\n    renormalize=True, routed_scaling_factor=2.827,\n    e_score_correction_bias=self.expert_bias, swiglu_limit=10.0)",
      weights: [],
    },
    {
      id: "experts", tone: "moe", kicker: "256 ROUTED · CLAMPED SWIGLU", title: "routed experts(clamped)",
      summary: "Top-8 专家计算 clamped SwiGLU(gate 截 max=10,up 截 ±10),按 ŵ 加权归并。",
      input: "Û + ŵ + 专家 id", inputShape: "[T,6144]+8",
      output: "Y_routed", outputShape: "[T,6144]",
      formula: "Σŵ·W₂[clip(g)⊙σ(clip g)⊙clip(u)]", formulaNote: "专家是融合 all-experts 张量:gate_up [256,4096,6144] 运行时切 w1/w3。",
      runtime: "FusedMoE · experts.w13/w2", source: "nvidia/moe.py · forward", sourceUrl: MOE_URL,
      code: "final_hidden_states = self.experts(\n    hidden_states=hidden_states, router_logits=router_logits)",
      weights: [
        { key: "model.layers.{l}.mlp.experts.gate_up_proj", shape: "[256,4096,6144]", dtype: "BF16", shard: SHARD, runtime: "experts.w13 · w1|w3", params: "6.44B" },
        { key: "model.layers.{l}.mlp.experts.down_proj", shape: "[256,6144,2048]", dtype: "BF16", shard: SHARD, runtime: "experts.w2", params: "3.22B" },
      ],
    },
    {
      id: "shared", tone: "moe", kicker: "SHARED · 恒在支路", title: "1 shared expert(无 clamp)",
      summary: "所有 token 必经的共享专家,普通 SwiGLU,与单个 routed expert 同形。",
      input: "Û", inputShape: "[T,6144]", output: "Y_shared", outputShape: "[T,6144]",
      formula: "W₂[silu(g)⊙u]", formulaNote: "shared 不截断、不参与 Top-K。",
      runtime: "HYV4FeedForward · shared_experts", source: "nvidia/moe.py · shared_experts", sourceUrl: MOE_URL,
      code: "self.shared_experts = HYV4FeedForward(\n    intermediate_size=config.expert_hidden_dim * 1, reduce_results=False)",
      weights: [
        { key: "model.layers.{l}.mlp.shared_experts.gate_proj.weight", shape: "[2048,6144]", dtype: "BF16", shard: SHARD, runtime: "gate_up · gate", params: "12.58M" },
        { key: "model.layers.{l}.mlp.shared_experts.up_proj.weight", shape: "[2048,6144]", dtype: "BF16", shard: SHARD, runtime: "gate_up · up", params: "12.58M" },
        { key: "model.layers.{l}.mlp.shared_experts.down_proj.weight", shape: "[6144,2048]", dtype: "BF16", shard: SHARD, params: "12.58M" },
      ],
    },
    {
      id: "sum", tone: "output", kicker: "MOE OUTPUT", title: "routed ⊕ shared",
      summary: "FusedMoE 内部已按 ŵ 加权归并;这里只合并 routed 与 shared 两条支路。",
      input: "Y_routed + Y_shared", inputShape: "2 × [T,6144]", output: "Y_moe", outputShape: "[T,6144]",
      formula: "Y_moe=Y_routed+Y_shared", formulaNote: "交回 hc_post。",
      runtime: "MoE branch add", source: "nvidia/moe.py · forward", sourceUrl: MOE_URL,
      code: "final_hidden_states = self.experts(hidden_states, router_logits)\nreturn final_hidden_states.view(orig_shape)",
      weights: [],
    },
  ];
}
