"use client";

import { useState } from "react";

type Op = {
  name: string;
  impl: string;
  role: string;
  input: string;
  actualIn: string;
  output: string;
  actualOut: string;
  formula: string;
  note?: string;
  code?: string;
};

const visionOps: Op[] = [
  {
    name: "Patchify + Conv3D Patch Embedding",
    impl: "MiniMaxV3PatchEmbed.proj",
    role: "把图像或视频切成时空 patch，并一次投影到视觉隐藏空间。静态图像会在时间轴上复制为 2 帧。",
    input: "[Nᵥ, C·Pₜ·P²]",
    actualIn: "[2304, 3·2·14²] = [2304, 1176]（672² 示例）",
    output: "[Nᵥ, Hᵥ]",
    actualOut: "[2304, 1280]",
    formula: "Xᵥ = Conv3D(pixel_values), kernel = stride = (Pₜ, P, P)",
    note: "Pₜ=2，P=14。2304 来自 (672/14)²；不同动态分辨率会改变 Nᵥ。",
  },
  {
    name: "Pre LayerNorm",
    impl: "MiniMaxV3VisionTransformer.pre_layernorm",
    role: "稳定视觉编码器入口的通道尺度。",
    input: "[Nᵥ, Hᵥ]",
    actualIn: "[2304, 1280]",
    output: "[Nᵥ, Hᵥ]",
    actualOut: "[2304, 1280]",
    formula: "LN(x) = γ ⊙ (x-μ) / √(σ²+ε) + β",
  },
  {
    name: "QKV Projection",
    impl: "MiniMaxV3VisionAttention.qkv",
    role: "每个视觉层一次线性映射同时生成 16 组 Q/K/V。",
    input: "[Nᵥ, Hᵥ]",
    actualIn: "[2304, 1280]",
    output: "[1, Nᵥ, 3, Hᵥq, Dᵥ]",
    actualOut: "[1, 2304, 3, 16, 80]",
    formula: "[Q,K,V] = XWqkv + bqkv；Dᵥ = Hᵥ / Hᵥq",
  },
  {
    name: "3D partial RoPE",
    impl: "apply_rotary_pos_emb_vision",
    role: "把时间、高度、宽度坐标写入 Q/K。每个轴使用 26 维，共旋转 78 维，剩余 2 维直通。",
    input: "Q,K: [1, Nᵥ, Hᵥq, Dᵥ]",
    actualIn: "[1, 2304, 16, 80]",
    output: "Q̃,K̃: 同形状",
    actualOut: "[1, 2304, 16, 80]",
    formula: "Q̃ = concat(RoPEₜ(Q₀:₂₆), RoPEₕ(Q₂₆:₅₂), RoPE𝓌(Q₅₂:₇₈), Q₇₈:₈₀)",
  },
  {
    name: "Full Vision Self-Attention",
    impl: "MiniMaxV3VisionAttention.attn",
    role: "在同一图像/视频样本的全部视觉 patch 间做全注意力，cu_seqlens 隔离 batch 中不同媒体。",
    input: "Q̃,K̃,V: [1, Nᵥ, Hᵥq, Dᵥ]",
    actualIn: "3 × [1, 2304, 16, 80]",
    output: "[Nᵥ, Hᵥ]",
    actualOut: "[2304, 1280]",
    formula: "O = softmax(Q̃K̃ᵀ / √Dᵥ + Mmedia)V",
  },
  {
    name: "Encoder Layer ×32",
    impl: "MiniMaxV3EncoderLayer",
    role: "Pre-Norm Attention 与 GELU MLP 各自带残差；视觉宽度保持不变。",
    input: "xₗ: [Nᵥ, Hᵥ]",
    actualIn: "[2304, 1280]",
    output: "xₗ₊₁: [Nᵥ, Hᵥ]",
    actualOut: "[2304, 1280]",
    formula: "u=xₗ+Attn(LN₁(xₗ)); xₗ₊₁=u+W₂·GELU(W₁·LN₂(u))",
    code: "x = x + self.self_attn(self.layer_norm1(x), ...)\nreturn residual + self.mlp(self.layer_norm2(x))",
  },
  {
    name: "Project to Language Width",
    impl: "MiniMaxV3MultiModalProjector",
    role: "把视觉通道 1280 对齐到语言模型 hidden_size=6144。",
    input: "[Nᵥ, Hᵥ]",
    actualIn: "[2304, 1280]",
    output: "[Nᵥ, H]",
    actualOut: "[2304, 6144]",
    formula: "Z = W₂·GELU(W₁Xᵥ)",
  },
  {
    name: "2×2 Spatial Merge",
    impl: "MiniMaxV3PatchMerger",
    role: "每 2×2 相邻视觉 token 先拼接，再压回语言宽度，token 数缩小 4 倍。",
    input: "[Nᵥ/4, 4H]",
    actualIn: "[576, 24576]",
    output: "[Nᵥ/4, H]",
    actualOut: "[576, 6144]",
    formula: "Yᵥ = W₂·GELU(W₁·concat₂×₂(Z))",
    note: "672×672 示例最终产生 24×24=576 个视觉 token；576 不是所有分辨率的固定结果。",
  },
];

const denseOps: Op[] = [
  {
    name: "Gemma RMSNorm",
    impl: "input_layernorm",
    role: "对 decoder 输入做 Pre-Norm；vLLM 在后续层可把残差加法与下一次归一化融合。",
    input: "X: [B, S, H]",
    actualIn: "[B, S, 6144]",
    output: "X̂: [B, S, H]",
    actualOut: "[B, S, 6144]",
    formula: "X̂ = X / √(mean(X²)+ε) ⊙ (1+γ)",
  },
  {
    name: "Fused QKV Projection",
    impl: "QKVParallelLinear",
    role: "生成 64 个 Query 头与 4 组 Key/Value；每 16 个 Q 头共享一组 KV。",
    input: "[B, S, H]",
    actualIn: "[B, S, 6144]",
    output: "Q:[B,Hq,S,Dh]；K,V:[B,Hkv,S,Dh]",
    actualOut: "Q [B,64,S,128]；K,V [B,4,S,128]",
    formula: "Q=X̂Wq, K=X̂Wk, V=X̂Wv；group_size=Hq/Hkv=16",
  },
  {
    name: "Per-head QK Norm + partial RoPE",
    impl: "fused_qk_norm_rope_cache_insert",
    role: "Q/K 按 head 做 RMSNorm，仅前 Dr=64 维应用 NeoX RoPE；K/V 同时写入 paged KV cache。",
    input: "Q,K,V + position + slot_mapping",
    actualIn: "Q 64 heads；K/V 4 heads；Dh=128",
    output: "Q̃,K̃,V + cache",
    actualOut: "RoPE 64/128 dims；4-head KV cache",
    formula: "q̃=[RoPE(RMSNorm(q)₀:Dr), RMSNorm(q)Dr:Dh]",
  },
  {
    name: "Causal Dense GQA",
    impl: "Attention(..., use_mla=False)",
    role: "Layer 0–2 对全部可见历史做精确因果注意力；Query head h 使用 KV head ⌊h/16⌋。",
    input: "Q:[B,Hq,S,Dh]；K,V cache:[B,Hkv,T,Dh]",
    actualIn: "[B,64,S,128]；2×[B,4,T,128]",
    output: "O:[B,S,Hq·Dh]",
    actualOut: "[B,S,8192]",
    formula: "Oₕ=softmax(Q̃ₕK̃⌊h/16⌋ᵀ/√Dh + Mcausal)V⌊h/16⌋",
  },
  {
    name: "Output Projection + Residual",
    impl: "RowParallelLinear(o_proj)",
    role: "拼接所有 attention heads，投影回 6144，再与 decoder 残差相加。",
    input: "[B,S,Hq·Dh]",
    actualIn: "[B,S,8192]",
    output: "[B,S,H]",
    actualOut: "[B,S,6144]",
    formula: "Y = concat(O₀…O₆₃)Wo + X",
  },
  {
    name: "Dense SwiGLU-OAI MLP",
    impl: "MiniMaxM3MLP",
    role: "仅 Layer 0–2 使用 dense FFN；gate/up 合并投影，激活采用带 clamp 的 SwiGLU-OAI。",
    input: "[B,S,H]",
    actualIn: "[B,S,6144]",
    output: "[B,S,H]",
    actualOut: "6144 → 2×12288 → 12288 → 6144",
    formula: "MLP(x)=Wdown[clamp(g)·σ(1.702·clamp(g))·clamp(u+1)]",
  },
];

const msaOps: Op[] = [
  {
    name: "Gemma RMSNorm",
    impl: "input_layernorm",
    role: "Layer 3–59 的稀疏注意力入口归一化。",
    input: "X:[B,S,H]",
    actualIn: "[B,S,6144]",
    output: "X̂:[B,S,H]",
    actualOut: "[B,S,6144]",
    formula: "X̂ = RMSNorm(X)",
  },
  {
    name: "Fused 5-way Projection",
    impl: "MiniMaxM3MSAAttention.qkv_proj",
    role: "一次 GEMM 同时产生主注意力 Q/K/V 与索引侧路 Qidx/Kidx。索引侧路只检索，不生成内容。",
    input: "[B,S,H]",
    actualIn: "[B,S,6144]",
    output: "Q|K|V|Qidx|Kidx",
    actualOut: "8192 | 512 | 512 | 512 | 128 channels",
    formula: "[Q,K,V,Qidx,Kidx] = X̂·Wfused",
  },
  {
    name: "Main QK Norm + partial RoPE",
    impl: "fused_qk_norm_rope_cache_insert",
    role: "主分支规范化 Q/K、应用 50% RoPE，并按 slot_mapping 写入主 KV cache。",
    input: "Q:[B,64,S,128]；K,V:[B,4,S,128]",
    actualIn: "RoPE Dr=64，θ=5,000,000",
    output: "Q̃ + paged K/V cache",
    actualOut: "每 token 4×128 的 K 与 V",
    formula: "cache[physical_slot] ← (K̃,V)",
  },
  {
    name: "Index QK Norm + Index-K Cache",
    impl: "MiniMaxM3Indexer",
    role: "4 个 GQA group 各自拥有 Qidx；Kidx 在各组之间共享，并写入单独的 key-only paged cache。",
    input: "Qidx:[B,S,Hi,Di]；Kidx:[B,S,1,Di]",
    actualIn: "[B,S,4,128]；[B,S,1,128]",
    output: "normalized Qidx + index key cache",
    actualOut: "每物理 slot 仅 1×128 key",
    formula: "sᵢⱼʳ = ⟨RMS(Qidxᵢʳ), RMS(Kidxⱼ)⟩ / √Di",
  },
  {
    name: "Causal Token Score → Block Max",
    impl: "indexer_paged_forward",
    role: "先屏蔽未来 token，再把每 128 个 token 的分数做 max 聚合，得到每个 index head 的 block relevance。",
    input: "token score:[B,Hi,S,T]",
    actualIn: "Hi=4；Bk=128；j≤i 才有效",
    output: "block score:[B,Hi,S,⌈T/Bk⌉]",
    actualOut: "2048 tokens → [B,4,S,16]",
    formula: "Mᵢ,bʳ = maxⱼ∈block(b), j≤i sᵢⱼʳ；未来位置 = −∞",
  },
  {
    name: "Priority-aware Top-K Blocks",
    impl: "index_topk",
    role: "每个 index head 选 k=16 个逻辑 block。当前/local block 强制保留；发布配置没有首块强制。",
    input: "[B,Hi,S,Nb]",
    actualIn: "Hi=4，k=16，local=1，init=0",
    output: "topk_indices_buffer:[B,S,Hi,k]",
    actualOut: "[B,S,4,16]，不足 16 的槽位为 -1",
    formula: "score(local)=10²⁹；score(init)=10³⁰，但 sparse_init_block=0",
    code: "score = where(is_local, 1e29, where(is_init, 1e30, score))",
    note: "当可见 block 数 ≤16 时，real_topk=min(16, visible_blocks)，因此全部可见 block 都进入结果。",
  },
  {
    name: "Block Table Address Translation",
    impl: "paged KV cache metadata",
    role: "Top-K 给出逻辑 block id；block_table 再映射到物理 KV page。这样请求可使用非连续显存。",
    input: "logical block ids + block_table",
    actualIn: "16 logical ids / index head",
    output: "physical page addresses",
    actualOut: "16 pages × 128 slots",
    formula: "physical_page = block_table[request, logical_block]",
  },
  {
    name: "Exact Block-sparse GQA",
    impl: "MiniMaxM3SparseAttention",
    role: "主分支只读取被索引选中的 KV pages；在选中块内部仍做精确 token attention，并再次施加因果与 padding mask。",
    input: "Q + selected K/V pages",
    actualIn: "Q:[B,64,S,128]；≤16×128 KV / head",
    output: "O:[B,S,Hq·Dh]",
    actualOut: "[B,S,8192]",
    formula: "Oₕ=softmax(QₕK𝒮ᵀ/√Dh + Mcausal + Mpad)V𝒮",
    code: "self.indexer(index_query)\nreturn self.impl.forward(..., topk_indices_buffer, ...)" ,
  },
  {
    name: "Output Projection + Residual",
    impl: "RowParallelLinear(o_proj)",
    role: "把 64 个稀疏 attention head 合并回语言隐藏宽度，并进入 post-attention RMSNorm。",
    input: "[B,S,Hq·Dh]",
    actualIn: "[B,S,8192]",
    output: "[B,S,H]",
    actualOut: "[B,S,6144]",
    formula: "Y = Wo·concat(O₀…O₆₃) + X",
  },
];

const moeOps: Op[] = [
  {
    name: "Router Logits",
    impl: "GateLinear",
    role: "每个 token 计算到 128 个 routed experts 的 FP32 路由 logits。",
    input: "[B,S,H]",
    actualIn: "[B,S,6144]",
    output: "[B,S,E]",
    actualOut: "[B,S,128]",
    formula: "r = XWrouter",
  },
  {
    name: "Sigmoid + Correction Bias + Top-4",
    impl: "FusedMoE routing method",
    role: "sigmoid 分数加 correction bias 后决定专家；被选专家的原始 sigmoid 权重重新归一化。",
    input: "r:[B,S,E]",
    actualIn: "E=128",
    output: "expert_ids, weights:[B,S,K]",
    actualOut: "2×[B,S,4]",
    formula: "𝓔=TopK₄(σ(r)+b)；ŵₑ=σ(rₑ)/Σⱼ∈𝓔σ(rⱼ)",
  },
  {
    name: "Routed Expert SwiGLU-OAI",
    impl: "FusedMoE experts",
    role: "每个 token 只执行命中的 4 个 routed expert；各 expert 的中间宽度为 3072。",
    input: "4 × [tokensₑ,H]",
    actualIn: "4 × [tokensₑ,6144]",
    output: "4 × [tokensₑ,H]",
    actualOut: "6144 → 2×3072 → 3072 → 6144",
    formula: "Eₑ(x)=Wdownₑ[SwiGLU-OAI(Wgate/upₑx)]",
  },
  {
    name: "Shared Expert",
    impl: "shared_experts",
    role: "1 个共享专家对所有 token 始终执行，用来承载通用能力，不参与 Top-K 竞争。",
    input: "[B,S,H]",
    actualIn: "[B,S,6144]",
    output: "[B,S,H]",
    actualOut: "6144 → 2×3072 → 3072 → 6144",
    formula: "Yshared = Eshared(X)",
  },
  {
    name: "Weighted Combine + Residual",
    impl: "FusedMoE output",
    role: "将 4 个 routed expert 输出按权重合并、乘 routed scaling=2，再加共享专家与 decoder 残差。",
    input: "4 routed outputs + shared output",
    actualIn: "5 × [B,S,6144]（逻辑形状）",
    output: "[B,S,H]",
    actualOut: "[B,S,6144]",
    formula: "Y = X + 2·Σₑ∈𝓔 ŵₑEₑ(X) + Eshared(X)",
  },
];

const sections = [
  ["overall", "总 Forward"], ["config", "配置与符号"], ["vit", "ViT 视觉塔"], ["gqa", "Dense GQA"],
  ["msa", "MSA 稀疏注意力"], ["cache", "Cache / Mask"], ["moe", "MoE FFN"], ["sources", "依据与边界"],
];

function Shape({ symbolic, actual }: { symbolic: string; actual: string }) {
  return <div className="shape"><div><span>符号形状</span><code>{symbolic}</code></div><div><span>MiniMax-M3 实值</span><code>{actual}</code></div></div>;
}

function OperatorChain({ ops, tone }: { ops: Op[]; tone: string }) {
  return <div className={`op-chain ${tone}`}>
    {ops.map((op, i) => <div className="op-wrap" key={op.name}>
      <article className="op-card">
        <div className="op-index">{String(i + 1).padStart(2, "0")}</div>
        <div className="op-main">
          <div className="op-title"><div><span>{op.impl}</span><h3>{op.name}</h3></div><p>{op.role}</p></div>
          <div className="io-grid"><Shape symbolic={op.input} actual={op.actualIn}/><div className="io-arrow">↓</div><Shape symbolic={op.output} actual={op.actualOut}/></div>
          <details>
            <summary>展开运算细节、公式{op.code ? "与代码" : ""}</summary>
            <div className="compute"><div><span>COMPUTE</span><code>{op.formula}</code>{op.note && <p>{op.note}</p>}</div>{op.code && <pre><code>{op.code}</code></pre>}</div>
          </details>
        </div>
      </article>
      {i < ops.length - 1 && <div className="chain-arrow"><span>↓</span></div>}
    </div>)}
  </div>;
}

function SectionHeader({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: React.ReactNode }) {
  return <header className="section-head" id={id}><span>{eyebrow}</span><h2>{title}</h2><p>{children}</p></header>;
}

function CacheMaskLab() {
  const [scenario, setScenario] = useState<"short" | "long">("short");
  const count = scenario === "short" ? 16 : 32;
  const selectedLong = new Set([2,3,5,7,8,11,13,15,17,19,21,23,26,28,29,31]);
  return <section className="report-section" aria-labelledby="cache-title">
    <SectionHeader id="cache" eyebrow="MEMORY & MASK" title="KV Cache、Block Table 与真实选择规则">
      删除随机“点选 block”演示，改为与当前 vLLM 实现一致的确定性推演。索引选的是逻辑 block，Block Table 负责把它翻译为物理 KV page。
    </SectionHeader>
    <div className="scenario-tabs" role="tablist" aria-label="上下文长度场景">
      <button className={scenario === "short" ? "active" : ""} onClick={() => setScenario("short")}>2048 tokens · 全选</button>
      <button className={scenario === "long" ? "active" : ""} onClick={() => setScenario("long")}>4096 tokens · Top-16</button>
    </div>
    <div className="cache-card">
      <div className="cache-summary">
        <div><span>可见 KV 长度 T</span><b>{scenario === "short" ? "2048" : "4096"}</b></div>
        <div><span>可见 blocks ⌈T/Bk⌉</span><b>{count}</b></div>
        <div><span>real_topk=min(k, visible)</span><b>16</b></div>
        <div><span>每 block</span><b>128 tokens</b></div>
      </div>
      <div className="block-explain">
        {Array.from({length: count}, (_, i) => {
          const selected = scenario === "short" || selectedLong.has(i);
          const local = i === count - 1;
          return <div key={i} className={`block ${selected ? "selected" : ""} ${local ? "local" : ""}`}><small>L{i}</small><b>{local ? "LOCAL" : selected ? (scenario === "short" ? "VISIBLE" : "TOP-K") : "SKIP"}</b></div>
        })}
      </div>
      <p className="cache-verdict">{scenario === "short" ? "结论：2048 / 128 = 16，恰好等于 k=16，所以 16 个可见 block 全部被选择；这里不存在“只绿一部分”的合理结果。" : "结论：32 个可见 block 中选择 16 个。L31 是当前 local block，固定优先级 10²⁹；其余 15 个由 Indexer 分数决定。图中 Top-K 仅用于解释数量，不代表某次真实请求的模型得分。"}</p>
    </div>
    <div className="mask-grid">
      <article><span>01 · Indexer token mask</span><h3>先因果，再做 block max</h3><code>j ≤ i → score(qᵢ,kⱼ)<br/>j &gt; i → −∞</code><p>query 所在 block 内未来 token 不得参与 max，避免 Indexer 偷看未来。</p></article>
      <article><span>02 · Block priority</span><h3>尾块强制，首块不强制</h3><code>local block → 10²⁹<br/>init block → 10³⁰（数量为 0）</code><p><b>sparse_local_block=1</b>；<b>sparse_init_block=0</b>。因此首块只是普通候选，不具备固定 sink 优先级。</p></article>
      <article><span>03 · Main attention mask</span><h3>读到 page 后仍要 mask</h3><code>future token → −∞<br/>padding slot → −∞</code><p>选中一个 block 不等于其中 128 个位置都可见；尾块内部仍按 query 绝对位置遮住未来 token。</p></article>
    </div>
    <div className="translation"><div><span>Indexer 输出</span><code>logical ids [L₂, L₇, …, L₃₁]</code></div><i>→</i><div><span>Block Table</span><code>request × logical_block</code></div><i>→</i><div><span>物理 KV pages</span><code>[P₄₁, P₃, …, P₁₈]</code></div></div>
  </section>;
}

export default function Home() {
  return <main>
    <aside className="sidebar">
      <a className="wordmark" href="#overall"><span>MM</span><div><b>MiniMax-M3</b><small>MODEL TRACE</small></div></a>
      <nav>{sections.map(([id,label], i) => <a href={`#${id}`} key={id}><span>{String(i+1).padStart(2,"0")}</span>{label}</a>)}</nav>
      <div className="verified"><i/> config + source aligned</div>
    </aside>

    <div className="content">
      <section className="report-section first">
        <SectionHeader id="overall" eyebrow="FORWARD TRACE · AUG 2026" title="MiniMax-M3 模型逐算子分析">
          从输入到 logits 的实现链路。所有 shape 先写配置变量，再写发布配置实值；默认只展示结构，点击每个算子可展开计算公式与关键代码。
        </SectionHeader>
        <div className="model-facts"><div><b>60</b><span>decoder layers</span></div><div><b>1M</b><span>context</span></div><div><b>64:4</b><span>Q : KV heads</span></div><div><b>3 + 57</b><span>Dense + MSA/MoE</span></div></div>
        <div className="overview-chain">
          {[
            ["Input processor", "text ids + image/video grids"],
            ["Vision tower（可选）", "pixels → visual embeddings [Nᵥ/4, H]"],
            ["Embedding integration", "视觉向量替换占位 token embedding"],
            ["embed_tokens", "input_ids → X₀ [B,S,H]"],
            ["Decoder Layer 0–2", "Dense GQA + Dense SwiGLU"],
            ["Decoder Layer 3–59", "MSA + Top-4 MoE + shared expert"],
            ["final RMSNorm", "X₆₀ → normalized hidden states"],
            ["lm_head", "[B,S,H] → logits [B,S,V]"],
          ].map(([a,b],i) => <div className="overview-step" key={a}><span>{String(i+1).padStart(2,"0")}</span><div><b>{a}</b><code>{b}</code></div>{i<7&&<i>↓</i>}</div>)}
        </div>
        <div className="layer-strip">{Array.from({length:60},(_,i)=><div className={i<3?"dense":"sparse"} key={i} title={`Layer ${i}`}><span>{i}</span></div>)}</div>
        <div className="legend"><span><i className="dense"/>L0–2 Dense GQA + Dense MLP</span><span><i className="sparse"/>L3–59 MSA + MoE</span></div>
      </section>

      <section className="report-section">
        <SectionHeader id="config" eyebrow="CONFIGURATION" title="符号表与发布配置实值">
          页面统一使用 B/S/T 区分 batch、本轮 query 长度与 KV 总长度；避免把 prefill 和 decode 都含糊写成一个 T。
        </SectionHeader>
        <div className="config-grid">
          {[
            ["B", "batch / sequence 数", "运行时"], ["S", "本轮 query token 数", "prefill&gt;1；decode=1"], ["T", "含 cache 的 KV 总长度", "≤1,048,576"],
            ["H", "text hidden_size", "6144"], ["Hq / Hkv", "query / KV heads", "64 / 4"], ["Dh", "attention head_dim", "128"],
            ["Dr", "rotary_dim", "64（50%）"], ["Hi / Di", "index heads / dim", "4 / 128"], ["Bk / k", "block size / top blocks", "128 / 16"],
            ["E / K", "routed experts / active", "128 / 4"], ["I / Id", "MoE / dense FFN width", "3072 / 12288"], ["V", "vocab_size", "200,064"],
            ["Hᵥ / Hᵥq / Dᵥ", "vision width / heads / dim", "1280 / 16 / 80"], ["Lᵥ", "vision layers", "32"], ["Pₜ / P / M", "temporal patch / patch / merge", "2 / 14 / 2"],
          ].map(([s,m,v])=><div key={s}><code>{s}</code><span>{m}</span><b>{v}</b></div>)}
        </div>
      </section>

      <section className="report-section">
        <SectionHeader id="vit" eyebrow="VISION TOWER" title="ViT：pixels → language-width visual tokens">32 层视觉 Transformer、3D partial RoPE、语言宽度投影和 2×2 token merge。以下用 672×672 单图展示实值链路。</SectionHeader>
        <OperatorChain ops={visionOps} tone="vision"/>
      </section>

      <section className="report-section">
        <SectionHeader id="gqa" eyebrow="DECODER · LAYER 0–2" title="Dense GQA：完整历史的因果注意力">前三层是全量注意力与 dense MLP。GQA 比例 64:4，使 16 个 Query heads 共享一个 KV head。</SectionHeader>
        <OperatorChain ops={denseOps} tone="dense"/>
      </section>

      <section className="report-section">
        <SectionHeader id="msa" eyebrow="DECODER · LAYER 3–59" title="MSA：索引侧路先找 block，主分支再精算">MSA 不是窗口注意力。Indexer 为每个 GQA group 动态检索相关逻辑块，主分支只在所选 paged KV cache 上执行精确 GQA。</SectionHeader>
        <div className="branch-note"><div><b>Main branch</b><span>Q / K / V → exact sparse GQA → semantic output</span></div><div><b>Index branch</b><span>Qidx / shared Kidx → causal block score → Top-16 ids</span></div></div>
        <OperatorChain ops={msaOps} tone="sparse"/>
      </section>

      <CacheMaskLab/>

      <section className="report-section">
        <SectionHeader id="moe" eyebrow="DECODER FFN · LAYER 3–59" title="MoE：128 选 4，再加 1 个共享专家">这里聚焦算子语义，不展示部署并行拓扑。路由决策与专家计算均按公开配置和 vLLM 实现描述。</SectionHeader>
        <OperatorChain ops={moeOps} tone="moe"/>
      </section>

      <footer className="report-section" id="sources">
        <span className="eyebrow">PROVENANCE</span><h2>分析依据与实现边界</h2>
        <p>结构参数以发布配置为准，运行链路以 Transformers / vLLM 当前实现交叉核验。技术报告中的算法定义用于解释 MSA；报告实验数字不与线上 M3 博客的产品指标混用。</p>
        <div className="sources">
          <a href="https://www.minimaxi.com/blog/minimax-m3" target="_blank" rel="noreferrer"><b>官方 MiniMax-M3 博客</b><span>模型定位、1M 上下文、线上加速</span></a>
          <a href="https://arxiv.org/html/2606.13392" target="_blank" rel="noreferrer"><b>MiniMax Sparse Attention 技术报告</b><span>Indexer、block score、复杂度与 kernel</span></a>
          <a href="https://github.com/MiniMax-AI/MiniMax-M3/" target="_blank" rel="noreferrer"><b>MiniMax-AI/MiniMax-M3</b><span>官方模型仓库与用法</span></a>
          <a href="https://github.com/vllm-project/vllm/tree/main/vllm/models/minimax_m3" target="_blank" rel="noreferrer"><b>vLLM MiniMax-M3</b><span>model、vision、indexer、sparse attention</span></a>
          <a href="https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json" target="_blank" rel="noreferrer"><b>发布配置 config.json</b><span>维度、层型、稀疏与 MoE 参数</span></a>
          <a href="https://gitcode.com/gcw_hasgjVbP/model_analysis/blob/main/MiniMax-M3-analysis-report.html" target="_blank" rel="noreferrer"><b>原分析报告</b><span>章节结构与 Ascend 分析方法参考</span></a>
        </div>
        <small className="disclaimer">Shape 为逻辑布局；实际张量会因 batch packing、tensor layout、backend 与 dtype 改写。2048/4096 block 图只解释选择与 mask 规则，不伪造模型相关性分数。</small>
      </footer>
    </div>
  </main>;
}
