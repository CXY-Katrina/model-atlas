"use client";

import { useMemo, useState } from "react";

type ModuleKey = "vit" | "gqa" | "msa" | "moe";

const moduleMeta: Record<ModuleKey, { eyebrow: string; title: string; description: string; accent: string }> = {
  vit: {
    eyebrow: "视觉入口",
    title: "ViT · 从像素到 576 个视觉 token",
    description: "Conv3D Patch Embed、32 层视觉编码、3D RoPE、投影与 2×2 Patch Merge。",
    accent: "#70e1f5",
  },
  gqa: {
    eyebrow: "Dense · Layer 0–2",
    title: "GQA · 64 个 Q 头共享 4 个 KV 头",
    description: "每 16 个 Query 头复用一组 Key / Value；前三层保持全量因果注意力。",
    accent: "#f6be5e",
  },
  msa: {
    eyebrow: "Sparse · Layer 3–59",
    title: "MSA · 先检索 block，再做稀疏 GQA",
    description: "Indexer 侧路选出 Top‑16 KV blocks，主注意力只读取被选中的 paged cache。",
    accent: "#b8f36b",
  },
  moe: {
    eyebrow: "Experts · Layer 3–59",
    title: "MoE · 128 个路由专家，Top‑4 激活",
    description: "Sigmoid router + correction bias，稀疏专家之外还有 1 个始终参与的共享专家。",
    accent: "#c7a7ff",
  },
};

const detailData: Record<ModuleKey, Array<{ label: string; meaning: string; shape: string; formula: string }>> = {
  vit: [
    { label: "01 · Patch Embed", meaning: "把视频时间片或复制后的静态图像按 2×14×14 切块；每块展平后由 Conv3D 映射到视觉隐藏空间。", shape: "[N, 3×2×14×14] → [N, 1280]", formula: "x₀ = Conv3D(pixel_patch), kernel = stride = (2, 14, 14)" },
    { label: "02 · 3D RoPE Attention", meaning: "16 头视觉自注意力。head_dim=80，其中 78 维均分给时间、高度、宽度旋转位置编码，剩余维度直通。", shape: "Q,K,V: [1, N, 16, 80]", formula: "A = softmax(RoPE₃ᴅ(Q) · RoPE₃ᴅ(K)ᵀ / √80) · V" },
    { label: "03 · Encoder ×32", meaning: "标准预归一化视觉 Transformer：LayerNorm → Attention → 残差 → LayerNorm → GELU MLP → 残差。", shape: "[N, 1280] → [N, 1280]", formula: "x′ = x + Attn(LN(x));  y = x′ + MLP(LN(x′))" },
    { label: "04 · Project + Merge", meaning: "先将视觉通道投影到文本 hidden_size=6144，再把相邻 2×2 token 拼接并压回 6144 维。", shape: "[2304,1280] → [2304,6144] → [576,6144]", formula: "z = W₂ GELU(W₁x);  y = MLP(concat₂×₂(z))" },
  ],
  gqa: [
    { label: "01 · QKV Projection", meaning: "64 个 Query 头，但只有 4 个 KV 头。每组 16 个 Query 头共享同一 K/V，显著缩小 KV cache。", shape: "Q [T,64,128] · K,V [T,4,128]", formula: "Q=XWq, K=XWk, V=XWv; group_size = 64 / 4 = 16" },
    { label: "02 · QK Norm + partial RoPE", meaning: "Q/K 逐 head 使用 Gemma RMSNorm；仅 head_dim 的前 50%（64 维）应用 NeoX 风格 RoPE。", shape: "[T,H,128] → [T,H,128]", formula: "q̂ = RMSNorm(q); q̃ = concat(RoPE(q̂[:64]), q̂[64:])" },
    { label: "03 · Causal GQA", meaning: "每个 Q 头连接到它所属的 KV 头，前三个 decoder layer 对全部历史 token 做因果注意力。", shape: "Scores [64,Tq,Tkv] → O [T,64,128]", formula: "Oₕ = softmax(qₕ k⌊h/16⌋ᵀ / √128 + mask) v⌊h/16⌋" },
    { label: "04 · Output + residual", meaning: "64 个头拼接为 8192 维，经 RowParallelLinear 投影回 6144；All‑Reduce 与下一次 RMSNorm 可融合。", shape: "[T,8192] → [T,6144]", formula: "Y = concat(O₀…O₆₃)Wo + residual" },
  ],
  msa: [
    { label: "01 · Fused 5-way Projection", meaning: "一次 GEMM 同时生成主分支 q/k/v 与检索分支 index_q/index_k；后者只用于算 block 相关性。", shape: "[T,6144] → q8192 | k512 | v512 | iq512 | ik512", formula: "[Q,K,V,Qᵢ,Kᵢ] = X · Wfused" },
    { label: "02 · Index Cache + Score", meaning: "Index K 写入独立的 key-only paged cache。Qᵢ 与历史 Kᵢ 计算分数，再把每个 128-token block 聚合为一个分数。", shape: "Qᵢ,Kᵢ [T,4,128] → block_scores [4,B]", formula: "score(h,b) = maxₜ∈block(b) (qᵢ,h · kᵢ,t,h / √128)" },
    { label: "03 · Top‑16 Block Select", meaning: "每个 index head 选 16 个 block；当前 local block 被强制保留。结果写入稳定地址的 topk_indices_buffer。", shape: "[token,4,B] → indices [token,4,16]", formula: "Sₕ = TopK₁₆(scoreₕ) ∪ {local block}" },
    { label: "04 · Sparse GQA", meaning: "主 Q 只从 Block Table 映射到的 16 个物理 KV page 读取数据；单头最多覆盖约 2048 个历史 token。", shape: "16 blocks × 128 tokens → O [T,64,128]", formula: "Oₕ = softmax(Qₕ Kᵀₛ / √128) Vₛ,  s ∈ selected pages" },
  ],
  moe: [
    { label: "01 · Sigmoid Router", meaning: "每个 token 产生 128 个专家 logits；sigmoid 分数加 correction bias 后用于选择，权重再归一化。", shape: "[T,6144] → router_logits [T,128]", formula: "s = sigmoid(XWg);  E = TopK₄(s + b);  ŵ = normalize(s[E])" },
    { label: "02 · All‑to‑All Dispatch", meaning: "token 按 Top‑4 专家去向分发到 EP rank。EP16 = DP4×TP4，因此这里仍是 16 个 worker，不是 256 个。", shape: "token assignments [T,4] → 16 EP ranks", formula: "rank(expert e) = ⌊e / 8⌋" },
    { label: "03 · Routed + shared experts", meaning: "每个 routed expert 是 6144→(gate/up 3072)→6144 的 SwiGLU‑OAI MLP；共享专家对所有 token 始终执行。", shape: "Top‑4 × [6144→3072→6144] + shared", formula: "Expert(x)=Wdown(clamp(g)·σ(1.702g)·clamp(u+1))" },
    { label: "04 · Combine", meaning: "各 expert 输出按路由权重乘加，乘 routed_scaling_factor=2，再加共享专家输出并回到原 DP/TP 布局。", shape: "4×[T,6144] → [T,6144]", formula: "y = 2·Σⱼ ŵⱼEⱼ(x) + Eshared(x)" },
  ],
};

const stageNodes: Record<ModuleKey, Array<{ name: string; shape: string }>> = {
  vit: [
    { name: "Pixels", shape: "672×672×3" }, { name: "Patchify", shape: "2304×1176" }, { name: "ViT ×32", shape: "2304×1280" }, { name: "Project", shape: "2304×6144" }, { name: "2×2 merge", shape: "576×6144" },
  ],
  gqa: [
    { name: "Hidden", shape: "T×6144" }, { name: "Q / K / V", shape: "64 / 4 / 4 heads" }, { name: "QK norm", shape: "per head" }, { name: "RoPE 50%", shape: "64 / 128 dims" }, { name: "Dense attend", shape: "T×6144" },
  ],
  msa: [
    { name: "Hidden", shape: "T×6144" }, { name: "5-way proj", shape: "main + index" }, { name: "Indexer", shape: "4×B scores" }, { name: "Top‑16", shape: "16×128 tokens" }, { name: "Sparse GQA", shape: "T×6144" },
  ],
  moe: [
    { name: "Hidden", shape: "T×6144" }, { name: "Router", shape: "T×128" }, { name: "Top‑4", shape: "T×4" }, { name: "EP All‑to‑All", shape: "16 ranks" }, { name: "Combine", shape: "T×6144" },
  ],
};

const topBlocks = [0, 3, 7, 11, 15, 18, 22, 27, 31, 36, 42, 47, 51, 56, 60, 63];
const physicalMap = Array.from({ length: 64 }, (_, i) => (i * 29 + 7) % 64);

function ModuleFlow({ active }: { active: ModuleKey }) {
  return (
    <div className="module-flow" aria-label={`${moduleMeta[active].title} 数据流`}>
      {stageNodes[active].map((node, index) => (
        <div className="flow-unit" key={node.name}>
          <div className="flow-node" style={{ "--delay": `${index * 90}ms` } as React.CSSProperties}>
            <span className="node-number">{String(index + 1).padStart(2, "0")}</span>
            <strong>{node.name}</strong>
            <code>{node.shape}</code>
          </div>
          {index < stageNodes[active].length - 1 && <span className="flow-arrow">→</span>}
        </div>
      ))}
    </div>
  );
}

function CacheLab() {
  const [sequence, setSequence] = useState(8192);
  const [selected, setSelected] = useState(63);
  const logicalBlocks = Math.ceil(sequence / 128);
  const visible = Math.min(logicalBlocks, 64);
  const selectedSet = new Set(topBlocks.filter((b) => b < visible));
  if (visible > 0) selectedSet.add(visible - 1);
  const physical = physicalMap[selected] ?? 0;
  const score = (0.48 + ((selected * 37) % 47) / 100).toFixed(2);

  return (
    <section className="cache-lab" id="cache">
      <div className="section-heading compact">
        <div>
          <span className="kicker">INTERACTIVE 02 · PAGED MEMORY</span>
          <h2>KV Cache 与 Block Table</h2>
        </div>
        <p>拖动上下文长度，再点任意逻辑 block。绿色是 Indexer 为当前 query 选中的 page。</p>
      </div>

      <div className="cache-controls">
        <label htmlFor="seq-range">上下文长度 <strong>{sequence.toLocaleString()} tokens</strong></label>
        <input id="seq-range" type="range" min="2048" max="8192" step="128" value={sequence} onChange={(e) => { setSequence(Number(e.target.value)); setSelected(Math.min(Number(e.target.value) / 128 - 1, 63)); }} />
        <div className="cache-metrics">
          <span><b>{logicalBlocks}</b> logical blocks</span>
          <span><b>128</b> tokens / block</span>
          <span><b>16</b> selected / head</span>
          <span><b>≤2048</b> KV tokens read</span>
        </div>
      </div>

      <div className="cache-layout">
        <div className="block-map">
          <div className="map-caption"><span>Logical block table</span><span className="legend"><i /> Top‑K selected</span></div>
          <div className="blocks">
            {Array.from({ length: visible }, (_, i) => (
              <button key={i} className={`${selectedSet.has(i) ? "hot" : ""} ${selected === i ? "focused" : ""}`} onClick={() => setSelected(i)} aria-label={`逻辑 block ${i}`}>
                <small>L{i}</small><b>P{physicalMap[i]}</b>
              </button>
            ))}
          </div>
          <div className="cache-path"><span>slot_mapping</span><div /><span>block_table</span><div /><span>physical KV pages</span></div>
        </div>

        <aside className="block-inspector">
          <span className="inspector-tag">SELECTED BLOCK</span>
          <div className="block-id">L{selected} <span>→</span> P{physical}</div>
          <dl>
            <div><dt>Token range</dt><dd>{selected * 128}–{selected * 128 + 127}</dd></div>
            <div><dt>Indexer score</dt><dd>{score}</dd></div>
            <div><dt>主 KV / TP rank</dt><dd>[128, 1, 2×128]</dd></div>
            <div><dt>Index key cache</dt><dd>[128, 1, 128]</dd></div>
          </dl>
          <p>{selectedSet.has(selected) ? "这个 page 会被主稀疏注意力 kernel 读取。" : "这个 page 仍在 cache 中，但本次 query 不读取。"}</p>
        </aside>
      </div>
    </section>
  );
}

function ParallelLab() {
  const [view, setView] = useState<"attention" | "moe" | "vit">("attention");
  const [rank, setRank] = useState(0);
  const rankInfo = useMemo(() => ({ dp: Math.floor(rank / 4), tp: rank % 4, expertStart: rank * 8 }), [rank]);
  return (
    <section className="parallel-lab" id="parallel">
      <div className="section-heading compact">
        <div><span className="kicker">INTERACTIVE 03 · 16 WORKERS</span><h2>DP4 × TP4 ⇒ EP16</h2></div>
        <p>同一组 16 个 worker，在 dense attention 中组成 4 个 DP 副本；进入 MoE 时重组为 1 个 EP16 通信域。</p>
      </div>
      <div className="view-tabs" role="tablist">
        <button className={view === "attention" ? "active" : ""} onClick={() => setView("attention")}>Attention · DP4 / TP4</button>
        <button className={view === "moe" ? "active" : ""} onClick={() => setView("moe")}>MoE · EP16</button>
        <button className={view === "vit" ? "active" : ""} onClick={() => setView("vit")}>ViT · encoder DP</button>
      </div>
      <div className="parallel-grid-wrap">
        <div className={`worker-grid mode-${view}`}>
          <span className="axis top">TP rank →</span>
          <span className="axis side">DP rank →</span>
          {Array.from({ length: 16 }, (_, i) => {
            const dp = Math.floor(i / 4), tp = i % 4;
            return (
              <button key={i} onClick={() => setRank(i)} className={rank === i ? "selected" : ""}>
                <span>GPU {String(i).padStart(2, "0")}</span>
                {view === "attention" && <><b>DP{dp} · TP{tp}</b><small>Q{tp * 16}–{tp * 16 + 15} · KV{tp}</small></>}
                {view === "moe" && <><b>EP rank {i}</b><small>E{i * 8}–{i * 8 + 7}</small></>}
                {view === "vit" && <><b>ViT replica</b><small>full 16 heads</small></>}
              </button>
            );
          })}
        </div>
        <aside className="rank-card">
          <span className="inspector-tag">GPU {String(rank).padStart(2, "0")}</span>
          {view === "attention" && <>
            <h3>DP{rankInfo.dp} / TP{rankInfo.tp}</h3>
            <p>持有 Q heads {rankInfo.tp * 16}–{rankInfo.tp * 16 + 15} 与 KV head {rankInfo.tp}。同一 DP 行共享请求批次，但各自只计算 1/4 attention heads。</p>
            <code>Q [T,16,128]<br />K,V [T,1,128]</code>
          </>}
          {view === "moe" && <>
            <h3>EP rank {rank}</h3>
            <p>常驻 routed experts {rankInfo.expertStart}–{rankInfo.expertStart + 7}。Router 通过 All‑to‑All 把命中的 token 发到这里，计算后再送回原 rank。</p>
            <code>128 experts ÷ 16 = 8 / rank</code>
          </>}
          {view === "vit" && <>
            <h3>Encoder DP worker</h3>
            <p>源码在 ViT DP 模式下设置 <code>disable_tp=true</code>，每个 worker 保留完整 16-head 视觉编码器，以多模态样本并行换吞吐。</p>
            <code>Q,K,V [1,N,16,80]</code>
          </>}
        </aside>
      </div>
      <div className="parallel-note"><strong>关键点</strong><span>EP16 = TP4 × DP4；EP 是 MoE 层对这 16 个 worker 的重新分组，不再额外乘一次。</span></div>
    </section>
  );
}

export default function Home() {
  const [active, setActive] = useState<ModuleKey>("msa");
  const [openDetail, setOpenDetail] = useState(0);
  const meta = moduleMeta[active];
  const selectModule = (key: ModuleKey) => { setActive(key); setOpenDetail(0); };

  return (
    <main style={{ "--module-accent": meta.accent } as React.CSSProperties}>
      <header className="topbar">
        <a className="brand" href="#top"><span>MM</span><b>MiniMax M3</b><small>ARCHITECTURE ATLAS</small></a>
        <nav><a href="#explorer">结构</a><a href="#cache">Cache</a><a href="#parallel">并行</a><a href="#sources">源码</a></nav>
        <div className="status"><i /> source-aligned</div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="kicker">MINIMAX M3 · INTERACTIVE MODEL EXPLAINER</span>
          <h1>看见模型<br /><em>真正流动起来</em></h1>
          <p>从 672×672 图像进入 ViT，到 64:4 GQA，再到 Indexer 驱动的 MSA 与 128 专家 MoE。点开每一层，看清公式、意义与 shape。</p>
          <a className="primary-cta" href="#explorer">进入模型 <span>↓</span></a>
        </div>
        <div className="hero-system" aria-label="MiniMax M3 系统摘要">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" />
          <div className="core"><span>hidden</span><b>6144</b><small>bf16</small></div>
          <div className="sat sat-a"><small>ViT</small><b>32L</b></div>
          <div className="sat sat-b"><small>Decoder</small><b>60L</b></div>
          <div className="sat sat-c"><small>Context</small><b>1M</b></div>
          <div className="sat sat-d"><small>Experts</small><b>128</b></div>
        </div>
      </section>

      <section className="stat-strip" aria-label="模型关键参数">
        <div><span>VISION</span><b>1280</b><small>16 heads · 32 layers</small></div>
        <div><span>GQA RATIO</span><b>64 : 4</b><small>16 query heads / KV</small></div>
        <div><span>SPARSE WINDOW</span><b>16 × 128</b><small>top blocks × tokens</small></div>
        <div><span>MOE</span><b>Top‑4 / 128</b><small>+ 1 shared expert</small></div>
      </section>

      <section className="explorer" id="explorer">
        <div className="section-heading">
          <div><span className="kicker">INTERACTIVE 01 · DATA FLOW</span><h2>逐层拆解</h2></div>
          <p>选择模块，再展开任意步骤。页面中的尺寸采用 MiniMax‑M3 发布配置；T 表示本轮 token 数，N 表示视觉 patch 数。</p>
        </div>
        <div className="module-tabs" role="tablist" aria-label="模型模块">
          {(Object.keys(moduleMeta) as ModuleKey[]).map((key, i) => (
            <button key={key} className={active === key ? "active" : ""} onClick={() => selectModule(key)} style={{ "--tab-accent": moduleMeta[key].accent } as React.CSSProperties}>
              <small>0{i + 1}</small><b>{key.toUpperCase()}</b><span>{moduleMeta[key].eyebrow}</span>
            </button>
          ))}
        </div>

        <article className="module-panel">
          <div className="module-intro"><span>{meta.eyebrow}</span><h3>{meta.title}</h3><p>{meta.description}</p></div>
          <ModuleFlow active={active} />
          <div className="detail-list">
            {detailData[active].map((item, index) => (
              <div className={`detail-row ${openDetail === index ? "open" : ""}`} key={item.label}>
                <button onClick={() => setOpenDetail(openDetail === index ? -1 : index)} aria-expanded={openDetail === index}>
                  <span>{item.label}</span><code>{item.shape}</code><i>{openDetail === index ? "−" : "+"}</i>
                </button>
                <div className="detail-body"><p>{item.meaning}</p><div><span>COMPUTE</span><code>{item.formula}</code></div></div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <CacheLab />
      <ParallelLab />

      <section className="layer-map">
        <div className="section-heading compact"><div><span className="kicker">MODEL MAP · 60 DECODER LAYERS</span><h2>Dense 开场，Sparse + MoE 主体</h2></div><p>配置中的 sparse_attention_freq 与 moe_layer_freq 都从第 4 层（index 3）开始启用。</p></div>
        <div className="layers">
          {Array.from({ length: 60 }, (_, i) => <div key={i} className={i < 3 ? "dense" : "sparse"} title={`Layer ${i}: ${i < 3 ? "Dense GQA + dense MLP" : "MSA + MoE"}`}><span>{i}</span></div>)}
        </div>
        <div className="layer-legend"><span><i className="dense" /> Layer 0–2 · Dense GQA + Dense MLP</span><span><i className="sparse" /> Layer 3–59 · MSA + Top‑4 MoE</span></div>
      </section>

      <footer id="sources">
        <div><span className="kicker">SOURCE NOTES</span><h2>结构来自代码，不是想象图</h2></div>
        <div className="source-links">
          <a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/minimax" target="_blank" rel="noreferrer"><span>01</span><b>Hugging Face Transformers</b><small>configuration · modeling · modular</small></a>
          <a href="https://github.com/vllm-project/vllm/tree/main/vllm/models/minimax_m3" target="_blank" rel="noreferrer"><span>02</span><b>vLLM MiniMax M3</b><small>vision · indexer · sparse attention · MSA</small></a>
          <a href="https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json" target="_blank" rel="noreferrer"><span>03</span><b>MiniMax‑M3 Config</b><small>dimensions · layer map · sparse config</small></a>
        </div>
        <p className="footnote">教学可视化 · shapes 为逻辑布局；具体 KV cache 物理布局会随 vLLM backend 与 dtype 改变。</p>
      </footer>
    </main>
  );
}
