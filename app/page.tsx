"use client";

import { useState } from "react";
import {
  denseNodes,
  layerShard,
  sparseNodes,
  visionNodes,
  type Node,
  type Weight,
} from "./model-data";

type Tab = "io" | "formula" | "code" | "weights";

const TEXT_NODE: Node = {
  id: "token-embedding",
  tone: "projection",
  kicker: "TEXT INPUT · VOCAB 200064",
  title: "Token Embedding",
  summary: "文本 token id 查表得到语言宽度 6144 的向量；视觉 token 也会在融合后对齐到这一宽度。",
  input: "input_ids",
  inputShape: "[B,S]",
  output: "text embeddings",
  outputShape: "[B,S,6144]",
  formula: "X₀[i] = Wembed[input_ids[i]]",
  formulaNote: "embedding 与 lm_head 均覆盖 200,064 个 vocabulary entries。",
  runtime: "VocabParallelEmbedding · embed_tokens",
  source: "nvidia/model.py · MiniMaxM3Model",
  sourceUrl: "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/nvidia/model.py",
  code: "inputs_embeds = self.embed_tokens(input_ids)",
  weights: [{ key: "language_model.model.embed_tokens.weight", shape: "[200064,6144]", dtype: "BF16", shard: "model-00001-of-00059.safetensors", params: "1.229B" }],
};

const OUTPUT_NODE: Node = {
  id: "language-output",
  tone: "output",
  kicker: "LANGUAGE OUTPUT",
  title: "RMSNorm → LM Head",
  summary: "第 60 层输出先做最终 RMSNorm，再投影到完整词表得到下一 token logits。",
  input: "X₆₀",
  inputShape: "[B,S,6144]",
  output: "logits",
  outputShape: "[B,S,200064]",
  formula: "logits = RMSNorm(X₆₀) · Wlmᵀ",
  formulaNote: "推理时通常只保留需要采样位置的 logits。",
  runtime: "MiniMAXGemmaRMSNorm → ParallelLMHead",
  source: "nvidia/model.py · MiniMaxM3ForCausalLM",
  sourceUrl: "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/nvidia/model.py",
  code: "hidden_states = self.model(input_ids, positions, ... )\nlogits = self.compute_logits(hidden_states)",
  weights: [
    { key: "language_model.model.norm.weight", shape: "[6144]", dtype: "BF16", shard: "model-00059-of-00059.safetensors", params: "6,144" },
    { key: "language_model.lm_head.weight", shape: "[200064,6144]", dtype: "BF16", shard: "model-00001-of-00059.safetensors", params: "1.229B" },
  ],
};

function ModuleNode({ node, active, compact = false, onHover, onLeave, onSelect }: {
  node: Node;
  active: boolean;
  compact?: boolean;
  onHover: (node: Node) => void;
  onLeave: () => void;
  onSelect: (node: Node) => void;
}) {
  return (
    <button
      className={`module-node tone-${node.tone} ${compact ? "compact" : ""} ${active ? "active" : ""}`}
      onMouseEnter={() => onHover(node)}
      onMouseLeave={onLeave}
      onFocus={() => onHover(node)}
      onBlur={onLeave}
      onClick={() => onSelect(node)}
      aria-label={`${node.title}：${node.inputShape} 到 ${node.outputShape}`}
    >
      <span>{node.kicker}</span>
      <b>{node.title}</b>
      <code>{node.outputShape}</code>
    </button>
  );
}

function DetailPanel({ node, pinned, tab, setTab }: { node: Node; pinned: boolean; tab: Tab; setTab: (tab: Tab) => void }) {
  const tabs: [Tab, string][] = [["io", "I/O"], ["formula", "公式"], ["code", "代码"], ["weights", "权重"]];
  return (
    <aside className="detail-panel" aria-live="polite">
      <header className="detail-header">
        <div><span>{pinned ? "CLICK PINNED" : "HOVER PREVIEW"}</span><h2>{node.title}</h2></div>
        <i className={`tone-dot tone-${node.tone}`} />
        <p>{node.summary}</p>
        <code>{node.runtime}</code>
      </header>
      <div className="detail-tabs" role="tablist">
        {tabs.map(([id, label]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>)}
      </div>
      <div className="detail-content">
        {tab === "io" && <div className="shape-view"><article><span>INPUT</span><b>{node.input}</b><code>{node.inputShape}</code></article><i>→</i><article><span>OUTPUT</span><b>{node.output}</b><code>{node.outputShape}</code></article></div>}
        {tab === "formula" && <div className="formula-view"><span>COMPUTE</span><code>{node.formula}</code><p>{node.formulaNote}</p></div>}
        {tab === "code" && <div className="code-view"><a href={node.sourceUrl} target="_blank" rel="noreferrer"><span>SOURCE</span><b>{node.source}</b><i>↗</i></a><pre><code>{node.code}</code></pre></div>}
        {tab === "weights" && <WeightView weights={node.weights} />}
      </div>
      <footer><i /> hover 预览 · click 固定 · shape / code / weight 同源核对</footer>
    </aside>
  );
}

function WeightView({ weights }: { weights: Weight[] }) {
  if (!weights.length) return <div className="empty-weight"><b>没有可训练权重</b><p>这是路由、缓存、mask 或张量合并操作。</p></div>;
  return <div className="weight-view">{weights.map(weight => <article key={weight.key}>
    <code className="weight-key">{weight.key}</code>
    <div><b>{weight.dtype}</b><code>{weight.shape}</code>{weight.params && <span>{weight.params}</span>}{weight.note && <em>{weight.note}</em>}</div>
    <small>{weight.shard}</small>{weight.runtime && <small>→ {weight.runtime}</small>}
  </article>)}</div>;
}

export default function Home() {
  const [layer, setLayer] = useState(3);
  const [pinnedNode, setPinnedNode] = useState<Node>(sparseNodes(3)[1]);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);
  const [tab, setTab] = useState<Tab>("io");
  const [dark, setDark] = useState(false);
  const denseLayer = layer < 3 ? layer : 0;
  const sparseLayer = layer >= 3 ? layer : 3;
  const dense = denseNodes(denseLayer);
  const sparse = sparseNodes(sparseLayer);
  const activeNode = hoveredNode ?? pinnedNode;
  const activeId = activeNode.id;
  const selectNode = (node: Node) => { setPinnedNode(node); setHoveredNode(null); };
  const selectLayer = (next: number) => {
    setLayer(next);
    const nextNode = next < 3 ? denseNodes(next)[1] : sparseNodes(next)[1];
    setPinnedNode(nextNode);
    setHoveredNode(null);
  };
  const nodeProps = { active: false, onHover: setHoveredNode, onLeave: () => setHoveredNode(null), onSelect: selectNode };

  return <main className={`atlas-app ${dark ? "dark" : ""}`}>
    <header className="app-header">
      <div className="brand-lockup"><span className="brand-glyph"><i /><i /><i /></span><div><b>MiniMax-M3</b><small>ARCHITECTURE × CODE × WEIGHTS</small></div></div>
      <div className="headline"><b>一屏看完整结构</b><span>hover 预览 · click 固定</span></div>
      <div className="model-facts"><span><b>428B</b> total</span><span><b>23B</b> active</span><span><b>1M</b> context</span><span><b>59</b> shards</span></div>
      <button className="theme-button" onClick={() => setDark(value => !value)} aria-label="切换明暗主题">{dark ? "☀" : "☾"}</button>
    </header>

    <div className="screen-grid">
      <section className="map-panel">
        <header className="map-header">
          <div><span>COMPLETE FORWARD MAP</span><h1>输入、60 层主干与输出，一次看完</h1></div>
          <div className="current-layer"><span>INSPECTING</span><b>L{layer}</b><code>{layerShard(layer)}</code></div>
        </header>

        <div className="layer-strip" aria-label="选择 Decoder 层">
          <div className="strip-label"><span>DECODER</span><b>60 layers</b></div>
          <div className="layer-cells">{Array.from({ length: 60 }, (_, index) => <button key={index} className={`${index < 3 ? "dense" : "sparse"} ${index === layer ? "active" : ""}`} onClick={() => selectLayer(index)} aria-label={`选择第 ${index} 层`}><span>{index}</span></button>)}</div>
          <div className="strip-legend"><span><i className="dense" /> Dense ×3</span><span><i className="sparse" /> MSA + MoE ×57</span></div>
        </div>

        <div className="architecture-map">
          <section className="stage inputs-stage">
            <header><span>01 · INPUTS</span><b>双模态入口</b></header>
            <ModuleNode node={TEXT_NODE} {...nodeProps} active={activeId === TEXT_NODE.id} />
            <div className="mini-flow">{visionNodes.slice(0, 4).map((node, index) => <div key={node.id}><ModuleNode compact node={node} {...nodeProps} active={activeId === node.id} />{index < 3 && <i>↓</i>}</div>)}</div>
          </section>
          <div className="stage-link"><span>→</span><code>H=6144</code></div>

          <section className="stage fusion-stage">
            <header><span>02 · FUSION</span><b>统一序列</b></header>
            <ModuleNode node={visionNodes[4]} {...nodeProps} active={activeId === visionNodes[4].id} />
            <div className="fusion-note"><b>[B,S,6144]</b><span>后续不再区分模态</span></div>
          </section>
          <div className="stage-link"><span>→</span><code>X₀</code></div>

          <section className="stage dense-stage">
            <header><span>03 · L{denseLayer} / DENSE ×3</span><b>完整历史注意力</b></header>
            <div className="vertical-flow">{dense.map((node, index) => <div key={node.id}><ModuleNode compact node={node} {...nodeProps} active={activeId === node.id} />{index < dense.length - 1 && <i>↓</i>}</div>)}</div>
          </section>
          <div className="stage-link"><span>→</span><code>X₃</code></div>

          <section className="stage sparse-stage">
            <header><span>04 · L{sparseLayer} / SPARSE ×57</span><b>MSA + Top-4 MoE</b></header>
            <ModuleNode compact node={sparse[0]} {...nodeProps} active={activeId === sparse[0].id} />
            <div className="branch-label"><i>↙</i><span>INDEX 选块 / MAIN 算内容</span><i>↘</i></div>
            <div className="attention-branches">
              <div>{[sparse[1], sparse[2]].map((node, index) => <div key={node.id}><ModuleNode compact node={node} {...nodeProps} active={activeId === node.id} />{index === 0 && <i>↓</i>}</div>)}</div>
              <div className="main-attention"><span>Q/K/V + paged KV</span><ModuleNode compact node={sparse[3]} {...nodeProps} active={activeId === sparse[3].id} /></div>
            </div>
            <div className="join-line"><span>↓ selected pages</span></div>
            <ModuleNode compact node={sparse[4]} {...nodeProps} active={activeId === sparse[4].id} />
            <div className="moe-branches">
              <ModuleNode compact node={sparse[5]} {...nodeProps} active={activeId === sparse[5].id} />
              <ModuleNode compact node={sparse[6]} {...nodeProps} active={activeId === sparse[6].id} />
            </div>
            <ModuleNode compact node={sparse[7]} {...nodeProps} active={activeId === sparse[7].id} />
          </section>
          <div className="stage-link"><span>→</span><code>X₆₀</code></div>

          <section className="stage output-stage">
            <header><span>05 · OUTPUT</span><b>词表 logits</b></header>
            <ModuleNode node={OUTPUT_NODE} {...nodeProps} active={activeId === OUTPUT_NODE.id} />
            <div className="output-facts"><span>VOCAB</span><b>200,064</b><span>SHAPE</span><b>[B,S,V]</b></div>
          </section>
        </div>
        <footer className="map-footer"><span><i /> 参数、dtype、shape、shard：官方 checkpoint</span><span>运行路径：vLLM main</span><span>MSA 数学：技术报告</span></footer>
      </section>

      <DetailPanel node={activeNode} pinned={!hoveredNode} tab={tab} setTab={setTab} />
    </div>
  </main>;
}
