/* eslint-disable react-hooks/static-components */
import type { OpNode, Weight } from "../types";
import type { ExpandedStage, LayerVariant, ArchifyDiagram } from "./types";
import type { GraphEdge } from "../../graph/types";
import { GraphSurface } from "../../graph/surface";
import { GraphPan } from "../../graph/pan";
import { Op, Tensor } from "../../graph/nodes";

export const ARCHIFY_DIAGRAMS: ArchifyDiagram[] = [
  { id: "overview", title: "模型总览", description: "embed → 78 层 iHC+DSA → hc_head → lm_head,含共享索引 buffer 与 MTP 草稿" },
  { id: "layer", title: "Decoder 层 × iHC", description: "hc_pre/hc_post 包裹的注意力与 FFN 子块,4 通道残差" },
  { id: "dsa-indexer", title: "DSA lightning indexer", description: "32×128 索引头、FP8 打分、top-2048 与 shared 层复用" },
  { id: "dsa-mla", title: "稀疏 MLA", description: "压缩 KV、候选内精确 softmax、learnable sink、gated MLA" },
  { id: "moe", title: "MoE 子块", description: "FP32 sigmoid 路由 Top-8、clamped SwiGLU、shared expert" },
];

function diagramUrl(id: string) {
  return new URL(`models/hy4-preview/diagrams/${id}.html`, document.baseURI).href;
}

export function ArchifyDiagramBar({ onOpen }: { onOpen: (diagram: ArchifyDiagram) => void }) {
  return <div className="archify-bar" role="navigation" aria-label="Archify 静态图示入口">
    <span className="archify-bar-label">ARCHIFY 图示</span>
    {ARCHIFY_DIAGRAMS.map(item => <button key={item.id} onClick={() => onOpen(item)} title={item.description}>{item.title}</button>)}
  </div>;
}

export function ArchifyDiagramModal({ diagram, onClose }: { diagram: ArchifyDiagram; onClose: () => void }) {
  return <div className="modal-backdrop" onMouseDown={onClose}>
    <section className="archify-modal" onMouseDown={event => event.stopPropagation()} role="dialog" aria-modal="true" aria-label={`Archify 图示:${diagram.title}`}>
      <header><div><span>ARCHIFY · 独立交付图示</span><b>{diagram.title}</b><small>{diagram.description}</small></div><div className="archify-modal-actions"><a href={diagramUrl(diagram.id)} target="_blank" rel="noreferrer">新窗口打开 ↗</a><button onClick={onClose} aria-label="关闭图示">×</button></div></header>
      <iframe title={`Archify 图示:${diagram.title}`} src={diagramUrl(diagram.id)} />
    </section>
  </div>;
}

function checkpointWeightName(weight?: Weight) {
  const name = weight?.key.replace(/^model\.layers\.(?:l|\d+)\./, "") ?? "weight";
  return breakable(name);
}

/** zero-width breaks after separators so long weight keys wrap at sane points */
function breakable(name: string) {
  return name.replace(/\./g, ".\u200b").replace(/_/g, "_\u200b");
}

function InputWeightedOp({ node, active, onHover, onLeave, onSelect, inputName, inputShape, inputGraphId, graphId, weightGraphId }: {
  node: OpNode; active: boolean; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void;
  inputName: string; inputShape: string; inputGraphId: string; graphId: string; weightGraphId: string;
}) {
  const weight = node.weights[0];
  return <div className="input-weighted-op">
    <div className="co-input-row">
      <Tensor name={inputName} shape={inputShape} graphId={inputGraphId} />
      {weight && <Tensor name={checkpointWeightName(weight)} shape={weight.shape} role="weight" graphId={weightGraphId} />}
    </div>
    <Op node={node} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} graphId={graphId} />
  </div>;
}

type Common = { g: Record<string, OpNode>; active: string; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void };

function AttentionZoom({ variant, g, active, onHover, onLeave, onSelect, onClose }: Common & { variant: LayerVariant; onClose: () => void }) {
  const p = { active: false, onHover, onLeave, onSelect };
  const full = variant !== "moe-shared";
  const edges: GraphEdge[] = [
    { from: "attn-x", to: "attn-aproj", fromPort: "right", toPort: "left" },
    { from: "attn-aproj", to: "attn-qan-in" },
    { from: "attn-aproj", to: "attn-kvn-in" },
    { from: "attn-qan", to: "attn-qt" },
    { from: "attn-qt", to: "attn-qb", toPort: "top-left", approach: 30 },
    { from: "attn-qb", to: "attn-rope", departure: 24 },
    { from: "attn-pos", to: "attn-rope", fromPort: "right", toPort: "left" },
    { from: "attn-rope", to: "attn-qr" },
    { from: "attn-kvn", to: "attn-kvt" },
    { from: "attn-kvt", to: "attn-cache", departure: 24 },
    { from: "attn-slots", to: "attn-cache", fromPort: "left", toPort: "right" },
    { from: "attn-kvt", to: "attn-kpe" },
    { from: "attn-kpe", to: "attn-rope", fromPort: "left", toPort: "right" },
    { from: "attn-cache", to: "attn-kvb", toPort: "top-left", approach: 30 },
    { from: "attn-kvb", to: "attn-kv" },
    ...full
      ? [
          { from: "attn-xh", to: "attn-iwk", fromPort: "bottom-right", toPort: "top", approach: 24 },
          { from: "attn-xh", to: "attn-gate", fromPort: "left", toPort: "top", route: "bus-left", approach: 30 },
          { from: "attn-qt", to: "attn-iwqb", fromPort: "left", toPort: "right" },
          { from: "attn-iwqb", to: "attn-iquant" },
          { from: "attn-iwk", to: "attn-ikn" },
          { from: "attn-ikn", to: "attn-icache" },
          { from: "attn-iquant", to: "attn-iscore", toPort: "top-right", approach: 26 },
          { from: "attn-icache", to: "attn-iscore", toPort: "top-left", approach: 26 },
          { from: "attn-iscore", to: "attn-topids" },
        ] as GraphEdge[]
      : [
          { from: "attn-xh", to: "attn-gate", fromPort: "left", toPort: "top", route: "bus-left", approach: 30 },
          { from: "attn-buffer", to: "attn-ishared", fromPort: "right", toPort: "left" },
          { from: "attn-ishared", to: "attn-topids" },
        ] as GraphEdge[],
    { from: "attn-qr", to: "attn-qk", fanin: "attn-qk" },
    { from: "attn-kv", to: "attn-qk", fanin: "attn-qk" },
    { from: "attn-topids", to: "attn-qk", fanin: "attn-qk" },
    { from: "attn-qk", to: "attn-sink", fromPort: "right", toPort: "left" },
    { from: "attn-sink", to: "attn-p", fromPort: "right", toPort: "left" },
    { from: "attn-p", to: "attn-pv", fromPort: "right", toPort: "left" },
    { from: "attn-kv", to: "attn-pv", toPort: "top-right", approach: 28 },
    { from: "attn-pv", to: "attn-heads" },
    { from: "attn-heads", to: "attn-gate", fromPort: "left", toPort: "right" },
    { from: "attn-gate", to: "attn-oproj", fromPort: "left", toPort: "right" },
    { from: "attn-oproj", to: "attn-y", fromPort: "left", toPort: "right" },
  ];
  return <section className="stage-zoom lesson-zoom attention-lesson">
    <header><span>{full ? "DSA SPARSE MLA · 78 层同构" : "DSA SPARSE MLA · SHARED INDEXER"}</span><button onClick={onClose}>收起 ×</button></header>
    <GraphPan><GraphSurface className={`attention-flowchart hy-attention-graph ${full ? "is-full" : "is-shared"}`} edges={edges}>
      <div className="compact-chain">
        <Tensor name="X̂" shape="[T,6144]" graphId="attn-x" />
        <Op node={g.aproj} {...p} active={active === g.aproj.id} graphId="attn-aproj" />
        {!full && <Tensor name="hidden(子块输入)" shape="[T,6144]" role="side" graphId="attn-xh" />}
      </div>
      <div className="attn-branches">
        <div className="index-ribbon">
          <header className="index-ribbon-label">{full ? "LIGHTNING INDEXER · FULL" : "SHARED INDEXER · 复用"}</header>
          {full ? <>
            <Tensor name="hidden(子块输入)" shape="[T,6144]" role="side" graphId="attn-xh" />
            <Op node={g.iwk} {...p} active={active === g.iwk.id} graphId="attn-iwk" />
            <Op node={g.iwqb} {...p} active={active === g.iwqb.id} graphId="attn-iwqb" />
            <Op node={g.ikn} {...p} active={active === g.ikn.id} graphId="attn-ikn" />
            <Op node={g.iquant} {...p} active={active === g.iquant.id} graphId="attn-iquant" />
            <Tensor name="FP8 key cache" shape="[T,128]+scale" role="side" graphId="attn-icache" />
            <Op node={g.iscore} {...p} active={active === g.iscore.id} graphId="attn-iscore" />
          </> : <>
            <Tensor name="topk buffer" shape="[T,2048] int32" graphId="attn-buffer" />
            <Op node={g.ishared} {...p} active={active === g.ishared.id} graphId="attn-ishared" />
          </>}
          <Tensor name="候选索引 · top-2048" shape="[T,2048] int32" graphId="attn-topids" />
        </div>
        <div className="attention-data-path">
          <div className="hy-lanes">
            <section>
              <header>Q PATH</header>
              <InputWeightedOp node={g.qan} {...p} active={active === g.qan.id} inputName="q_a" inputShape="[T,2048]" inputGraphId="attn-qan-in" graphId="attn-qan" weightGraphId="attn-wqan" />
              <Tensor name="q̃" shape="[T,2048]" graphId="attn-qt" />
              <Op node={g.qb} {...p} active={active === g.qb.id} graphId="attn-qb" />
              <div className="io-row">
                <Tensor name="positions" shape="[T]" role="side" graphId="attn-pos" />
                <Op node={g.rope} {...p} active={active === g.rope.id} graphId="attn-rope" />
              </div>
              <Tensor name="Qᵣ · 64×(192+64)" shape="[T,64,256]" graphId="attn-qr" />
            </section>
            <section>
              <header>KV PATH</header>
              <InputWeightedOp node={g.kvn} {...p} active={active === g.kvn.id} inputName="kv" inputShape="[T,512]" inputGraphId="attn-kvn-in" graphId="attn-kvn" weightGraphId="attn-wkvn" />
              <Tensor name="k̃ | k_pe" shape="512 | 64" graphId="attn-kvt" />
              <Tensor name="k_pe(旋转)" shape="[T,1,64]" role="side" graphId="attn-kpe" />
              <div className="io-row">
                <Op node={g.cache} {...p} active={active === g.cache.id} graphId="attn-cache" />
                <div className="side-stack">
                  <Tensor name="slot_mapping" shape="[T]" role="side" graphId="attn-slots" />
                  <Tensor name="压缩 KV 页" shape="默认 BF16 · fp8 量化部署为 fp8_ds_mla" role="side" graphId="attn-pages" />
                </div>
              </div>
              <Op node={g.kvb} {...p} active={active === g.kvb.id} graphId="attn-kvb" />
              <Tensor name="K_nope | V" shape="64×192 | 64×256" graphId="attn-kv" />
            </section>
          </div>
        </div>
      </div>
      <div className="score-pipeline">
        <div className="pipeline-row">
          <Op node={g.qk} {...p} active={active === g.qk.id} graphId="attn-qk" />
          <Op node={g.sink} {...p} active={active === g.sink.id} graphId="attn-sink" />
          <Tensor name="P(sink 折入分母)" shape="[64,≤2048]" graphId="attn-p" />
          <Op node={g.pv} {...p} active={active === g.pv.id} graphId="attn-pv" />
        </div>
        <div className="pipeline-row">
          <Tensor name="Y_attn" shape="[T,6144]" graphId="attn-y" />
          <Op node={g.oproj} {...p} active={active === g.oproj.id} graphId="attn-oproj" />
          <Op node={g.gate} {...p} active={active === g.gate.id} graphId="attn-gate" />
          <Tensor name="heads" shape="[T,64,256]" graphId="attn-heads" />
        </div>
        <footer className="score-pipeline-label">ATTENTION SCORE PIPELINE · 单内核数学分解 — QKᵀ/scale、softmax(+sink)、P·V 由 FlashMLA sparse 一次内核算出,score/P 为内核内部量,不独立物化 · 候选 2048 内精确注意力</footer>
      </div>
    </GraphSurface></GraphPan>
  </section>;
}

function FfnZoom({ variant, g, active, onHover, onLeave, onSelect, onClose }: Common & { variant: LayerVariant; onClose: () => void }) {
  const p = { active: false, onHover, onLeave, onSelect };
  if (variant === "dense-full") {
    const edges: GraphEdge[] = [
      { from: "ffn-u", to: "ffn-gateup" },
      { from: "ffn-w", to: "ffn-gateup", fromPort: "right", toPort: "left" },
      { from: "ffn-gateup", to: "ffn-packed" },
      { from: "ffn-packed", to: "ffn-act" },
      { from: "ffn-act", to: "ffn-down" },
      { from: "ffn-wdown", to: "ffn-down", fromPort: "left", toPort: "right" },
      { from: "ffn-down", to: "ffn-y" },
    ];
    return <section className="stage-zoom lesson-zoom">
      <header><span>DENSE FFN · L0 · 18432</span><button onClick={onClose}>收起 ×</button></header>
      <GraphPan><GraphSurface className="ffn-node-graph hy-ffn-graph" edges={edges}>
        <Tensor name="Û" shape="[T,6144]" graphId="ffn-u" />
        <Tensor name={breakable("mlp.{gate,up}_proj.weight")} shape="2×[18432,6144]" role="weight" graphId="ffn-w" />
        <Op node={g.gateup} {...p} active={active === g.gateup.id} graphId="ffn-gateup" />
        <Tensor name="packed gate_up" shape="[T,36864]" graphId="ffn-packed" />
        <Op node={g.act} {...p} active={active === g.act.id} graphId="ffn-act" />
        <Op node={g.down} {...p} active={active === g.down.id} graphId="ffn-down" />
        <Tensor name={breakable("mlp.down_proj.weight")} shape="[6144,18432]" role="weight" graphId="ffn-wdown" />
        <Tensor name="Y_ffn" shape="[T,6144]" graphId="ffn-y" />
      </GraphSurface></GraphPan>
    </section>;
  }
  const edges: GraphEdge[] = [
    { from: "moe-u", to: "moe-router", fromPort: "bottom-left", toPort: "top", approach: 28 },
    { from: "moe-wrouter", to: "moe-router", fromPort: "right", toPort: "left" },
    { from: "moe-router", to: "moe-logits" },
    { from: "moe-logits", to: "moe-select" },
    { from: "moe-select", to: "moe-w", fromPort: "bottom-left", toPort: "top", approach: 28 },
    { from: "moe-w", to: "moe-experts", toPort: "top-left", approach: 28 },
    { from: "moe-wexp", to: "moe-experts", fromPort: "right", toPort: "left" },
    { from: "moe-u", to: "moe-experts", toPort: "top", approach: 40 },
    { from: "moe-u", to: "moe-mshare", fromPort: "bottom-right", toPort: "top", approach: 40 },
    { from: "moe-experts", to: "moe-msum", fanin: "moe-msum" },
    { from: "moe-mshare", to: "moe-msum", fanin: "moe-msum" },
    { from: "moe-msum", to: "moe-y" },
  ];
  return <section className="stage-zoom lesson-zoom">
    <header><span>TOP-8 MOE + SHARED EXPERT · L1–77</span><button onClick={onClose}>收起 ×</button></header>
    <GraphPan><GraphSurface className="moe-node-graph hy-moe-graph" edges={edges}>
      <Tensor name="Û" shape="[T,6144]" graphId="moe-u" />
      <Tensor name="mlp.gate.weight(FP32)" shape="[256,6144]" role="weight" graphId="moe-wrouter" />
      <Op node={g.router} {...p} active={active === g.router.id} graphId="moe-router" />
      <Tensor name="router_logits" shape="[T,256]" graphId="moe-logits" />
      <Op node={g.select} {...p} active={active === g.select.id} graphId="moe-select" />
      <Tensor name="ŵ · Top-8" shape="8 / token" graphId="moe-w" />
      <Tensor name={breakable("experts.{gate_up,down}_proj")} shape="256 专家 · 融合" role="weight" graphId="moe-wexp" />
      <Op node={g.experts} {...p} active={active === g.experts.id} graphId="moe-experts" />
      <Op node={g.mshare} {...p} active={active === g.mshare.id} graphId="moe-mshare" />
      <Op node={g.msum} {...p} active={active === g.msum.id} graphId="moe-msum" />
      <Tensor name="Y_moe" shape="[T,6144]" graphId="moe-y" />
    </GraphSurface></GraphPan>
  </section>;
}

export function DecoderDiagram({ variant, g, active, expanded, onExpand, onHover, onLeave, onSelect }: {
  variant: LayerVariant; g: Record<string, OpNode>; active: string; expanded: ExpandedStage;
  onExpand: (stage: ExpandedStage) => void; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void;
}) {
  const p = { active: false, onHover, onLeave, onSelect };
  const edges: GraphEdge[] = [
    { from: "main-x", to: "main-hcpa" },
    { from: "main-hcpa", to: "main-n1in" },
    { from: "main-hcpa", to: "main-hcpo1", fromPort: "right", toPort: "right", route: "side-right" },
    { from: "main-norm1", to: "main-attn" },
    { from: "main-attn", to: "main-hcpo1" },
    { from: "main-x", to: "main-hcpo1", fromPort: "left", toPort: "left", route: "side-left" },
    { from: "main-hcpo1", to: "main-hcpm" },
    { from: "main-hcpm", to: "main-n2in" },
    { from: "main-hcpm", to: "main-hcpo2", fromPort: "right", toPort: "right", route: "side-right" },
    { from: "main-norm2", to: "main-ffn" },
    { from: "main-ffn", to: "main-hcpo2" },
    { from: "main-hcpo1", to: "main-hcpo2", fromPort: "left", toPort: "left", route: "side-left" },
    { from: "main-hcpo2", to: "main-out" },
  ];
  return <div className={`decoder-workbench ${expanded ? "has-zoom" : ""}`}>
    {!expanded && <GraphPan><GraphSurface className="decoder-column decoder-node-graph hy-decoder-graph" edges={edges}>
      <Tensor name="Xₗ · hidden / 4 通道残差" shape="[T,4,6144]" graphId="main-x" />
      <Op node={g.hcprea} {...p} active={active === g.hcprea.id} graphId="main-hcpa" />
      <InputWeightedOp node={g.norm1} {...p} active={active === g.norm1.id} inputName="归约 hidden" inputShape="[T,6144]" inputGraphId="main-n1in" graphId="main-norm1" weightGraphId="main-w1" />
      <button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={() => onExpand(expanded === "attention" ? null : "attention")}>
        <small>点击展开</small><b>{variant === "moe-shared" ? "DSA Sparse MLA · 复用索引" : "DSA Sparse MLA · 自算索引"}</b>
      </button>
      <Op node={g.hcposta} {...p} active={active === g.hcposta.id} graphId="main-hcpo1" />
      <Op node={g.hcprem} {...p} active={active === g.hcprem.id} graphId="main-hcpm" />
      <InputWeightedOp node={g.norm2} {...p} active={active === g.norm2.id} inputName="归约 hidden" inputShape="[T,6144]" inputGraphId="main-n2in" graphId="main-norm2" weightGraphId="main-w2" />
      <button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={() => onExpand(expanded === "ffn" ? null : "ffn")}>
        <small>点击展开</small><b>{variant === "dense-full" ? "Dense SwiGLU FFN · 18432" : "Top-8 MoE + Shared Expert"}</b>
      </button>
      <Op node={g.hcpostm} {...p} active={active === g.hcpostm.id} graphId="main-hcpo2" />
      <Tensor name="Xₗ₊₁ · 4 通道 hidden" shape="[T,4,6144]" graphId="main-out" />
    </GraphSurface></GraphPan>}
    {expanded && (expanded === "attention"
      ? <AttentionZoom variant={variant} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={() => onExpand(null)} />
      : <FfnZoom variant={variant} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={() => onExpand(null)} />)}
  </div>;
}

export function LayerNavigator({ variant, onChange }: { variant: LayerVariant; onChange: (v: LayerVariant) => void }) {
  return <div className="layer-nav layer-type-nav">
    <div className="layer-nav-head"><b>{variant === "dense-full" ? "Dense FFN + Full Indexer" : variant === "moe-full" ? "MoE + Full Indexer" : "MoE + Shared Indexer"}</b></div>
    <div className="layer-type-options">
      <button className={variant === "dense-full" ? "active dense" : "dense"} onClick={() => onChange("dense-full")}>
        <span>L0</span><b>Dense SwiGLU FFN · Full Indexer</b><small>1 层 · 自算 top-2048 索引</small>
      </button>
      <button className={variant === "moe-full" ? "active sparse" : "sparse"} onClick={() => onChange("moe-full")}>
        <span>L1, 5, …, 77</span><b>Top-8 MoE · Full Indexer</b><small>20 层 · 自算 top-2048 索引</small>
      </button>
      <button className={variant === "moe-shared" ? "active shared" : "shared"} onClick={() => onChange("moe-shared")}>
        <span>L2, 3, 4, …</span><b>Top-8 MoE · Shared Indexer</b><small>57 层 · 复用最近 full 层索引</small>
      </button>
    </div>
  </div>;
}
