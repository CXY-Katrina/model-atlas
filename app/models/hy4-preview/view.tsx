import { useState } from "react";
import type { OpNode, Tab } from "../types";
import type { LayerVariant, ExpandedStage, ArchifyDiagram } from "./types";
import { nextDetailState, type DetailEvent, type DetailState } from "../../detail-selection";
import { buildGraph } from "./operators";
import { DecoderDiagram, LayerNavigator, ArchifyDiagramBar, ArchifyDiagramModal } from "./diagrams";
import { DetailPanel } from "./details";
import { Arrow } from "../../graph/nodes";

export function Hy4View({ onExpandedChange }: { onExpandedChange: (expanded: boolean) => void }) {
  const [variant, setVariant] = useState<LayerVariant>("moe-full");
  const [expanded, setExpandedState] = useState<ExpandedStage>(null);
  const setExpanded = (stage: ExpandedStage) => { setExpandedState(stage); onExpandedChange(stage !== null); };
  const [tab, setTab] = useState<Tab>("io");
  const graph = buildGraph(variant);
  const [detail, setDetail] = useState<DetailState<OpNode>>({ hovered: null, pinned: null });
  const active = detail.pinned ?? detail.hovered;
  const updateDetail = (event: DetailEvent<OpNode>) => setDetail(state => nextDetailState(state, event));
  const changeVariant = (next: LayerVariant) => { setVariant(next); setExpanded(null); updateDetail({ type: "clear" }); };
  const [diagram, setDiagram] = useState<ArchifyDiagram | null>(null);
  return <div className="screen-grid"><section className="map-panel">
    <div className="model-overview">
      <div className="model-step">Token IDs <code>[B,S]</code></div><Arrow />
      <div className="model-step">Embedding <code>[B,S,6144]</code></div><Arrow />
      <div className="overview-stack"><b>Decoder ×78 · iHC 4 通道</b><span><i className="dense" />L0 · dense FFN 18432 + full indexer</span><span><i className="sparse" />L1–77 · Top-8 MoE(20 full / 57 shared indexer)</span></div><Arrow />
      <div className="model-step">hc_head + RMSNorm <code>[B,S,6144]</code></div><Arrow />
      <div className="model-step">lm_head(FP32 logits)<code>[B,S,120832]</code></div>
    </div>
    <LayerNavigator variant={variant} onChange={changeVariant} />
    <ArchifyDiagramBar onOpen={setDiagram} />
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按层族展示</span><h1>{variant === "dense-full" ? "iHC + DSA Sparse MLA + Dense FFN · L0" : variant === "moe-full" ? "iHC + DSA Sparse MLA + Top-8 MoE · 20 个 full 层" : "iHC + DSA Sparse MLA + Top-8 MoE · 57 个 shared 层"}</h1></div><div className="node-legend"><span><i className="tensor-swatch" />TENSOR</span><span><i className="external-swatch" />EXTERNAL</span><span><i className="weight-swatch" />WEIGHT</span><span title="颜色区分算子类型"><i className="operator-swatch" />OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><DecoderDiagram variant={variant} g={graph} active={active?.id ?? ""} expanded={expanded} onExpand={setExpanded} onHover={node => updateDetail({ type: "hover", node })} onLeave={() => updateDetail({ type: "leave" })} onSelect={node => { updateDetail({ type: "pin", node }); setTab("io") }} /></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={() => updateDetail({ type: "clear" })} expanded={expanded} variant={variant} />
    {diagram && <ArchifyDiagramModal diagram={diagram} onClose={() => setDiagram(null)} />}
  </div>;
}
