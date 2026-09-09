import { useState } from "react";
import type { OpNode, Tab } from "../types";
import type { LayerType, ExpandedStage } from "./types";
import { nextDetailState, type DetailEvent, type DetailState } from "../../detail-selection";
import { denseGraph, sparseGraph } from "./operators";
import { DecoderDiagram, LayerNavigator } from "./diagrams";
import { DetailPanel } from "./details";
import { Arrow } from "../../graph/nodes";

export function MiniMaxView({onExpandedChange}:{onExpandedChange:(expanded:boolean)=>void}){
  const [layerType,setLayerType]=useState<LayerType>("sparse"); const [expanded,setExpandedState]=useState<ExpandedStage>(null);
  const setExpanded=(stage:ExpandedStage)=>{setExpandedState(stage);onExpandedChange(stage!==null)}; const [tab,setTab]=useState<Tab>("io");
  const layer=layerType==="dense"?2:3; const graph=layerType==="dense"?denseGraph(layer):sparseGraph(layer); const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null}); const active=detail.pinned??detail.hovered;
  const updateDetail=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
  const changeLayerType=(next:LayerType)=>{setLayerType(next);setExpanded(null);updateDetail({type:"clear"})};
  return <div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><div className="model-step">Text / Vision Inputs</div><Arrow/><div className="model-step">Embedding Fusion <code>[B,S,H]</code></div><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>L0–2 · GQA + Partial RoPE + SwiGLU-OAI MLP</span><span><i className="sparse"/>L3–59 · MiniMax Sparse Attention + Partial RoPE + MoE</span></div><Arrow/><div className="model-step">Final Gemma RMSNorm <code>[B,S,H]</code></div><Arrow/><div className="model-step">LM Head <code>[B,S,V]</code></div></div>
    <LayerNavigator type={layerType} onChange={changeLayerType}/>
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按结构类型展示</span><h1>{layerType==="dense"?"GQA + Partial RoPE + SwiGLU-OAI MLP · L0–L2 同构":"MiniMax Sparse Attention + Partial RoPE + Top-4 MoE · L3–L59 同构"}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span title="颜色区分算子类型"><i className="operator-swatch"/>OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><DecoderDiagram type={layerType} g={graph} active={active?.id??""} expanded={expanded} onExpand={setExpanded} onHover={node=>updateDetail({type:"hover",node})} onLeave={()=>updateDetail({type:"leave"})} onSelect={node=>{updateDetail({type:"pin",node});setTab("io")}}/></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>updateDetail({type:"clear"})} expanded={expanded} layerType={layerType}/></div>;
}
