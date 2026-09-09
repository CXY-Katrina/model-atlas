import type { OpNode, Weight } from "../types";
import type { LayerType, ExpandedStage } from "./types";
import type { GraphEdge, GraphAlignment } from "../../graph/types";
import { GraphSurface } from "../../graph/surface";
import { GraphPan } from "../../graph/pan";
import { Op, Tensor, AddCircle } from "../../graph/nodes";

function checkpointWeightName(weight?:Weight){
  return weight?.key.replace(/^language_model\.model\.layers\.\d+\./,"")??"weight";
}

function InputWeightedOp({node,active,onHover,onLeave,onSelect,inputName,inputShape,weightIndex=0,inputGraphId,graphId,weightGraphId,className=""}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;inputName:string;inputShape:string;weightIndex?:number;inputGraphId:string;graphId:string;weightGraphId:string;className?:string}){
  const weight=node.weights[weightIndex];
  const symbolicWeightShape=weight?.shape.replaceAll("6144","H").replaceAll("128","Dₕ")??"[H]";
  return <div className={`input-weighted-op ${className}`}><div className="co-input-row"><Tensor name={inputName} shape={inputShape} graphId={inputGraphId}/><Tensor name={checkpointWeightName(weight)} shape={symbolicWeightShape} role="weight" graphId={weightGraphId}/></div><Op node={node} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} graphId={graphId}/></div>;
}

function StageZoom({type,stage,g,active,onHover,onLeave,onSelect,onClose}:{type:LayerType;stage:Exclude<ExpandedStage,null>;g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;onClose:()=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const N=({id,graphId}:{id:string;graphId?:string})=><Op node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const IW=({id,inputName,inputShape,weightIndex,inputGraphId,graphId,weightGraphId,className}:{id:string;inputName:string;inputShape:string;weightIndex?:number;inputGraphId:string;graphId:string;weightGraphId:string;className?:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} weightIndex={weightIndex} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId} className={className}/>;
  if(stage==="ffn"&&type==="dense"){
    const edges:GraphEdge[]=[
      {from:"mlp-uhat",to:"mlp-gateup",toPort:"top"},{from:"mlp-wgate",to:"mlp-gateup",toPort:"top-left"},{from:"mlp-wup",to:"mlp-gateup",toPort:"top-right"},{from:"mlp-gateup",to:"mlp-packed"},{from:"mlp-packed",to:"mlp-split"},{from:"mlp-split",to:"mlp-gate"},{from:"mlp-split",to:"mlp-up"},{from:"mlp-gate",to:"mlp-gate-act"},{from:"mlp-up",to:"mlp-up-act"},{from:"mlp-gate-act",to:"mlp-mul"},{from:"mlp-up-act",to:"mlp-mul"},{from:"mlp-mul",to:"mlp-activated"},{from:"mlp-activated",to:"mlp-down"},{from:"mlp-wdown",to:"mlp-down",fromPort:"left",toPort:"right"},{from:"mlp-down",to:"mlp-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>SWIGLU-OAI MLP · L0–2</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className="mlp-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="mlp-uhat"/><Tensor name="mlp.gate_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wgate"/><N id="gateup" graphId="mlp-gateup"/><Tensor name="mlp.up_proj.weight⁽ʳ⁾" shape="[H_dense/TP,H]" role="weight" graphId="mlp-wup"/><Tensor name="packed gate_up⁽ʳ⁾" shape="[B,S,2H_dense/TP]" graphId="mlp-packed"/><N id="gatesplit" graphId="mlp-split"/><Tensor name="G⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-gate"/><Tensor name="U⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-up"/><button type="button" className="mini-math activation-step" data-graph-id="mlp-gate-act" aria-label="Gate 分支：先对 G⁽ʳ⁾ 做上界截断，再计算门控激活" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><b>min(G⁽ʳ⁾, C) · σ(α·min(G⁽ʳ⁾, C))</b></button><button type="button" className="mini-math activation-step" data-graph-id="mlp-up-act" aria-label="Up 分支：对 U⁽ʳ⁾ 做双边截断后加 beta" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><b>clip(U⁽ʳ⁾, −C, C) + β</b></button><button className="multiply-circle" data-graph-id="mlp-mul" aria-label="两个分支逐元素相乘" title="逐元素相乘" aria-pressed={active===g.swiglu.id} onPointerDown={()=>onSelect(g.swiglu)} onClick={event=>{if(event.detail===0)onSelect(g.swiglu)}} onMouseEnter={()=>onHover(g.swiglu)} onMouseLeave={onLeave}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.25 4.25 11.75 11.75M11.75 4.25 4.25 11.75"/></svg></button><Tensor name="Z⁽ʳ⁾" shape="[B,S,H_dense/TP]" graphId="mlp-activated"/><N id="down" graphId="mlp-down"/><Tensor name="mlp.down_proj.weight⁽ʳ⁾" shape="[H,H_dense/TP]" role="weight" graphId="mlp-wdown"/><Tensor name="Yffn" shape="[B,S,H]" graphId="mlp-y"/>
    </GraphSurface></GraphPan></section>;
  }
  if(stage==="ffn"){
    const edges:GraphEdge[]=[
      {from:"moe-u",to:"moe-router",fromPort:"bottom-left",approach:28},{from:"moe-wrouter",to:"moe-router",fromPort:"right",toPort:"left"},{from:"moe-router",to:"moe-router-logits"},{from:"moe-u",to:"moe-experts"},{from:"moe-router-logits",to:"moe-experts",toPort:"top-left",approach:28},{from:"moe-wexperts",to:"moe-experts",fromPort:"right",toPort:"left"},{from:"moe-experts",to:"moe-routed"},{from:"moe-u",to:"moe-shared",fromPort:"bottom-right",toPort:"top",approach:28},{from:"moe-wshared",to:"moe-shared",fromPort:"left",toPort:"right"},{from:"moe-shared",to:"moe-shared-out"},{from:"moe-routed",to:"moe-sum",toPort:"top-left",approach:24},{from:"moe-shared-out",to:"moe-sum",toPort:"top-right",approach:24},{from:"moe-sum",to:"moe-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>TOP-4 MOE + SHARED EXPERT · L3–59</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className="moe-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="moe-u"/>
      <Tensor name="router gate weight" shape="[E,H]" role="weight" graphId="moe-wrouter"/><N id="router" graphId="moe-router"/>
      <Tensor name="router logits" shape="[B,S,E]" graphId="moe-router-logits"/>
      <Tensor name="routed expert weights · correction bias" shape="E × expert weights · [E]" role="weight" graphId="moe-wexperts"/><N id="experts" graphId="moe-experts"/>
      <Tensor name="weighted routed output" shape="[B,S,H]" graphId="moe-routed"/>
      <N id="shared" graphId="moe-shared"/><Tensor name="shared expert weights ×3" shape="gate / up / down" role="weight" graphId="moe-wshared"/>
      <Tensor name="shared output" shape="[B,S,H]" graphId="moe-shared-out"/>
      <N id="sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[B,S,H]" graphId="moe-y"/>
    </GraphSurface></GraphPan></section>;
  }
  const dense=type==="dense";
  const ids=dense?{project:"qkv",split:"split",qnorm:"qnorm",knorm:"knorm",ropeq:"ropeq",ropek:"ropek"}:{project:"packed",split:"split",qnorm:"mainnorm",knorm:"mainnorm",ropeq:"rope",ropek:"rope"};
  const edges:GraphEdge[]=[
    {from:"attn-x",to:"attn-project"},{from:"attn-project",to:"attn-packed"},{from:"attn-packed",to:"attn-split"},{from:"attn-split",to:"attn-q",fanout:"attn-five-way",departure:32},{from:"attn-split",to:"attn-k",fanout:"attn-five-way",departure:32},{from:"attn-split",to:"attn-v",fanout:"attn-five-way",departure:32},{from:"attn-q",to:"attn-qnorm",toPort:"top-left",approach:38},{from:"attn-wq",to:"attn-qnorm",toPort:"top-right",approach:38},{from:"attn-qnorm",to:"attn-qt"},{from:"attn-qt",to:"attn-qrope",toPort:"top-left",approach:38},{from:"attn-posq",to:"attn-qrope",toPort:"top-right",approach:38},{from:"attn-qrope",to:"attn-qr"},{from:"attn-k",to:"attn-knorm",toPort:"top-left",approach:38},{from:"attn-wk",to:"attn-knorm",toPort:"top-right",approach:38},{from:"attn-knorm",to:"attn-kt"},{from:"attn-kt",to:"attn-krope",toPort:"top-left",approach:38},{from:"attn-posk",to:"attn-krope",toPort:"top-right",approach:38},{from:"attn-krope",to:"attn-kr"},{from:"attn-kr",to:"attn-cache",toPort:"top-left",approach:72},{from:"attn-v",to:"attn-cache",toPort:"top-right",approach:72},
    {from:"attn-cache-meta",to:"attn-cache",fromPort:"right",toPort:"left"},
    {from:"attn-cache",to:"attn-paged-k",fanout:"attn-cache-outputs",departure:32},
    {from:"attn-cache",to:"attn-paged-v",fanout:"attn-cache-outputs",departure:32},
    {from:"attn-qr",to:"attn-qk",toPort:"top-left",approach:28},
    {from:"attn-paged-k",to:"attn-qk"},
    {from:"attn-qk",to:"attn-scale"},{from:"attn-scale",to:"attn-scaled"},{from:"attn-scaled",to:"attn-mask"},{from:"attn-bounds",to:"attn-mask",fromPort:"right",toPort:"left"},{from:"attn-mask",to:"attn-softmax"},{from:"attn-softmax",to:"attn-p"},{from:"attn-p",to:"attn-pv",toPort:"top-left",approach:34},{from:"attn-paged-v",to:"attn-pv",toPort:"top-right",approach:34},{from:"attn-pv",to:"attn-heads"},{from:"attn-heads",to:"attn-oproj"},{from:"attn-oproj",to:"attn-y"},
    ...(!dense?([
      {from:"attn-split",to:"attn-qidx",fanout:"attn-five-way",departure:32},{from:"attn-split",to:"attn-kidx",fanout:"attn-five-way",departure:32},
      {from:"attn-qidx",to:"attn-idxnorm",toPort:"top-left",approach:34},{from:"attn-kidx",to:"attn-idxnorm"},
      {from:"attn-idxnorm",to:"attn-idxquery",fromPort:"bottom-left",approach:30},{from:"attn-idxnorm",to:"attn-idxcache"},
      {from:"attn-idxslots",to:"attn-idxcache",fromPort:"left",toPort:"right"},
      {from:"attn-idxquery",to:"attn-idxscore",toPort:"top-left",approach:30},{from:"attn-idxcache",to:"attn-idxscore"},
      {from:"attn-idxscore",to:"attn-idxmask"},{from:"attn-idxbounds",to:"attn-idxmask",fromPort:"left",toPort:"right"},{from:"attn-idxmask",to:"attn-blockmax"},{from:"attn-blockmax",to:"attn-topk"},{from:"attn-topk",to:"attn-topids"},{from:"attn-topids",to:"attn-qk",toPort:"top-right",approach:28}
    ] satisfies GraphEdge[]) : []),
  ];
  const alignments:GraphAlignment[]=[
    {node:"attn-kidx",between:["attn-idxnorm"]},
    {node:"attn-paged-k",between:["attn-qk"]},
    {node:"attn-cache",between:["attn-paged-k","attn-paged-v"]},
  ];
  return <section className="stage-zoom lesson-zoom attention-lesson"><header><span>{dense?"GQA + PARTIAL ROPE · L0–2":"MINIMAX SPARSE ATTENTION + PARTIAL ROPE · L3–59"}</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className={`attention-flowchart connected-attention-graph ${dense?"dense-attention":"sparse-attention"}`} edges={edges} alignments={alignments}>
    <div className="compact-chain"><Tensor name="X̂" shape="[B,S,H]" graphId="attn-x"/><N id={ids.project} graphId="attn-project"/><Tensor name="packed" shape={dense?"[B,S,(Nₕ+2Nₖᵥ)·Dₕ]":"[B,S,(Nₕ+2Nₖᵥ)·Dₕ+(N_idx+1)·D_idx]"} graphId="attn-packed"/><N id={ids.split} graphId="attn-split"/></div>
    <div className={`attention-branches ${dense?"dense":""}`}>{!dense&&<div className="index-ribbon"><header className="index-ribbon-label">LIGHTNING INDEXER</header><div className="multi-source"><Tensor name="Qidx" shape="[B,S,N_idx,D_idx]" graphId="attn-qidx"/><Tensor name="Kidx" shape="[B,T,1,D_idx]" graphId="attn-kidx"/></div><N id="idxnorm" graphId="attn-idxnorm"/><Tensor name="Index Q query" shape="[B,N_idx,S,D_idx]" graphId="attn-idxquery"/><N id="idxcache" graphId="attn-idxcache"/><Tensor name="index slot_mapping" shape="[Nq]" role="side" graphId="attn-idxslots"/><N id="idxscore" graphId="attn-idxscore"/><Tensor name="position_ids · future bound" shape="[B,S]" role="side" graphId="attn-idxbounds"/><N id="idxmask" graphId="attn-idxmask"/><N id="blockmax" graphId="attn-blockmax"/><N id="topk" graphId="attn-topk"/><Tensor name="block_indices · Top-K_block" shape="[B,N_idx,S,K_block]" graphId="attn-topids"/></div>}
      <div className="attention-data-path"><div className="qkv-lanes"><section><header>Q PATH</header><IW id={ids.qnorm} inputName="Q" inputShape="[B,Nₕ,S,Dₕ]" inputGraphId="attn-q" graphId="attn-qnorm" weightGraphId="attn-wq"/><div className="two-source"><Tensor name="Q̃" shape="same" graphId="attn-qt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posq"/></div><N id={ids.ropeq} graphId="attn-qrope"/><Tensor name="Qᵣ" shape="[B,Nₕ,S,Dₕ]" graphId="attn-qr"/></section><section><header>K PATH</header><IW id={ids.knorm} inputName="K" inputShape="[B,Nₖᵥ,S,Dₕ]" weightIndex={dense?0:1} inputGraphId="attn-k" graphId="attn-knorm" weightGraphId="attn-wk"/><div className="two-source"><Tensor name="K̃" shape="same" graphId="attn-kt"/><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-posk"/></div><N id={ids.ropek} graphId="attn-krope"/><Tensor name="Kᵣ" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-kr"/></section><section><header>V PATH</header><Tensor name="V" shape="[B,Nₖᵥ,S,Dₕ]" graphId="attn-v"/></section></div><div className="kv-cache-flow"><Tensor name="slot_mapping · block_table" shape="runtime" role="side" graphId="attn-cache-meta"/><N id="cache" graphId="attn-cache"/><div className="two-source"><Tensor name="paged K" shape="KV pages" graphId="attn-paged-k"/></div></div></div></div>
    <div className="score-pipeline"><header className="score-pipeline-label">ATTENTION SCORE PIPELINE · 候选 blocks 内计算概率 P</header><N id="qk" graphId="attn-qk"/><N id="scale" graphId="attn-scale"/><Tensor name="scaled scores" shape={dense?"[B,Nₕ,S,T]":"[B,Nₕ,S,K_sel]"} graphId="attn-scaled"/><Tensor name={dense?"causal / pad bounds":"token causal / pad bounds"} shape="runtime metadata" role="side" graphId="attn-bounds"/><N id="mask" graphId="attn-mask"/><N id="softmax" graphId="attn-softmax"/><Tensor name="P" shape={dense?"[B,Nₕ,S,T]":"[B,Nₕ,S,K_sel]"} graphId="attn-p"/><Tensor name="paged V" shape="KV pages" graphId="attn-paged-v"/></div>
    <div className="context-pipeline"><N id="pv" graphId="attn-pv"/><Tensor name="heads" shape="[B,S,Nₕ·Dₕ]" graphId="attn-heads"/><N id="oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[B,S,H]" graphId="attn-y"/></div>
  </GraphSurface></GraphPan></section>;
}

export function DecoderDiagram({type,g,active,expanded,onExpand,onHover,onLeave,onSelect}:{type:LayerType;g:Record<string,OpNode>;active:string;expanded:ExpandedStage;onExpand:(stage:ExpandedStage)=>void;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const IW=({id,inputName,inputShape,inputGraphId,graphId,weightGraphId}:{id:string;inputName:string;inputShape:string;inputGraphId:string;graphId:string;weightGraphId:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId}/>;
  const A=({id,graphId}:{id:string;graphId:string})=><AddCircle node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const edges:GraphEdge[]=[{from:"main-x",to:"main-norm",toPort:"top-left",approach:48},{from:"main-win",to:"main-norm",toPort:"top-right",approach:48},{from:"main-norm",to:"main-attn"},{from:"main-attn",to:"main-add1"},{from:"main-x",to:"main-add1",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add1",to:"main-u",approach:52},{from:"main-u",to:"main-post",toPort:"top-left",approach:48},{from:"main-wpost",to:"main-post",toPort:"top-right",approach:48},{from:"main-post",to:"main-ffn"},{from:"main-ffn",to:"main-add2"},{from:"main-u",to:"main-add2",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-add2",to:"main-out",approach:44}];
  return <div className={`decoder-workbench ${expanded?"has-zoom":""}`}>{!expanded&&<GraphPan><GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <IW id="norm" inputName="Xₗ · hidden_states / residual stream" inputShape="[B,S,H]" inputGraphId="main-x" graphId="main-norm" weightGraphId="main-win"/><button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={()=>onExpand(expanded==="attention"?null:"attention")}><small>点击展开</small><b>{type==="dense"?"GQA + Partial RoPE":"MiniMax Sparse Attention + Partial RoPE"}</b></button><A id={type==="dense"?"add1":"addattn"} graphId="main-add1"/><IW id="postnorm" inputName="U · updated residual stream" inputShape="[B,S,H]" inputGraphId="main-u" graphId="main-post" weightGraphId="main-wpost"/><button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={()=>onExpand(expanded==="ffn"?null:"ffn")}><small>点击展开</small><b>{type==="dense"?"SwiGLU-OAI MLP":"Top-4 MoE + Shared Expert"}</b></button><A id={type==="dense"?"add2":"addout"} graphId="main-add2"/><Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,H]" graphId="main-out"/>
  </GraphSurface></GraphPan>}{expanded&&<StageZoom type={type} stage={expanded} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={()=>onExpand(null)}/>}</div>;
}
/* eslint-enable react-hooks/static-components */

export function LayerNavigator({type,onChange}:{type:LayerType;onChange:(type:LayerType)=>void}){
  return <div className="layer-nav layer-type-nav"><div className="layer-nav-head"><b>{type==="dense"?"GQA + SwiGLU-OAI MLP":"MiniMax Sparse Attention + MoE"}</b></div><div className="layer-type-options"><button className={type==="dense"?"active dense":"dense"} onClick={()=>onChange("dense")}><span>L0–L2</span><b>GQA + Partial RoPE + SwiGLU-OAI MLP</b><small>3 层共享同一实现</small></button><button className={type==="sparse"?"active sparse":"sparse"} onClick={()=>onChange("sparse")}><span>L3–L59</span><b>MiniMax Sparse Attention + Partial RoPE + Top-4 MoE</b><small>57 层共享同一实现</small></button></div></div>;
}
