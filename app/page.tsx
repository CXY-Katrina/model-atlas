"use client";

import { useEffect, useRef, useState } from "react";
import { denseNodes, layerShard, sparseNodes, visionNodes, type Node, type Weight } from "./model-data";

type Tab = "io" | "formula" | "code" | "weights";
type OpKind = "io" | "norm" | "linear" | "split" | "rope" | "matmul" | "scale" | "mask" | "softmax" | "activation" | "route" | "cache" | "add";
type OpNode = Node & { kind: OpKind };

const VLLM_COMMIT = "edd4c8176cfd98ece8a29beda574378c42971967";
const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3";

const MODEL_REGISTRY = [
  { id: "minimax-m3", name: "MiniMax-M3", enabled: true },
  { id: "kimi-k3", name: "Kimi K3 · 待添加", enabled: false },
  { id: "deepseek-v4", name: "DeepSeek V4 · 待添加", enabled: false },
  { id: "step-3.7", name: "Step 3.7 · 待添加", enabled: false },
];

const cloneOp = (base: Node, values: Partial<Node> & { id: string; kind: OpKind; title: string }): OpNode => ({ ...base, ...values });
const pinSource = (url: string) => url.replace("/blob/main/", `/blob/${VLLM_COMMIT}/`);

function denseGraph(layer: number): Record<string, OpNode> {
  const [norm, qkv, attn, out, mlp] = denseNodes(layer);
  const shard = layerShard(layer);
  const postNorm: Weight = { key: `language_model.model.layers.${layer}.post_attention_layernorm.weight`, shape: "[6144]", dtype: "BF16", shard, params: "6,144" };
  return {
    input: cloneOp(norm,{id:"d-input",kind:"io",title:"Hidden states",kicker:`L${layer} INPUT`,input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[],formula:"residual ← Xₗ; working ← Xₗ"}),
    norm: cloneOp(norm,{id:"d-norm",kind:"norm",title:"RMSNorm"}),
    qkv: cloneOp(qkv,{id:"d-qkv",kind:"linear",title:"QKV Projection"}),
    split: cloneOp(qkv,{id:"d-split",kind:"split",title:"Split Q / K / V",input:"packed qkv",inputShape:"[B,S,9216]",output:"Q · K · V",outputShape:"8192 · 512 · 512",formula:"split(qkv,[8192,512,512],dim=-1)",formulaNote:"checkpoint 中三块矩阵分离；vLLM 运行时一次 GEMM 后切分。",weights:[]}),
    qnorm: cloneOp(attn,{id:"d-qnorm",kind:"norm",title:"Q RMSNorm",input:"Q",inputShape:"[B,64,S,128]",output:"Q̃",outputShape:"[B,64,S,128]",formula:"Q̃=Q/√(mean(Q²)+ε)⊙(1+γq)",weights:attn.weights.filter(w=>w.key.includes("q_norm"))}),
    knorm: cloneOp(attn,{id:"d-knorm",kind:"norm",title:"K RMSNorm",input:"K",inputShape:"[B,4,T,128]",output:"K̃",outputShape:"[B,4,T,128]",formula:"K̃=K/√(mean(K²)+ε)⊙(1+γk)",weights:attn.weights.filter(w=>w.key.includes("k_norm"))}),
    ropeq: cloneOp(attn,{id:"d-ropeq",kind:"rope",title:"Partial RoPE (Q)",input:"Q̃ + positions",inputShape:"[B,64,S,128] + [S]",output:"Qᵣ",outputShape:"[B,64,S,128]",formula:"Qᵣ[:64]=RoPE(Q̃[:64],pos); Qᵣ[64:]=Q̃[64:]",weights:[]}),
    ropek: cloneOp(attn,{id:"d-ropek",kind:"rope",title:"Partial RoPE (K)",input:"K̃ + positions",inputShape:"[B,4,T,128] + [T]",output:"Kᵣ",outputShape:"[B,4,T,128]",formula:"Kᵣ[:64]=RoPE(K̃[:64],pos); Kᵣ[64:]=K̃[64:]",weights:[]}),
    cache: cloneOp(attn,{id:"d-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"[T,4,128] ×2",output:"visible K,V",outputShape:"[B,4,T,128] ×2",formula:"slot = block_table[seq, logical_block] + offset",formulaNote:"Dense 层读取完整可见历史；block table 决定物理 page。",weights:[]}),
    qk: cloneOp(attn,{id:"d-qk",kind:"matmul",title:"Q × Kᵀ",input:"Qᵣ,Kᵣ",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"scores",outputShape:"[B,64,S,T]",formula:"A=QᵣKᵣᵀ",weights:[]}),
    scale: cloneOp(attn,{id:"d-scale",kind:"scale",title:"Scale 1/√128",input:"A",inputShape:"[B,64,S,T]",output:"scaled scores",outputShape:"[B,64,S,T]",formula:"A←A/√128",weights:[]}),
    mask: cloneOp(attn,{id:"d-mask",kind:"mask",title:"+ Causal Mask",input:"scores + causal mask",inputShape:"[B,64,S,T] + [S,T]",output:"masked scores",outputShape:"[B,64,S,T]",formula:"Aᵢⱼ←Aᵢⱼ + (j≤i ? 0 : −∞)",weights:[]}),
    softmax: cloneOp(attn,{id:"d-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64,S,T]",output:"attention prob",outputShape:"[B,64,S,T]",formula:"P=softmax(A,dim=-1)",weights:[]}),
    pv: cloneOp(attn,{id:"d-pv",kind:"matmul",title:"P × V",input:"P,V",inputShape:"[B,64,S,T] · [B,4,T,128]",output:"heads",outputShape:"[B,S,8192]",formula:"Oₕ=PₕV⌊h/16⌋",weights:[]}),
    oproj: cloneOp(out,{id:"d-oproj",kind:"linear",title:"O Projection"}),
    add1: cloneOp(out,{id:"d-add1",kind:"add",title:"+ Attention Residual",input:"Yattn + residual",inputShape:"2 × [B,S,6144]",output:"U",outputShape:"[B,S,6144]",formula:"U=residual+Yattn",weights:[]}),
    postnorm: cloneOp(norm,{id:"d-postnorm",kind:"norm",title:"Post-attn RMSNorm",input:"U",inputShape:"[B,S,6144]",output:"Û",outputShape:"[B,S,6144]",weights:[postNorm]}),
    gateup: cloneOp(mlp,{id:"d-gateup",kind:"linear",title:"Gate + Up Projection",output:"gate · up",outputShape:"2 × [B,S,12288]",weights:mlp.weights.filter(w=>!w.key.includes("down_proj"))}),
    swiglu: cloneOp(mlp,{id:"d-swiglu",kind:"activation",title:"SwiGLU-OAI",input:"gate,up",inputShape:"2 × [B,S,12288]",output:"activated",outputShape:"[B,S,12288]",weights:[]}),
    down: cloneOp(mlp,{id:"d-down",kind:"linear",title:"Down Projection",input:"activated",inputShape:"[B,S,12288]",output:"Yffn",outputShape:"[B,S,6144]",weights:mlp.weights.filter(w=>w.key.includes("down_proj"))}),
    add2: cloneOp(mlp,{id:"d-add2",kind:"add",title:"+ MLP Residual",input:"Yffn + U",inputShape:"2 × [B,S,6144]",output:"Xₗ₊₁",outputShape:"[B,S,6144]",formula:"Xₗ₊₁=U+MLP(RMSNorm(U))",weights:[]}),
  };
}

function sparseGraph(layer: number): Record<string, OpNode> {
  const [packed,indexer,topk,attn,router,experts,shared,combine]=sparseNodes(layer);
  const shard=layerShard(layer);
  const inputNorm: Weight={key:`language_model.model.layers.${layer}.input_layernorm.weight`,shape:"[6144]",dtype:"BF16",shard,params:"6,144"};
  return {
    input:cloneOp(packed,{id:"s-input",kind:"io",title:"Hidden states",input:"Xₗ",inputShape:"[B,S,6144]",output:"residual + working copy",outputShape:"2 × [B,S,6144]",weights:[]}),
    norm:cloneOp(packed,{id:"s-norm",kind:"norm",title:"RMSNorm",input:"Xₗ",inputShape:"[B,S,6144]",output:"X̂",outputShape:"[B,S,6144]",weights:[inputNorm]}),
    packed:cloneOp(packed,{id:"s-packed",kind:"linear",title:"QKV + Index Projection"}),
    split:cloneOp(packed,{id:"s-split",kind:"split",title:"Split 5 outputs",input:"packed projection",inputShape:"[B,S,9856]",output:"Q/K/V · Qidx/Kidx",outputShape:"8192/512/512 · 512/128",formula:"split(x,[8192,512,512,512,128],dim=-1)",weights:[]}),
    idxnorm:cloneOp(indexer,{id:"s-idxnorm",kind:"norm",title:"Index Q/K Norm",input:"Qidx,Kidx",inputShape:"[B,S,4,128] · [B,T,1,128]",output:"Q̃idx,K̃idx",outputShape:"same",weights:indexer.weights}),
    idxscore:cloneOp(indexer,{id:"s-idxscore",kind:"matmul",title:"Index Q × Kᵀ",input:"Q̃idx,K̃idx",inputShape:"[B,4,S,128] · [B,1,T,128]",output:"token scores",outputShape:"[B,4,S,T]",weights:[]}),
    blockmax:cloneOp(indexer,{id:"s-blockmax",kind:"route",title:"Block Max (128 tokens)",input:"causal token scores",inputShape:"[B,4,S,T]",output:"block scores",outputShape:"[B,4,S,⌈T/128⌉]",weights:[]}),
    topk:cloneOp(topk,{id:"s-topk",kind:"route",title:"Top-16 Blocks",input:"block scores + local priority",inputShape:"[B,4,S,Nblocks]",output:"logical block ids",outputShape:"[B,S,4,16]"}),
    mainnorm:cloneOp(attn,{id:"s-mainnorm",kind:"norm",title:"Main Q/K Norm",input:"Q,K",inputShape:"[B,64,S,128] · [B,4,T,128]",output:"Q̃,K̃",outputShape:"same",weights:attn.weights.filter(w=>w.key.includes("_norm"))}),
    rope:cloneOp(attn,{id:"s-rope",kind:"rope",title:"Partial RoPE",input:"Q̃,K̃ + positions",inputShape:"Q/K + [S]",output:"Qᵣ,Kᵣ",outputShape:"Q/K unchanged",weights:[]}),
    cache:cloneOp(attn,{id:"s-cache",kind:"cache",title:"Paged KV Cache",input:"Kᵣ,V + block table",inputShape:"KV pages + [B,Nblocks]",output:"paged K,V",outputShape:"[Npages,128,4,128] ×2",formula:"physical_page=block_table[logical_block]",weights:[]}),
    select:cloneOp(attn,{id:"s-select",kind:"route",title:"Select KV Pages",input:"paged K,V + Top-16 block ids",inputShape:"KV pages + [B,S,4,16]",output:"selected K,V",outputShape:"≤2048 KV tokens / group",formula:"physical_page=block_table[logical_top16]",weights:[]}),
    qk:cloneOp(attn,{id:"s-qk",kind:"matmul",title:"Q × selected Kᵀ",input:"Qᵣ, selected K",inputShape:"[B,64,S,128] · ≤16×128",output:"sparse scores",outputShape:"[B,64,S,≤2048]",weights:[]}),
    scale:cloneOp(attn,{id:"s-scale",kind:"scale",title:"Scale 1/√128",input:"scores",inputShape:"[B,64,S,≤2048]",output:"scaled scores",outputShape:"same",weights:[]}),
    mask:cloneOp(attn,{id:"s-mask",kind:"mask",title:"+ Causal / Pad Mask",input:"scores + masks",inputShape:"attention layout",output:"masked scores",outputShape:"same",weights:[]}),
    softmax:cloneOp(attn,{id:"s-softmax",kind:"softmax",title:"Softmax",input:"masked scores",inputShape:"[B,64,S,≤2048]",output:"probabilities",outputShape:"same",weights:[]}),
    pv:cloneOp(attn,{id:"s-pv",kind:"matmul",title:"P × selected V",input:"P, selected V",inputShape:"probabilities · KV pages",output:"heads",outputShape:"[B,S,8192]",weights:[]}),
    oproj:cloneOp(attn,{id:"s-oproj",kind:"linear",title:"O Projection",input:"heads",inputShape:"[B,S,8192]",output:"Yattn",outputShape:"[B,S,6144]",weights:attn.weights.filter(w=>w.key.includes("o_proj"))}),
    addattn:cloneOp(combine,{id:"s-addattn",kind:"add",title:"+ Attention Residual",input:"Yattn + residual",inputShape:"2 × [B,S,6144]",output:"U",outputShape:"[B,S,6144]",weights:[]}),
    router:cloneOp(router,{id:"s-router",kind:"route",title:"FP32 Router → Top-4"}),
    experts:cloneOp(experts,{id:"s-experts",kind:"activation",title:"Routed Experts ×4"}),
    shared:cloneOp(shared,{id:"s-shared",kind:"activation",title:"Shared Expert ×1"}),
    sum:cloneOp(combine,{id:"s-sum",kind:"add",title:"Weighted Sum",input:"4 routed + shared",inputShape:"5 × [B,S,6144]",output:"Ymoe",outputShape:"[B,S,6144]",weights:[]}),
    addout:cloneOp(combine,{id:"s-addout",kind:"add",title:"+ Decoder Residual"}),
  };
}

function Op({node,active,onHover,onLeave,onSelect}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  return <button className={`op-node op-${node.kind} ${active?"active":""}`} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onClick={()=>onSelect(node)}><small>OP · {node.kind}</small><b>{node.title}</b></button>;
}

function Tensor({name,shape,external=false}:{name:string;shape:string;external?:boolean}){
  return <div className={`tensor-node ${external?"external-tensor":""}`}><small>{external?"SIDE INPUT":"TENSOR"}</small><b>{name}</b><code>{shape}</code></div>;
}

const Arrow=({label}:{label?:string})=><span className="op-arrow"><i/>{label&&<small>{label}</small>}</span>;

/* eslint-disable react-hooks/static-components -- local alias only shortens a large, stateless operator graph */
function DenseDiagram({g,active,onHover,onLeave,onSelect}:{g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect}; const N=({id}:{id:string})=><Op node={g[id]} {...p} active={active===g[id].id}/>;
  return <div className="operator-diagram dense-diagram">
    <div className="flow-row"><Tensor name="Xₗ" shape="[B,S,6144]"/><Arrow/><N id="norm"/><Arrow/><Tensor name="X̂" shape="[B,S,6144]"/><Arrow/><N id="qkv"/><Arrow/><Tensor name="packed_qkv" shape="[B,S,9216]"/><Arrow/><N id="split"/><Arrow/><Tensor name="Q · K · V" shape="8192 · 512 · 512"/></div>
    <div className="branch-box qkv-branches">
      <section><header>Q BRANCH</header><div className="mini-flow"><Tensor name="Q" shape="[B,64,S,128]"/><Arrow/><N id="qnorm"/><Arrow/><Tensor name="Q̃" shape="same"/><Arrow/><N id="ropeq"/><Arrow/><Tensor name="Qᵣ" shape="[B,64,S,128]"/></div><Tensor name="positions" shape="[S]" external/></section>
      <section><header>K BRANCH</header><div className="mini-flow"><Tensor name="K" shape="[B,4,T,128]"/><Arrow/><N id="knorm"/><Arrow/><Tensor name="K̃" shape="same"/><Arrow/><N id="ropek"/><Arrow/><Tensor name="Kᵣ" shape="[B,4,T,128]"/></div><Tensor name="positions" shape="[T]" external/></section>
      <section><header>KV MEMORY</header><div className="mini-flow"><Tensor name="Kᵣ · V · block_table" shape="KV + [B,Nblocks]"/><Arrow/><N id="cache"/><Arrow/><Tensor name="Kcache · Vcache" shape="2 × [B,4,T,128]"/></div></section>
    </div>
    <div className="flow-row attention-row"><Tensor name="Qᵣ · Kcache" shape="Q [B,64,S,128] · K [B,4,T,128]"/><Arrow/><N id="qk"/><Arrow/><Tensor name="A" shape="[B,64,S,T]"/><Arrow/><N id="scale"/><Arrow/><Tensor name="A / √128" shape="[B,64,S,T]"/><Arrow/><N id="mask"/><Arrow/><Tensor name="A masked" shape="[B,64,S,T]"/><Arrow/><N id="softmax"/><Arrow/><Tensor name="P" shape="[B,64,S,T]"/></div>
    <div className="attached-input"><Tensor name="causal mask" shape="[S,T]" external/><span>↳ 输入到 + Causal Mask</span></div>
    <div className="flow-row"><Tensor name="P · Vcache" shape="P [B,64,S,T] · V [B,4,T,128]"/><Arrow/><N id="pv"/><Arrow/><Tensor name="heads" shape="[B,S,8192]"/><Arrow/><N id="oproj"/><Arrow/><Tensor name="Yattn · Xₗ" shape="2 × [B,S,6144]"/><Arrow/><N id="add1"/><Arrow/><Tensor name="U" shape="[B,S,6144]"/></div>
    <div className="flow-row"><Tensor name="U" shape="[B,S,6144]"/><Arrow/><N id="postnorm"/><Arrow/><Tensor name="Û" shape="[B,S,6144]"/><Arrow/><N id="gateup"/><Arrow/><Tensor name="gate · up" shape="2 × [B,S,12288]"/><Arrow/><N id="swiglu"/><Arrow/><Tensor name="activated" shape="[B,S,12288]"/><Arrow/><N id="down"/><Arrow/><Tensor name="Yffn · U" shape="2 × [B,S,6144]"/><Arrow/><N id="add2"/><Arrow/><Tensor name="Xₗ₊₁" shape="[B,S,6144]"/></div>
  </div>;
}

function SparseDiagram({g,active,onHover,onLeave,onSelect}:{g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect}; const N=({id}:{id:string})=><Op node={g[id]} {...p} active={active===g[id].id}/>;
  return <div className="operator-diagram sparse-diagram">
    <div className="flow-row"><Tensor name="Xₗ" shape="[B,S,6144]"/><Arrow/><N id="norm"/><Arrow/><Tensor name="X̂" shape="[B,S,6144]"/><Arrow/><N id="packed"/><Arrow/><Tensor name="packed_5" shape="[B,S,9856]"/><Arrow/><N id="split"/><Arrow/><Tensor name="Q · K · V · Qidx · Kidx" shape="8192 · 512 · 512 · 512 · 128"/></div>
    <div className="dual-path">
      <section><header>INDEX PATH · 只决定读哪些 block</header><div className="mini-flow"><Tensor name="Qidx · Kidx" shape="[B,4,S,128] · [B,1,T,128]"/><Arrow/><N id="idxnorm"/><Arrow/><Tensor name="Q̃idx · K̃idx" shape="same"/><Arrow/><N id="idxscore"/><Arrow/><Tensor name="token scores" shape="[B,4,S,T]"/><Arrow/><N id="blockmax"/><Arrow/><Tensor name="block scores" shape="[B,4,S,⌈T/128⌉]"/><Arrow/><N id="topk"/><Arrow/><Tensor name="Top-16 block ids" shape="[B,S,4,16]"/></div><Tensor name="causal mask · local priority" shape="routing metadata" external/></section>
      <section><header>MAIN PATH · 生成精确 attention 输入</header><div className="mini-flow"><Tensor name="Q · K" shape="[B,64,S,128] · [B,4,T,128]"/><Arrow/><N id="mainnorm"/><Arrow/><Tensor name="Q̃ · K̃" shape="same"/><Arrow/><N id="rope"/><Arrow/><Tensor name="Qᵣ · Kᵣ" shape="same"/></div><Tensor name="positions" shape="[S]" external/></section>
    </div>
    <div className="flow-row"><Tensor name="Kᵣ · V · block_table" shape="KV + [B,Nblocks]"/><Arrow/><N id="cache"/><Arrow/><Tensor name="paged K · V + Top-16 ids" shape="KV pages + [B,S,4,16]"/><Arrow/><N id="select"/><Arrow/><Tensor name="selected K · V" shape="≤2048 tokens / group"/></div>
    <div className="flow-row attention-row"><Tensor name="Qᵣ · selected K" shape="Q · Kselected"/><Arrow/><N id="qk"/><Arrow/><Tensor name="sparse scores" shape="[B,64,S,≤2048]"/><Arrow/><N id="scale"/><Arrow/><Tensor name="scaled scores" shape="same"/><Arrow/><N id="mask"/><Arrow/><Tensor name="masked scores" shape="same"/><Arrow/><N id="softmax"/><Arrow/><Tensor name="P" shape="same"/></div>
    <div className="flow-row"><Tensor name="P · selected V" shape="probabilities · Vselected"/><Arrow/><N id="pv"/><Arrow/><Tensor name="heads" shape="[B,S,8192]"/><Arrow/><N id="oproj"/><Arrow/><Tensor name="Yattn · Xₗ" shape="2 × [B,S,6144]"/><Arrow/><N id="addattn"/><Arrow/><Tensor name="U" shape="[B,S,6144]"/></div>
    <div className="flow-row moe-path"><Tensor name="U" shape="[B,S,6144]"/><Arrow/><N id="router"/><Arrow/><Tensor name="expert ids · weights" shape="Top-4 / token"/><Arrow/><div className="parallel-ops"><N id="experts"/><N id="shared"/></div><Arrow/><Tensor name="4 routed · 1 shared" shape="5 × [B,S,6144]"/><Arrow/><N id="sum"/><Arrow/><Tensor name="Ymoe · U" shape="2 × [B,S,6144]"/><Arrow/><N id="addout"/><Arrow/><Tensor name="Xₗ₊₁" shape="[B,S,6144]"/></div>
  </div>;
}
/* eslint-enable react-hooks/static-components */

function LayerNavigator({layer,onChange}:{layer:number;onChange:(n:number)=>void}){
  const ticksRef=useRef<HTMLDivElement>(null);
  useEffect(()=>{ticksRef.current?.querySelector(".active")?.scrollIntoView({behavior:"smooth",block:"nearest",inline:"center"})},[layer]);
  return <div className="layer-nav"><div className="layer-nav-head"><div><span>DECODER LAYER</span><b>L{layer}</b><small>{layer<3?"Dense GQA + Dense MLP":"MSA + Top-4 MoE"}</small></div><div className="layer-type-legend"><span><i className="dense"/>Dense · L0–2</span><span><i className="sparse"/>MSA+MoE · L3–59</span></div></div><div className="layer-ticks" ref={ticksRef}>{Array.from({length:60},(_,i)=><button key={i} className={`${i<3?"dense":"sparse"} ${i===layer?"active":""}`} onClick={()=>onChange(i)} title={`L${i} · ${i<3?"Dense":"MSA+MoE"}`}>{i}</button>)}</div><div className="layer-slider"><span>L0</span><input type="range" min="0" max="59" value={layer} onChange={e=>onChange(Number(e.target.value))}/><span>L59</span></div></div>;
}

function DetailPanel({node,tab,setTab}:{node:OpNode;tab:Tab;setTab:(t:Tab)=>void}){
  const tabs:[Tab,string][]=[["io","I/O"],["formula","公式"],["code","代码"],["weights","权重"]];
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div><i className={`kind-dot op-${node.kind}`}/><p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div className="detail-content">
    {tab==="io"&&<div className="shape-view"><article><span>INPUT</span><b>{node.input}</b><code>{node.inputShape}</code></article><i>→</i><article><span>OUTPUT</span><b>{node.output}</b><code>{node.outputShape}</code></article></div>}
    {tab==="formula"&&<div className="formula-view"><span>COMPUTE</span><code>{node.formula}</code><p>{node.formulaNote}</p></div>}
    {tab==="code"&&<div className="code-view"><a href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE</span><b>{node.source}</b><i>↗</i></a><pre><code>{node.code}</code></pre></div>}
    {tab==="weights"&&<WeightView weights={node.weights}/>}</div><footer>vLLM @ {VLLM_COMMIT.slice(0,7)} · official safetensors</footer></aside>;
}

function WeightView({weights}:{weights:Weight[]}){return weights.length?<div className="weight-view">{weights.map(w=><article key={w.key}><code>{w.key}</code><div><b>{w.dtype}</b><span>{w.shape}</span>{w.params&&<em>{w.params}</em>}</div><small>{w.shard}</small>{w.runtime&&<small>→ {w.runtime}</small>}</article>)}</div>:<div className="empty-weight"><b>无可训练权重</b><p>这是 shape、mask、缓存、路由选择或逐元素计算。</p></div>}

function HelpModal({onClose}:{onClose:()=>void}){
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="help-modal" onMouseDown={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="参数、符号与映射说明"><header><div><span>REFERENCE</span><h2>参数、符号与运行时映射</h2></div><button onClick={onClose} aria-label="关闭">×</button></header><div className="help-grid"><section><h3>Shape 符号</h3><table><tbody><tr><th>B</th><td>batch size</td><th>S</th><td>当前 query token 数</td></tr><tr><th>T</th><td>含历史 cache 的 KV 长度</td><th>H</th><td>hidden size = 6144</td></tr><tr><th>d</th><td>head dim = 128</td><th>V</th><td>vocab = 200064</td></tr><tr><th>E</th><td>routed experts = 128</td><th>K</th><td>Top-K experts = 4</td></tr></tbody></table></section><section><h3>节点与底色</h3><div className="node-rule"><i className="tensor-swatch"/><b>张量 / 产物</b><span>→</span><i className="operator-swatch"/><b>计算算子</b><span>→ 新张量</span></div><p className="node-rule-note">白色虚线框是数据，彩色实线框是运算；支路输入也必须连到消费它的算子。</p><div className="color-legend">{[["norm","Norm"],["linear","Linear"],["matmul","MatMul"],["rope","RoPE"],["scale","Scale"],["mask","Mask"],["activation","Activation"],["route","Routing"],["cache","Cache"],["add","Residual Add"]].map(([kind,label])=><span key={kind}><i className={`op-${kind}`}/>{label}</span>)}</div></section><section className="mapping-section"><h3>Checkpoint → vLLM runtime</h3><div className="mapping-table"><div><code>q_proj · k_proj · v_proj</code><span>→</span><b>QKVParallelLinear.qkv_proj</b></div><div><code>q/k/v + index_q/index_k</code><span>→</span><b>MinimaxM3QKV…WithIndexer</b></div><div><code>gate_proj · up_proj</code><span>→</span><b>gate_up_proj</b></div><div><code>experts.*.w1 · w3 · w2</code><span>→</span><b>FusedMoE w13 · w2</b></div></div></section></div></section></div>;
}

export default function Home(){
  const [layer,setLayer]=useState(3); const [tab,setTab]=useState<Tab>("io"); const [dark,setDark]=useState(false); const [help,setHelp]=useState(false);
  const graph=layer<3?denseGraph(layer):sparseGraph(layer); const [pinned,setPinned]=useState<OpNode>(sparseGraph(3).packed); const [hovered,setHovered]=useState<OpNode|null>(null); const active=hovered??pinned;
  const changeLayer=(next:number)=>{const g=next<3?denseGraph(next):sparseGraph(next);setLayer(next);setPinned(next<3?g.qkv:g.packed);setHovered(null)};
  const vision=visionNodes;
  return <main className={`atlas-app ${dark?"dark":""}`}><header className="app-header">
    <label className="model-select"><span>MODEL</span><select aria-label="选择模型" value="minimax-m3" onChange={()=>undefined}>{MODEL_REGISTRY.map(m=><option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}</option>)}</select></label>
    <div className="brand-lockup"><span className="brand-glyph"><i/><i/><i/></span><div><b>模型结构概览</b><small>MiniMax-M3</small></div></div>
    <nav className="resource-links"><a href={CODE_URL} target="_blank" rel="noreferrer"><b>CODE ↗</b><small>vLLM @ {VLLM_COMMIT.slice(0,7)}</small></a><a href={WEIGHTS_URL} target="_blank" rel="noreferrer"><b>WEIGHTS ↗</b><small>Hugging Face · 59 shards</small></a></nav>
    <div className="model-facts"><span><b>428B</b><small>模型总参数量</small></span><span><b>23B</b><small>每 token 激活参数</small></span><span><b>1M</b><small>最大上下文 token</small></span><span><b>869 GB</b><small>BF16 checkpoint</small></span></div>
    <button className="help-button" onClick={()=>setHelp(true)} aria-label="查看参数和符号说明">?</button><button className="theme-button" onClick={()=>setDark(v=>!v)} aria-label="切换明暗主题">{dark?"☀":"☾"}</button>
  </header><div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><button onMouseEnter={()=>setHovered(cloneOp(vision[0],{id:"overview-input",kind:"io",title:"Text / Vision Inputs"}))} onMouseLeave={()=>setHovered(null)}>Text / Vision Inputs</button><Arrow/><button onMouseEnter={()=>setHovered(cloneOp(vision[4],{id:"overview-fusion",kind:"linear",title:"Embedding Fusion"}))} onMouseLeave={()=>setHovered(null)}>Embedding Fusion <code>[B,S,6144]</code></button><Arrow/><div className="overview-stack"><b>Decoder ×60</b><span><i className="dense"/>Dense ×3</span><span><i className="sparse"/>MSA+MoE ×57</span></div><Arrow/><button onMouseEnter={()=>setHovered((layer<3?denseGraph(layer).add2:sparseGraph(layer).addout))} onMouseLeave={()=>setHovered(null)}>Final Norm → LM Head <code>[B,S,200064]</code></button></div>
    <LayerNavigator layer={layer} onChange={changeLayer}/>
    <section className="layer-canvas"><header><div><span>SELECTED LAYER DETAIL</span><h1>L{layer} · {layer<3?"Dense GQA + Dense MLP":"MiniMax Sparse Attention + MoE"}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>张量 / 产物</span><span><i className="operator-swatch"/>计算算子</span><code>{layerShard(layer)}</code></div></header>{layer<3?<DenseDiagram g={graph} active={active.id} onHover={setHovered} onLeave={()=>setHovered(null)} onSelect={setPinned}/>:<SparseDiagram g={graph} active={active.id} onHover={setHovered} onLeave={()=>setHovered(null)} onSelect={setPinned}/>}</section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab}/></div>{help&&<HelpModal onClose={()=>setHelp(false)}/>}</main>;
}
