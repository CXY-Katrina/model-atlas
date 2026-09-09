import katex from "katex";
import type { OpNode, Tab, IoBinding, BindingKind, CodeSection } from "../types";
import type { LayerType, ExpandedStage } from "./types";
import { SIMPLE_FORMULA, FORMULA_NOTE, FORMULA_STEPS_BY_ID, formulaTerms } from "./formulas";
import { INPUT_OVERRIDES, NEXT_BY_ID } from "./evidence";
import { VLLM_COMMIT } from "./sources";
import { pinSource } from "./operators";

function LatexExpression({formula,label,className=""}:{formula:string;label:string;className?:string}){
  const html=katex.renderToString(formula,{displayMode:true,throwOnError:false,strict:"ignore",output:"htmlAndMathml"});
  return <div className={`latex-render ${className}`.trim()} aria-label={label} dangerouslySetInnerHTML={{__html:html}}/>;
}

function LatexFormula({node}:{node:OpNode}){
  const formula=node.latex??SIMPLE_FORMULA[node.kind]??String.raw`y=f(x)`;
  return <LatexExpression formula={formula} label={`${node.title} 简化公式`}/>;
}

function symbolicShape(shape:string){
  return shape
    .replaceAll("[B,64/TP,S,128]","[B,Nₕ/TP,S,Dₕ]")
    .replaceAll("[B,max(1,4/TP),T,128]","[B,Nₖᵥ,rank,T,Dₕ]")
    .replaceAll("[B,max(1,4/TP),S,128]","[B,N_idx,rank,S,D_idx]")
    .replaceAll("[B,64/TP,S,T]","[B,Nₕ/TP,S,T]")
    .replaceAll("[B,64/TP,S,Ksel]","[B,Nₕ/TP,S,Ksel]")
    .replaceAll("[B,64/TP,S,≤2048]","[B,Nₕ/TP,S,Ksel]")
    .replaceAll("[B,S,8192/TP]","[B,S,Nₕ·Dₕ/TP]")
    .replaceAll("[B,64,S,128]","[B,Nₕ,S,Dₕ]")
    .replaceAll("[B,64,S,T]","[B,Nₕ,S,T]")
    .replaceAll("[B,4,S,128]","[B,Nₖᵥ,S,Dₕ]")
    .replaceAll("[B,4,T,128]","[B,Nₖᵥ,T,Dₕ]")
    .replaceAll("[B,S,24576/TP]","[B,S,2H_dense/TP]")
    .replaceAll("[B,S,12288/TP]","[B,S,H_dense/TP]")
    .replaceAll("[B,S,24576]","[B,S,2H_dense]")
    .replaceAll("[B,S,12288]","[B,S,H_dense]")
    .replaceAll("[B,S,6144]","[B,S,H]")
    .replaceAll("[B,S,8192]","[B,S,Nₕ·Dₕ]")
    .replaceAll("[B,S,9216]","[B,S,(Nₕ+2Nₖᵥ)·Dₕ]")
    .replaceAll("[B,S,9856]","[B,S,(Nₕ+2Nₖᵥ)·Dₕ+(N_idx+1)·D_idx]")
    .replaceAll("24576/TP","2H_dense/TP")
    .replaceAll("12288/TP","H_dense/TP")
    .replaceAll("8192/TP","Nₕ·Dₕ/TP")
    .replaceAll("max(1,4/TP)","Nₖᵥ,rank")
    .replaceAll("64/TP","Nₕ/TP")
    .replaceAll("24576","2H_dense")
    .replaceAll("12288","H_dense")
    .replaceAll("6144","H")
    .replaceAll("8192","Nₕ·Dₕ")
    .replaceAll("9216","(Nₕ+2Nₖᵥ)·Dₕ")
    .replaceAll("9856","(Nₕ+2Nₖᵥ)·Dₕ+(N_idx+1)·D_idx")
    .replaceAll("200064","V");
}

function ShapeRows({shape}:{shape:string}){
  return <div className="shape-rows"><span><i>符号</i><code title={symbolicShape(shape)}>{symbolicShape(shape)}</code></span><span><i>实际</i><code title={shape}>{shape}</code></span></div>;
}

function bindingsFor(node:OpNode):IoBinding[]{
  const dataInputs=INPUT_OVERRIDES[node.id]??[{kind:node.kind==="io"?"external":"upstream",label:node.input,shape:node.inputShape,from:node.kind==="io"?"模型调用方 / runtime":"图中紧邻的上游模块输出"}];
  const weightInputs=node.weights.map(weight=>{
    const tpShape=node.id==="d-gateup"?weight.shape.replace("[12288,6144]","[12288/TP,6144]"):node.id==="d-down"?weight.shape.replace("[6144,12288]","[6144,12288/TP]"):null;
    return {kind:"weight" as const,label:weight.key,shape:tpShape?`${weight.dtype} · TP shard ${tpShape}`:`${weight.dtype} · ${weight.shape}`,from:weight.runtime?`checkpoint → ${weight.runtime}`:`checkpoint · ${weight.shard}`,note:weight.params?`${weight.params} parameters`:undefined};
  });
  return [...dataInputs,...weightInputs];
}

function IoView({node}:{node:OpNode}){
  const bindings=bindingsFor(node);
  const labels:Record<BindingKind,string>={upstream:"上游张量",external:"外部输入",weight:"权重输入"};
  return <div className="io-binding-view"><section className="binding-list"><header><span>INPUT BINDINGS</span><b>{bindings.length} 路输入</b></header>{bindings.map((binding,index)=><article className={`binding binding-${binding.kind}`} key={`${binding.kind}-${binding.label}-${index}`}><div><span>{labels[binding.kind]}</span></div><b>{binding.label}</b><ShapeRows shape={binding.shape}/><p><i>来自</i>{binding.from}</p>{binding.note&&<small>{binding.note}</small>}</article>)}</section><section className="output-binding"><header><span>OUTPUT BINDING</span><b>1 路产物</b></header><article><div><span>计算产物</span></div><b>{node.output}</b><ShapeRows shape={node.outputShape}/><p><i>送往</i>{NEXT_BY_ID[node.id]??"图中下游模块"}</p></article></section></div>;
}

function codeSourceLabel(section:CodeSection){
  return section.url?.includes("github.com/huggingface/transformers")?"Transformers":"vLLM";
}

function CodeView({node}:{node:OpNode}){
  const sections=node.codeSections??[];
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {VLLM_COMMIT.slice(0,7)}</span><b>{node.source}</b><i>↗</i></a>
    {sections.length?<section className="code-call-chain"><header><span>IMPLEMENTATION TRACE</span></header>{sections.map((section,index)=>{const source=codeSourceLabel(section);return <article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><div className="code-section-kicker"><span>{section.stage}</span><span className={`code-source-tag source-${source.toLowerCase()}`}>{source}</span></div><b>{section.title}</b><small>{section.location}</small></div>{section.url&&<a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article>})}</section>:<div className="code-empty"><b>此节点没有独立 forward</b><p>它由所在模块的 forward 调度，或只是一个数学拆解步骤。</p></div>}
  </div>;
}

type StageOverview = { kicker:string; title:string; summary:string; flow:string; formula:string; formulaNote?:string; notes:string[]; parameters:readonly (readonly [string,string,string])[] };

function stageOverview(type:LayerType,stage:Exclude<ExpandedStage,null>):StageOverview{
  if(type==="dense"&&stage==="ffn")return {
    kicker:"DENSE FFN · L0–L2",title:"SwiGLU-OAI MLP",summary:"逐 token 扩维、门控，再投回 hidden size。",
    flow:"Û → TP-local Gate + Up Projection → Split → SwiGLU-OAI → RowParallel Down Projection → Yffn",
    formula:"G⁽ʳ⁾ = Û (W_gate⁽ʳ⁾)ᵀ\nU⁽ʳ⁾ = Û (W_up⁽ʳ⁾)ᵀ\nḠ⁽ʳ⁾ = min(G⁽ʳ⁾, c)\nŪ⁽ʳ⁾ = clip(U⁽ʳ⁾, −c, c)\nZ⁽ʳ⁾ = Ḡ⁽ʳ⁾ ⊙ σ(αḠ⁽ʳ⁾) ⊙ (Ū⁽ʳ⁾ + β)\nYffn = Σᵣ Z⁽ʳ⁾ (W_down⁽ʳ⁾)ᵀ",
    formulaNote:"r 表示 TP rank；G⁽ʳ⁾、U⁽ʳ⁾、Z⁽ʳ⁾ 的最后一维都是 H_dense/TP。σ 表示 sigmoid，⊙ 表示逐元素相乘；Down Projection 最后归并各 rank 的部分结果。",
    notes:["Gate 与 Up 权重和同一个输入 Û 直接进入 fused projection。","投影结果沿最后一维 Split；MLP 不混合不同 token。"],
    parameters:[["H","6144","hidden_size"],["H_dense","12288","dense_intermediate_size"],["α","1.702","swiglu_alpha"],["β","1.0","swiglu_beta"],["c","7.0","swiglu_limit"]],
  };
  if(stage==="attention")return type==="dense"?{
    kicker:"DENSE ATTENTION · L0–L2",title:"GQA + Partial RoPE",summary:"用 Q 检索完整可见 KV 历史，再按概率聚合 V。",
    flow:"QKV Projection → Split → Q/K Norm + Partial RoPE → Attention → O Projection",
    formula:"Attention(Q,K,V)=softmax(QKᵀ/√Dₕ + mask)V",
    notes:["Q / K / V 从 Split 节点分叉，并在 Attention 算子中汇合。","Partial RoPE 作用于 Q/K 每个 head 的前 64 维。"],
    parameters:[["Nₕ","64","query heads"],["Nₖᵥ","4","KV heads"],["Dₕ","128","head_dim"],["Dᵣ","64","rotary_dim"]],
  }:{
    kicker:"SPARSE ATTENTION · L3–L59",title:"MiniMax Sparse Attention + Partial RoPE",summary:"Indexer 先选 Top-16 KV blocks，主 Attention 再在候选页内计算。",
    flow:"QKV + Index Projection → Indexer → Top-16 Blocks → Sparse Attention → O Projection",
    formula:"Attention(Q,Kselected,Vselected)=softmax(QKselectedᵀ/√Dₕ + mask)Vselected",
    notes:["Top-16 只缩小候选 KV blocks，不替代 causal / padding mask。","Indexer 与主 Attention 通过明确的 KV page 边连接。"],
    parameters:[["K_block","16","selected blocks"],["B_block","128","tokens / block"],["N_idx","4","index heads"],["D_idx","128","index_dim"]],
  };
  return {
    kicker:"SPARSE FFN · L3–L59",title:"Top-4 MoE + Shared Expert",summary:"每个 token 进入 4 个路由专家，同时经过 1 个共享专家。",
    flow:"路由选择：Û → FP32 Router → router_logits\n路由专家：Û + router_logits → Fused Top-4 Routing + Experts → Y_routed\n共享专家：Û → Shared Expert → Y_shared\n最终合并：Y_routed + Y_shared → Add → Y_moe",
    formula:"Y_routed=Σₑ ŵₑEₑ(Û)\nY_shared=E_shared(Û)\nY_moe=Y_routed+Y_shared",
    notes:["Router 在 Python 层只产生 [B,S,128] 的 router_logits。","expert ids 与 router weights 由 FusedMoE 内部计算；Shared Expert 不经过 Top-K。"],
    parameters:[["E","128","routed experts"],["K","4","experts / token"],["E_shared","1","shared expert"],["H_expert","3072","expert width"]],
  };
}

function StageOverviewPanel({type,stage}:{type:LayerType;stage:Exclude<ExpandedStage,null>}){
  const overview=stageOverview(type,stage);
  return <aside className="detail-panel stage-overview-panel"><header className="stage-overview-header"><span>{overview.kicker}</span><h2>{overview.title}</h2><p>{overview.summary}</p></header><div className="stage-overview-body"><section className="stage-flow-section"><span>数据流</span><code>{overview.flow}</code></section><section className="stage-formula-section"><span>计算语义</span><code>{overview.formula}</code>{overview.formulaNote&&<p>{overview.formulaNote}</p>}</section><section className="stage-parameter-section"><span>关键参数</span><div className="stage-parameters">{overview.parameters.map(([symbol,value,source])=><article key={symbol}><b>{symbol}</b><strong>{value}</strong><small>{source}</small></article>)}</div></section><section><span>边界说明</span>{overview.notes.map(note=><p key={note}>{note}</p>)}</section></div><footer>展开图说明 · 点击算子查看独立详情</footer></aside>;
}

export function DetailPanel({node,tab,setTab,pinned,onClear,expanded,layerType}:{node:OpNode|null;tab:Tab;setTab:(t:Tab)=>void;pinned:boolean;onClear:()=>void;expanded:ExpandedStage;layerType:LayerType}){
  const tabs:[Tab,string][]=[["io","I/O + 权重"],["formula","公式"],["code","代码"]];
  if(!node&&expanded)return <StageOverviewPanel type={layerType} stage={expanded}/>;
  if(!node)return <aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后，可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;
  const formulaSteps=FORMULA_STEPS_BY_ID[node.id];
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div>{pinned?<button className="unpin-button" aria-label="取消固定" title="取消固定" onClick={onClear}>×</button>:<i className={`kind-dot op-${node.kind}`}/>}<p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id)}>{label}</button>)}</div><div key={tab} className={`detail-content detail-${tab}`}>
    {tab==="io"&&<IoView node={node}/>}
    {tab==="formula"&&<div className="formula-view"><span>作用</span><div className="formula-purpose">{node.summary}</div><span>实际公式</span><LatexFormula node={node}/>{formulaSteps?<div className="formula-steps">{formulaSteps.map(step=><article className="formula-step" key={step.title}><header>{step.title}</header><LatexExpression formula={step.formula} label={`${step.title} 公式`} className="formula-step-math"/><p>{step.explanation}</p></article>)}</div>:<><div className="formula-implementation"><b>一句话解释</b><p>{node.formulaNote??FORMULA_NOTE[node.kind]}</p></div><div className="formula-terms">{formulaTerms(node).map(([symbol,meaning])=><span key={symbol}><b>{symbol}</b>{meaning}</span>)}</div></>}</div>}
    {tab==="code"&&<CodeView node={node}/>}
    </div><footer>vLLM @ {VLLM_COMMIT.slice(0,7)} · official safetensors</footer></aside>;
}
