import type { OpNode } from "../models/types";
import type { CSSProperties } from "react";

// The variable lives on GraphSurface so alignment survives node remounts on hover.
function alignmentStyle(graphId?: string): CSSProperties | undefined {
  return graphId ? { translate: `var(--graph-shift-${graphId}, 0px) 0` } : undefined;
}

export function Op({node,active,onHover,onLeave,onSelect,graphId}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;graphId?:string}){
  return <button data-graph-id={graphId} style={alignmentStyle(graphId)} className={`op-node op-${node.kind} ${active?"active":""}`} aria-pressed={active} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event=>{if(event.detail===0)onSelect(node)}}><small>OP · {node.kind}</small><b>{node.title}</b></button>;
}

export type TensorRole = "input" | "tensor" | "output" | "side" | "weight";

export function Tensor({name,shape,role="tensor",graphId}:{name:string;shape:string;role?:TensorRole;graphId?:string}){
  const label={input:"TENSOR",tensor:"TENSOR",output:"TENSOR",side:"EXTERNAL",weight:"WEIGHT"}[role];
  return <div data-graph-id={graphId} style={alignmentStyle(graphId)} className={`tensor-node tensor-${role}`}><small>{label}</small><b>{name}</b><code>{shape}</code></div>;
}

export const Arrow=({label}:{label?:string})=><span className="op-arrow"><i/>{label&&<small>{label}</small>}</span>;

export function AddCircle({node,active,onHover,onLeave,onSelect,graphId}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;graphId?:string}){
  return <button data-graph-id={graphId} style={alignmentStyle(graphId)} className={`add-circle ${active?"active":""}`} aria-label={node.title} aria-pressed={active} title={node.title} onMouseEnter={()=>onHover(node)} onMouseLeave={onLeave} onFocus={()=>onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event=>{if(event.detail===0)onSelect(node)}}>+</button>;
}
