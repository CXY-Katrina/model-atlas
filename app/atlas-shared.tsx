import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import katex from "katex";
import { routeGraphEdge } from "./graph-routing";

// Shared value types and operator-graph primitives used by every model module.
// Model-specific datasets and diagram layouts live in app/model-data.tsx plus
// app/page.tsx (MiniMax-M3) and app/models/<model>.tsx (additional models).

export type Tab = "io" | "formula" | "code";
export type OpKind = "io" | "norm" | "linear" | "split" | "rope" | "matmul" | "scale" | "mask" | "softmax" | "activation" | "route" | "cache" | "add";
export type BindingKind = "upstream" | "external" | "weight";
export type IoBinding = { kind: BindingKind; label: string; shape: string; from: string; note?: string };
export type CodeSection = { stage: string; title: string; location: string; code: string; url?: string };
export type CodeSymbol = { symbol: string; resolvesTo: string; meaning: string };
export type CodeDetail = { sections: CodeSection[]; symbols: CodeSymbol[] };
export type Weight = {
  key: string;
  shape: string;
  dtype: "BF16" | "F32";
  shard: string;
  runtime?: string;
  note?: string;
  params?: string;
};
export type Node = {
  id: string;
  tone: string;
  kicker: string;
  title: string;
  summary: string;
  input: string;
  inputShape: string;
  output: string;
  outputShape: string;
  formula: string;
  formulaNote: string;
  runtime: string;
  source: string;
  sourceUrl: string;
  code: string;
  weights: Weight[];
};
export type OpNode = Node & { kind: OpKind; latex?: string; codeSections?: CodeSection[]; codeSymbols?: CodeSymbol[] };
export type ExpandedStage = "attention" | "ffn" | null;
export type EdgePort = "top" | "top-left" | "top-right" | "right" | "bottom" | "left";
export type GraphEdge = { from: string; to: string; fromPort?: EdgePort; toPort?: EdgePort; route?: "side-left" | "side-right" };
export type TensorRole = "input" | "tensor" | "output" | "side" | "weight";
export type FormulaTerm = readonly [symbol: string, meaning: string];
export type StageOverview = { kicker: string; title: string; summary: string; flow: string; formula: string; formulaNote?: string; notes: string[]; parameters: readonly (readonly [string, string, string])[] };
export type ConfigGroup = { title: string; rows: readonly (readonly [string, string])[] };

export const SIMPLE_FORMULA: Partial<Record<OpKind, string>> = {
  norm: String.raw`y=\operatorname{Norm}(x)`, linear: String.raw`y=xW^{\mathsf T}`, split: String.raw`(a,b,\ldots)=\operatorname{Split}(x)`, rope: String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`, matmul: String.raw`y=a\,b^{\mathsf T}`, scale: String.raw`y=x/\sqrt{d_h}`, mask: String.raw`y=x+\mathrm{mask}`, softmax: String.raw`p=\operatorname{softmax}(x)`, activation: String.raw`y=\bar g\,\sigma(\alpha\bar g)\,(\bar u+\beta)`, route: String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`, cache: String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(K,V)`, add: String.raw`y=x+f(x)`, io: String.raw`y=x`,
};

export const FORMULA_NOTE_DEFAULT: Partial<Record<OpKind, string>> = {
  norm: "把每个 token 的向量缩放到稳定范围；shape 不变。", linear: "W 是当前模块绑定的权重；最后一维由 W 的输出维决定。", split: "只切分最后一维，不做数值计算，也没有权重。", rope: "position 决定旋转角度。", matmul: "沿共同的 head_dim 相乘并求和。", scale: "缩放避免 score 随维度增大。", mask: "不可见位置加 −∞，softmax 后概率变为 0。", softmax: "把每行 score 转为和为 1 的概率。", activation: "g 是 gate，u 是 up。", route: "只选择去哪里计算；Top-K 本身不生成 expert 输出。", cache: "slot 与 block table 由 runtime 提供，权重不参与。", add: "残差支路与计算支路逐元素相加，shape 必须一致。", io: "这是数据入口或运行时元数据，不执行可训练计算。",
};

export function checkpointWeightName(weight?: Weight) {
  return weight?.key.replace(/^(?:language_model\.)?model\.layers\.\d+\./, "") ?? "weight";
}

export function GraphSurface({ edges, className, children }: { edges: GraphEdge[]; className: string; children: ReactNode }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const markerId = `graph-arrow-${useId().replace(/:/g, "")}`;
  const serializedEdges = JSON.stringify(edges);
  const edgeKey = edges.map(edge => `${edge.from}:${edge.fromPort ?? "bottom"}>${edge.to}:${edge.toPort ?? "top"}:${edge.route ?? "direct"}`).join("|");
  const [paths, setPaths] = useState<string[]>([]);
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    let frame = 0;
    const point = (rect: DOMRect, port: EdgePort, rootRect: DOMRect) => {
      const x = rect.left - rootRect.left; const y = rect.top - rootRect.top;
      if (port === "top") return [x + rect.width / 2, y];
      if (port === "top-left") return [x + rect.width * .34, y];
      if (port === "top-right") return [x + rect.width * .66, y];
      if (port === "right") return [x + rect.width, y + rect.height / 2];
      if (port === "left") return [x, y + rect.height / 2];
      return [x + rect.width / 2, y + rect.height];
    };
    const measure = () => {
      const rootRect = root.getBoundingClientRect();
      const currentEdges = JSON.parse(serializedEdges) as GraphEdge[];
      const nodeRects = [...root.querySelectorAll<HTMLElement>("[data-graph-id]")].map(node => node.getBoundingClientRect());
      const obstacleBounds = {
        left: Math.min(...nodeRects.map(rect => rect.left - rootRect.left)),
        right: Math.max(...nodeRects.map(rect => rect.right - rootRect.left)),
      };
      const next = currentEdges.flatMap(edge => {
        const source = root.querySelector<HTMLElement>(`[data-graph-id="${edge.from}"]`);
        const target = root.querySelector<HTMLElement>(`[data-graph-id="${edge.to}"]`);
        if (!source || !target) return [];
        const fromPort = edge.fromPort ?? "bottom"; const toPort = edge.toPort ?? "top";
        const [sx, sy] = point(source.getBoundingClientRect(), fromPort, rootRect);
        const [tx, ty] = point(target.getBoundingClientRect(), toPort, rootRect);
        const direction = edge.route ?? (fromPort === "right" || fromPort === "left" || toPort === "right" || toPort === "left" ? "horizontal" : "vertical");
        const safeClearance=direction==="side-left"
          ?Math.min(24,Math.max(4,obstacleBounds.left-8))
          :direction==="side-right"
            ?Math.min(24,Math.max(4,rootRect.width-obstacleBounds.right-8))
            :24;
        return [routeGraphEdge({ source: { x: sx, y: sy }, target: { x: tx, y: ty }, direction, obstacleBounds, clearance: safeClearance }).path];
      });
      setPaths(next);
    };
    const observer = new ResizeObserver(() => { cancelAnimationFrame(frame); frame = requestAnimationFrame(measure) });
    observer.observe(root);
    root.querySelectorAll<HTMLElement>("[data-graph-id]").forEach(node => observer.observe(node));
    frame = requestAnimationFrame(measure);
    return () => { cancelAnimationFrame(frame); observer.disconnect() };
  }, [serializedEdges]);
  return <div ref={rootRef} className={`graph-surface ${className}`}>{children}<svg className="graph-connectors" aria-hidden="true"><defs><marker id={markerId} markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 8 4 L 0 8 Z" /></marker></defs>{paths.map((path, index) => <path key={`${edgeKey}-connector-${index}`} d={path} markerEnd={`url(#${markerId})}`} />)}</svg></div>;
}

export function Op({ node, active, onHover, onLeave, onSelect, graphId }: { node: OpNode; active: boolean; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; graphId?: string }) {
  return <button data-graph-id={graphId} className={`op-node op-${node.kind} ${active ? "active" : ""}`} aria-pressed={active} onMouseEnter={() => onHover(node)} onMouseLeave={onLeave} onFocus={() => onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event => { if (event.detail === 0) onSelect(node) }}><small>OP · {node.kind}</small><b>{node.title}</b></button>;
}

export function Tensor({ name, shape, role = "tensor", graphId }: { name: string; shape: string; role?: TensorRole; graphId?: string }) {
  const label = { input: "TENSOR", tensor: "TENSOR", output: "TENSOR", side:"EXTERNAL", weight: "WEIGHT" }[role];
  return <div data-graph-id={graphId} className={`tensor-node tensor-${role}`}><small>{label}</small><b>{name}</b><code>{shape}</code></div>;
}

export const Arrow = ({ label }: { label?: string }) => <span className="op-arrow"><i />{label && <small>{label}</small>}</span>;

export function InputWeightedOp({ node, active, onHover, onLeave, onSelect, inputName, inputShape, weightIndex = 0, inputGraphId, graphId, weightGraphId, className = "", weightPrefix = "" }: { node: OpNode; active: boolean; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; inputName: string; inputShape: string; weightIndex?: number; inputGraphId: string; graphId: string; weightGraphId: string; className?: string; weightPrefix?: string }) {
  const weight = node.weights[weightIndex];
  const symbolicWeightShape = weight?.shape.replaceAll("6144", "H") ?? "[H]";
  return <div className={`input-weighted-op ${className}`}><div className="co-input-row"><Tensor name={inputName} shape={inputShape} graphId={inputGraphId} /><Tensor name={`${weightPrefix}${checkpointWeightName(weight)}`} shape={symbolicWeightShape} role="weight" graphId={weightGraphId} /></div><Op node={node} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} graphId={graphId} /></div>;
}

export function AddCircle({ node, active, onHover, onLeave, onSelect, graphId }: { node: OpNode; active: boolean; onHover: (n: OpNode) => void; onLeave: () => void; onSelect: (n: OpNode) => void; graphId?: string }) {
  return <button data-graph-id={graphId} className={`add-circle ${active ? "active" : ""}`} aria-label={node.title} aria-pressed={active} title={node.title} onMouseEnter={() => onHover(node)} onMouseLeave={onLeave} onFocus={() => onHover(node)} onBlur={onLeave} onPointerDown={()=>onSelect(node)} onClick={event => { if (event.detail === 0) onSelect(node) }}>+</button>;
}

export function RuntimeIORail({ N }: { N: ({ id }: { id: string }) => ReactNode }) {
  return <section className="runtime-io"><header><b>ATTENTION RUNTIME I/O</b><span>这些输入由 vLLM runner 生成并传入模型；mask 在内核中按边界隐式执行</span></header><div className="runtime-io-grid">
    <div className="io-lane"><Tensor name="num_computed_tokens · query offsets" shape="[B] + [Nq]" role="input" /><Arrow /><N id="position" /><Arrow /><Tensor name="positions → RoPE" shape="[Nq]" /></div>
    <div className="io-lane"><Tensor name="query_start_loc · seq_lens · causal" shape="[B+1] · [B] · True" role="input" /><Arrow /><N id="attnmeta" /><Arrow /><Tensor name="causal / padding layout → Attention" shape="implicit · 非稠密 [S,T]" /></div>
    <div className="io-lane"><Tensor name="positions · block_table" shape="[Nq] + [B,Nblocks]" role="input" /><Arrow /><N id="slots" /><Arrow /><Tensor name="slot_mapping · block_table → KV Cache" shape="[Nq] + [B,Nblocks]" /></div>
  </div></section>;
}

export function LatexFormula({ node }: { node: OpNode }) {
  const formula = node.latex??SIMPLE_FORMULA[node.kind] ?? String.raw`y=f(x)`;
  const html = katex.renderToString(formula, { displayMode: true, throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
  return <div className="latex-render" aria-label={`${node.title} 简化公式`} dangerouslySetInnerHTML={{ __html: html }} />;
}
