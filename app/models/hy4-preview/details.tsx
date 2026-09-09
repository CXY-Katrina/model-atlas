import katex from "katex";
import type { OpNode, Tab, IoBinding, BindingKind, CodeSection } from "../types";
import type { ExpandedStage, LayerVariant } from "./types";
import { SIMPLE_FORMULA, FORMULA_NOTE, FORMULA_STEPS_BY_ID, formulaTerms } from "./formulas";
import { INPUT_OVERRIDES, NEXT_BY_ID } from "./evidence";
import { VLLM_COMMIT, CONFIG_FETCH_NOTE } from "./sources";
import { pinSource } from "./operators";

function LatexExpression({ formula, label, className = "" }: { formula: string; label: string; className?: string }) {
  const html = katex.renderToString(formula, { displayMode: true, throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
  return <div className={`latex-render ${className}`.trim()} aria-label={label} dangerouslySetInnerHTML={{ __html: html }} />;
}

function LatexFormula({ node }: { node: OpNode }) {
  const formula = node.latex ?? SIMPLE_FORMULA[node.kind] ?? String.raw`y=f(x)`;
  return <LatexExpression formula={formula} label={`${node.title} 简化公式`} />;
}

function symbolicShape(shape: string) {
  return shape
    .replaceAll("[T,2048] int32", "[T,T_idx] int32")
    .replaceAll("[64,≤2048]", "[64,≤T_idx]")
    .replaceAll("每 query 2048", "每 query T_idx")
    .replaceAll("候选 2048", "候选 T_idx")
    .replaceAll("[T,2624]", "[T,R_q+C_kv+D_rope]")
    .replaceAll("[T,4,6144]", "[T,C_hc,H]")
    .replaceAll("[T,24576]", "[T,C_hc·H]")
    .replaceAll("[T,16384]", "[T,N_h·D_v]")
    .replaceAll("[T,36864]", "[T,2·H_ffn]")
    .replaceAll("[T,18432]", "[T,H_ffn]")
    .replaceAll("[T,2048]", "[T,R_q]")
    .replaceAll("[T,6144]", "[T,H]")
    .replaceAll("[T,512]", "[T,C_kv]")
    .replaceAll("[T,64,256]", "[T,N_h,D_v]")
    .replaceAll("16384", "N_h·D_v")
    .replaceAll("24576", "C_hc·H")
    .replaceAll("36864", "2·H_ffn")
    .replaceAll("18432", "H_ffn")
    .replaceAll("6144", "H")
    .replaceAll("512", "C_kv")
    .replaceAll("2048", "R_q");
}

function ShapeRows({ shape }: { shape: string }) {
  return <div className="shape-rows"><span><i>符号</i><code title={symbolicShape(shape)}>{symbolicShape(shape)}</code></span><span><i>实际</i><code title={shape}>{shape}</code></span></div>;
}

function bindingsFor(node: OpNode): IoBinding[] {
  const dataInputs = INPUT_OVERRIDES[node.id] ?? [{ kind: node.kind === "io" ? "external" as const : "upstream" as const, label: node.input, shape: node.inputShape, from: node.kind === "io" ? "模型调用方 / runtime" : "图中紧邻的上游模块输出" }];
  const weightInputs = node.weights.map(weight => ({ kind: "weight" as const, label: weight.key, shape: `${weight.dtype} · ${weight.shape}`, from: weight.runtime ? `checkpoint → ${weight.runtime}` : `checkpoint · ${weight.shard}`, note: weight.params ? `${weight.params} parameters` : undefined }));
  return [...dataInputs, ...weightInputs];
}

function IoView({ node }: { node: OpNode }) {
  const bindings = bindingsFor(node);
  const labels: Record<BindingKind, string> = { upstream: "上游张量", external: "外部输入", weight: "权重输入" };
  return <div className="io-binding-view"><section className="binding-list"><header><span>INPUT BINDINGS</span><b>{bindings.length} 路输入</b></header>{bindings.map((binding, index) => <article className={`binding binding-${binding.kind}`} key={`${binding.kind}-${binding.label}-${index}`}><div><span>{labels[binding.kind]}</span></div><b>{binding.label}</b><ShapeRows shape={binding.shape} /><p><i>来自</i>{binding.from}</p>{binding.note && <small>{binding.note}</small>}</article>)}</section><section className="output-binding"><header><span>OUTPUT BINDING</span><b>1 路产物</b></header><article><div><span>计算产物</span></div><b>{node.output}</b><ShapeRows shape={node.outputShape} /><p><i>送往</i>{NEXT_BY_ID[node.id] ?? "图中下游模块"}</p></article></section></div>;
}

function codeSourceLabel(section: CodeSection) {
  return section.url?.includes("github.com/vllm-project/vllm/blob/b2f685834a6456197e7033966fdef52a23f1abcd") ? "vLLM" : "vLLM";
}

function CodeView({ node }: { node: OpNode }) {
  const sections = node.codeSections ?? [];
  return <div className="code-view">
    <a className="code-source" href={pinSource(node.sourceUrl)} target="_blank" rel="noreferrer"><span>PINNED SOURCE · {VLLM_COMMIT.slice(0, 7)}</span><b>{node.source}</b><i>↗</i></a>
    {sections.length ? <section className="code-call-chain"><header><span>IMPLEMENTATION TRACE</span></header>{sections.map((section, index) => { const source = codeSourceLabel(section); return <article className="code-section" key={`${node.id}-${section.stage}-${index}`}><header><div><div className="code-section-kicker"><span>{section.stage}</span><span className={`code-source-tag source-${source.toLowerCase()}`}>{source}</span></div><b>{section.title}</b><small>{section.location}</small></div>{section.url && <a href={section.url} target="_blank" rel="noreferrer" aria-label={`打开 ${section.title} 固定源码`}>↗</a>}</header><pre><code>{section.code}</code></pre></article> })}</section> : <div className="code-empty"><b>此节点没有独立 forward</b><p>它由所在模块的 forward 调度,或只是一个数学拆解步骤。</p></div>}
  </div>;
}

type StageOverview = { kicker: string; title: string; summary: string; flow: string; formula: string; formulaNote?: string; notes: string[]; parameters: readonly (readonly [string, string, string])[] };

function stageOverview(variant: LayerVariant, stage: Exclude<ExpandedStage, null>): StageOverview {
  if (variant === "dense-full" && stage === "ffn") return {
    kicker: "DENSE FFN · L0", title: "SwiGLU MLP(18432)", summary: "L0 唯一的 dense FFN:扩维、silu 门控、投回 H。",
    flow: "Û → gate/up 投影 → packed → silu(g)⊙u → down 投影 → Y_ffn",
    formula: "[g‖u] = Û·Wᵀ\nz = silu(g)⊙u\nY_ffn = z·W_downᵀ",
    formulaNote: "dense FFN 与 shared expert 一样不做 swiglu clamp;只有 routed experts 截断到 ±10。",
    notes: ["gate/up 融合为一次 GEMM,forward 后按最后一维切分。", "L1–L77 把本子块替换为 MoE。"],
    parameters: [["H", "6144", "hidden_size"], ["H_ffn", "18432", "intermediate_size(L0)"], ["ε", "1e−5", "rms_norm_eps"]],
  };
  if (stage === "attention") return variant === "moe-shared" ? {
    kicker: "SHARED INDEXER · MoE 层", title: "稀疏 MLA(复用索引)", summary: "57 个 shared 层不建 indexer,直接读最近前驱 full 层的 top-2048 索引。",
    flow: "fused down proj → q/kv norm → RoPE → [读共享 buffer] → 候选内精确 attention → sink → gated → o_proj",
    formula: "S=QKᵀ/√256+M, j∈I_t\nout×=exp(lse)/(exp(lse)+exp(sink))",
    formulaNote: "I_t 来自模型级共享 buffer;checkpoint 证实 shared 层没有 indexer 权重。",
    notes: ["shared 层跳过 indexer 计算,注意力本体与 full 层完全一致。", "MTP draft 层不受此影响,始终自算索引。"],
    parameters: [["T_idx", "2048", "index_topk"], ["N_idx/D_idx", "32 / 128", "indexer(仅 full 层)"], ["sink", "64×FP32", "learnable_sink"]],
  } : {
    kicker: "FULL INDEXER · DSA", title: "稀疏 MLA(自算索引)", summary: "lightning indexer 打分选 top-2048 token,注意力只在候选内精确计算。",
    flow: "fused down proj → q/kv norm → RoPE → indexer 打分 → Top-2048 → 候选内 attention → sink → gated → o_proj",
    formula: "s_j=Σ_h w'⟨q_h,k_j⟩, I=TopK₂₀₄₈(s)\nS=QKᵀ/√256+M, j∈I",
    formulaNote: "indexer 的 k_norm 是 LayerNorm(ε=1e−6);q 按 128 元素组做 FP8(ue8m0)量化。",
    notes: ["候选集合写入共享 buffer,后续 shared 层直接复用。", "稀疏不回退 dense:后端不可用时直接报错。"],
    parameters: [["N_idx", "32", "index_n_heads"], ["D_idx", "128", "index_head_dim"], ["T_idx", "2048", "index_topk"], ["D_qk", "256", "qk_head_dim"]],
  };
  return {
    kicker: "TOP-8 MOE · L1–77", title: "MoE + Shared Expert", summary: "sigmoid 路由 256 选 8,shared expert 恒在。",
    flow: "路由:Û → FP32 gate → σ+bias Top-8 → ×2.827\n专家:Û → clamped SwiGLU(10) → Σŵ·E_e\n共享:Û → SwiGLU(无 clamp)\n合并:Y_routed + Y_shared",
    formula: "ŵ=2.827·s/Σs, s=σ(r)\nY_moe=Y_routed+Y_shared",
    formulaNote: "expert_bias 只影响选择;混合权重用未加 bias 的分数;激活参数约 49B 的来源就是 256 选 8。",
    notes: ["routed experts 是融合 all-experts 张量,运行时切 w1/w3。", "swiglu clamp 只在 routed;L0 是 dense FFN。"],
    parameters: [["E", "256", "routed experts"], ["K", "8", "experts/token"], ["E_sh", "1", "shared expert"], ["H_exp", "2048", "moe_intermediate_size"], ["s_route", "2.827", "routed_scaling_factor"]],
  };
}

function StageOverviewPanel({ variant, stage }: { variant: LayerVariant; stage: Exclude<ExpandedStage, null> }) {
  const overview = stageOverview(variant, stage);
  return <aside className="detail-panel stage-overview-panel"><header className="stage-overview-header"><span>{overview.kicker}</span><h2>{overview.title}</h2><p>{overview.summary}</p></header><div className="stage-overview-body"><section className="stage-flow-section"><span>数据流</span><code>{overview.flow}</code></section><section className="stage-formula-section"><span>计算语义</span><code>{overview.formula}</code>{overview.formulaNote && <p>{overview.formulaNote}</p>}</section><section className="stage-parameter-section"><span>关键参数</span><div className="stage-parameters">{overview.parameters.map(([symbol, value, source]) => <article key={symbol}><b>{symbol}</b><strong>{value}</strong><small>{source}</small></article>)}</div></section><section><span>边界说明</span>{overview.notes.map(note => <p key={note}>{note}</p>)}</section></div><footer>展开图说明 · 点击算子查看独立详情</footer></aside>;
}

export function DetailPanel({ node, tab, setTab, pinned, onClear, expanded, variant }: {
  node: OpNode | null; tab: Tab; setTab: (t: Tab) => void; pinned: boolean; onClear: () => void; expanded: ExpandedStage; variant: LayerVariant;
}) {
  const tabs: [Tab, string][] = [["io", "I/O + 权重"], ["formula", "公式"], ["code", "代码"]];
  if (!node && expanded) return <StageOverviewPanel variant={variant} stage={expanded} />;
  if (!node) return <aside className="detail-panel detail-empty"><div><span>MODULE DETAIL</span><b>尚未选择模块</b><p>点击左侧任一运算模块后,可在这里查看固定的 I/O、权重、公式和 forward 代码。</p></div></aside>;
  const formulaSteps = FORMULA_STEPS_BY_ID[node.id];
  return <aside className="detail-panel"><header className="detail-header"><div><span>{node.kicker}</span><h2>{node.title}</h2></div>{pinned ? <button className="unpin-button" aria-label="取消固定" title="取消固定" onClick={onClear}>×</button> : <i className={`kind-dot op-${node.kind}`} />}<p>{node.summary}</p><code>{node.runtime}</code></header><div className="detail-tabs">{tabs.map(([id, label]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>)}</div><div key={tab} className={`detail-content detail-${tab}`}>
    {tab === "io" && <IoView node={node} />}
    {tab === "formula" && <div className="formula-view"><span>作用</span><div className="formula-purpose">{node.summary}</div><span>实际公式</span><LatexFormula node={node} />{formulaSteps ? <div className="formula-steps">{formulaSteps.map(step => <article className="formula-step" key={step.title}><header>{step.title}</header><LatexExpression formula={step.formula} label={`${step.title} 公式`} className="formula-step-math" /><p>{step.explanation}</p></article>)}</div> : <><div className="formula-implementation"><b>一句话解释</b><p>{node.formulaNote ?? FORMULA_NOTE[node.kind]}</p></div><div className="formula-terms">{formulaTerms(node).map(([symbol, meaning]) => <span key={symbol}><b>{symbol}</b>{meaning}</span>)}</div></>}</div>}
    {tab === "code" && <CodeView node={node} />}
  </div><footer>vLLM @ {VLLM_COMMIT.slice(0, 7)} · {CONFIG_FETCH_NOTE}</footer></aside>;
}
