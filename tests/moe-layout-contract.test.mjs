import { readModelSource, readStyles } from "./source-helpers.mjs";
import assert from "node:assert/strict";
import test from "node:test";

const page = readModelSource();
const css = readStyles();

test("MoE expert branches use short independent lanes and a symmetric merge", () => {
  assert.doesNotMatch(page, /className="moe-expert-branches"/);
  assert.match(css, /\.lesson-zoom \.graph-pan-content\{height:100%\}/);
  assert.match(css, /\.graph-pan-content \.moe-node-graph\{[^}]*width:min\(1180px,100%\)[^}]*grid-template-columns:minmax\(140px,1fr\) 36px minmax\(180px,1\.15fr\) 36px minmax\(230px,1\.4fr\) 36px minmax\(180px,1\.15fr\) 36px minmax\(140px,1fr\)[^}]*grid-template-rows:52px 64px 64px 64px 72px 64px 76px[^}]*gap:24px 10px/);
  assert.match(css, /\[data-graph-id="moe-experts"\]\{grid-area:4\/5\}/);
  assert.match(css, /\[data-graph-id="moe-wexperts"\]\{grid-area:4\/1\}/);
  assert.match(css, /\[data-graph-id="moe-shared"\]\{grid-area:2\/7\}/);
  assert.match(css, /\[data-graph-id="moe-sum"\]\{grid-area:6\/5\/7\/8\}/);
  assert.doesNotMatch(page, /graphId="moe-expert-input"/);
  assert.doesNotMatch(page, /\{from:"moe-u",to:"moe-expert-input"/);
  assert.match(page, /\{from:"moe-u",to:"moe-experts"\}/);
  assert.match(page, /\{from:"moe-router-logits",to:"moe-experts",toPort:"top-left",approach:28\}/);
  assert.match(page, /\{from:"moe-routed",to:"moe-sum",toPort:"top-left",approach:24\}/);
  assert.match(page, /\{from:"moe-shared-out",to:"moe-sum",toPort:"top-right",approach:24\}/);
});

test("weight nodes and their legend use the same dashed outline", () => {
  assert.match(css, /\.weight-swatch\{border:1px dashed #a56e21/);
  assert.match(css, /\.tensor-weight\{[^}]*border:1px dashed #a56e21/);
});

test("Router exposes only materialized logits and owns only its code", () => {
  assert.match(page, /const ROUTER_SECTIONS: CodeSection\[\] = \[\s*\{stage:"1 · ROUTE"/);
  assert.match(page, /CODE_BY_ID\["s-router"\]=\{sections:ROUTER_SECTIONS,symbols:ROUTER_SYMBOLS\};/);
  assert.match(page, /title:"FP32 Router Logits"[\s\S]*output:"router_logits"[\s\S]*outputShape:"\[B,S,128\]"/);
  assert.match(page, /name="router logits" shape="\[B,S,E\]" graphId="moe-router-logits"/);
  assert.match(page, /\{from:"moe-router",to:"moe-router-logits"\}/);
  assert.match(page, /\{from:"moe-router-logits",to:"moe-experts",toPort:"top-left"/);
  assert.doesNotMatch(page, /graphId="moe-(?:ids|rweights)"/);
});

test("FusedMoE formula explains every symbol and shows both implementations", () => {
  for (const symbol of ["σ","sₑ","𝓔","ŵₑ","e / j","Û","Eₑ(Û)","Yᵣₒᵤₜₑd"]) {
    assert.ok(page.includes(`["${symbol}"`), `missing explanation for ${symbol}`);
  }
  assert.doesNotMatch(page, /\["gₑ \/ vₑ","expert gate \/ up 分支"\]/);
  assert.match(page, /const TRANSFORMERS_MOE_URL = "https:\/\/github\.com\/huggingface\/transformers\/blob\/main\/src\/transformers\/models\/minimax_m3_vl\/modeling_minimax_m3_vl\.py#L202-L239"/);
  assert.match(page, /stage:"3 · TRANSFORMERS ROUTER"/);
  assert.match(page, /title:"MiniMaxM3VLTopKRouter：可读路由实现"/);
  assert.match(page, /scores_for_choice = routing_weights \+ self\.e_score_correction_bias/);
  assert.match(page, /stage:"4 · TRANSFORMERS EXPERTS"/);
  assert.match(page, /title:"MiniMaxM3VLExperts：专家计算与加权归并"/);
  assert.match(page, /final\.index_add_\(0, token_idx, current\.to\(final\.dtype\)\)/);
});

test("source navigation, code provenance, and formula overflow stay usable", () => {
  assert.match(page, /label: "TRANSFORMERS"[^\n]+url: TRANSFORMERS_MOE_URL/);
  assert.match(page, /function codeSourceLabel\(section:CodeSection\)/);
  assert.match(page, /className=\{`code-source-tag source-\$\{source\.toLowerCase\(\)\}`\}/);
  assert.match(page, />\{source\}<\/span>/);
  assert.match(css, /\.code-source-tag\{[^}]*border-radius:/);
  assert.match(css, /\.formula-view \.latex-render\{[^}]*overflow:auto/);
  assert.match(css, /\.formula-view \.katex-display\{[^}]*min-width:max-content/);
  assert.match(page, /<div key=\{tab\} className=\{`detail-content detail-\$\{tab\}`\}>/);
});

test("FusedMoE explanation follows the formula line by line", () => {
  assert.match(page, /const FORMULA_STEPS_BY_ID: Partial<Record<string,readonly FormulaStep\[\]>>/);
  assert.match(page, /"s-experts":\[\s*\{title:"1 · 得分与选择"/);
  assert.match(page, /\{title:"2 · 生成混合权重"/);
  assert.match(page, /\{title:"3 · 专家计算与归并"/);
  assert.match(page, /const formulaSteps=FORMULA_STEPS_BY_ID\[node\.id\]/);
  assert.match(page, /formulaSteps\?<div className="formula-steps">/);
  assert.match(page, /<LatexExpression formula=\{step\.formula\}/);
  assert.match(page, /:<><div className="formula-implementation">/);
  assert.match(css, /\.formula-steps\{display:grid/);
  assert.match(css, /\.formula-step\{[^}]*grid-template-columns:1fr/);
});

test("MoE overview names both branches and their final add explicitly", () => {
  assert.match(page, /flow:"路由选择：Û → FP32 Router → router_logits\\n路由专家：Û \+ router_logits → Fused Top-4 Routing \+ Experts → Y_routed\\n共享专家：Û → Shared Expert → Y_shared\\n最终合并：Y_routed \+ Y_shared → Add → Y_moe"/);
  assert.match(page, /title:"Add Routed \+ Shared",summary:"把路由专家分支的 Y_routed 与共享专家分支的 Y_shared 逐元素相加。"/);
  assert.match(page, /className="stage-flow-section"/);
  assert.match(css, /\.stage-flow-section code,\.stage-formula-section code\{white-space:pre-line\}/);
  assert.doesNotMatch(page, /↘ Weighted Sum → Ymoe；Û → Shared Expert ↗/);
});
