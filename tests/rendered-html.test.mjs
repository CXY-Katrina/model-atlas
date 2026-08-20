import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import katex from "katex";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the MiniMax-M3 architecture workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /MiniMax-M3/);
  assert.match(html, /模型结构概览/);
  assert.match(html, /CODE ↗/);
  assert.match(html, /WEIGHTS ↗/);
  assert.match(html, /vLLM @/);
  assert.match(html, /edd4c81/);
  assert.match(html, /模型总参数量/);
  assert.match(html, /每 token 激活参数/);
  assert.match(html, /MSA \+ Top-4 MoE/);
  assert.match(html, />MSA</);
  assert.doesNotMatch(html, /Sparse GQA/i);
  assert.match(html, /Top-4 MoE \+ Shared Expert/);
  assert.match(html, /L3–L59/);
  assert.match(html, /57 层共享同一实现/);
  assert.match(html, /尚未选择模块/);
  assert.match(html, /OPERATOR/);
  assert.match(html, /TENSOR/);
  assert.match(html, /EXTERNAL/);
  assert.match(html, /WEIGHT/);
  assert.match(html, /Text \/ Vision Inputs/);
  assert.match(html, /Embedding Fusion/);
  assert.match(html, /Final Gemma RMSNorm/);
  assert.match(html, /LM Head/);
  assert.doesNotMatch(html, /QKV \+ Index Projection/);
  assert.doesNotMatch(html, /ATTENTION RUNTIME I\/O/);
  assert.match(html, /查看参数和符号说明/);
  assert.match(html, /og-tensor-operator-map\.png/);
  assert.doesNotMatch(html, /一屏看完整结构|ARCHITECTURE × CODE × WEIGHTS|hover 预览 · click 固定/);
  assert.doesNotMatch(html, /Building your site|SkeletonPreview|codex-preview/);
});

test("keeps code, checkpoint, formula, and shape evidence together", async () => {
  const [page, modelData, css] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/model-data.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  const source = `${page}\n${modelData}`;

  assert.match(source, /MinimaxM3QKVParallelLinearWithIndexer/);
  assert.match(source, /block_sparse_moe\.experts\.0\.w1\.weight/);
  assert.match(source, /safetensors/);
  assert.match(source, /inputShape/);
  assert.match(source, /formula/);
  assert.match(source, /weights/);
  assert.match(source, /type OpKind/);
  assert.match(source, /Top-16 Blocks/);
  assert.match(source, /Select KV Pages/);
  assert.match(source, /side:"EXTERNAL"/);
  assert.match(source, /position\(req,i\)=num_computed_tokens\[req\]\+i/);
  assert.match(source, /CommonAttentionMetadata/);
  assert.match(source, /compute_slot_mapping/);
  assert.match(source, /Apply Causal \/ Pad Bounds/);
  assert.match(source, /katex\.renderToString/);
  assert.match(source, /LATEX_BY_ID/);
  assert.match(source, /简化 LATEX/);
  assert.match(source, /operatorname\{TopK\}_4/);
  assert.match(source, /theta_\{p,j\}/);
  assert.match(source, /权重名称为什么与代码不同/);
  assert.match(source, /SiluAndMulWithClamp/);
  assert.match(source, /self\.act_fn/);
  assert.match(source, /forward_native/);
  assert.match(source, /torch\.clamp\(x\[\.\.\., :d\], max=self\.swiglu_limit\)/);
  assert.match(source, /FORWARD ONLY/);
  assert.match(source, /MergedColumnParallelLinear/);
  assert.match(source, /I\/O \+ 权重/);
  assert.match(source, /INPUT BINDINGS/);
  assert.match(source, /上游张量/);
  assert.match(source, /外部输入/);
  assert.match(source, /权重输入/);
  assert.match(source, /checkpoint →/);
  assert.match(source, /Build Position IDs 输出/);
  assert.match(source, /NEXT_BY_ID/);
  assert.doesNotMatch(source, /\["weights","权重"\]/);
  assert.doesNotMatch(page, /type="range"/);
  assert.match(source, /MODEL_REGISTRY/);
  assert.match(source, /尚未选择模块/);
  assert.match(source, /LayerType/);
  assert.match(source, /const active=detail\.pinned\?\?detail\.hovered/);
  assert.match(source, /onPointerDown=\{\(\)=>onSelect\(node\)\}/);
  assert.match(source, /nextDetailState/);
  assert.match(source, /function AddCircle/);
  assert.match(source, /function InputWeightedOp/);
  assert.match(source, /input_layernorm\.weight/);
  assert.match(source, /replaceAll\("6144","H"\)/);
  assert.match(source, /function checkpointWeightName/);
  assert.doesNotMatch(source, /label="γ(?:post|q|k)"/);
  assert.doesNotMatch(source, /Tensor name="W(?:gate|up|down|router|routed|shared)"/);
  assert.match(source, /mlp\.gate_proj\.weight/);
  assert.doesNotMatch(page, /DECODER LAYER TYPE|Dense GQA · SwiGLU MLP|Indexer Attention · Top-4 MoE/);
  assert.match(source, /block_sparse_moe\.gate\.weight/);
  assert.match(source, /block_sparse_moe\.experts\.\*\.\{w1,w3,w2\}\.weight/);
  assert.match(source, /title:"Gemma RMSNorm"/);
  assert.match(source, /title:"Post-attn Gemma RMSNorm"/);
  assert.doesNotMatch(source, /title:"RMSNorm"/);
  assert.match(source, /toPort:"top-left"/);
  assert.match(source, /toPort:"top-right"/);
  assert.match(source, /function GraphSurface/);
  assert.match(source, /routeGraphEdge/);
  assert.match(source, /data-graph-id/);
  assert.match(source, /ResizeObserver/);
  assert.match(source, /markerEnd/);
  assert.doesNotMatch(source, /graph-arrowheads/);
  assert.match(source, /from:"main-x",to:"main-add1",fromPort:"left",toPort:"left",route:"side-left"/);
  assert.match(source, /from:"main-u",to:"main-add2",fromPort:"left",toPort:"left",route:"side-left"/);
  const graphNodes = new Set(
    [...source.matchAll(/(?:graphId|inputGraphId|weightGraphId|data-graph-id)="((?:main|mlp|moe|attn)-[^"]+)"/g)].map((match) => match[1]),
  );
  const graphEndpoints = new Set(
    [...source.matchAll(/(?:from|to):"((?:main|mlp|moe|attn)-[^"]+)"/g)].map((match) => match[1]),
  );
  assert.deepEqual([...graphNodes].filter((id) => !graphEndpoints.has(id)), [], "every rendered graph node needs an edge");
  assert.deepEqual([...graphEndpoints].filter((id) => !graphNodes.has(id)), [], "every graph edge needs rendered endpoints");
  assert.match(source, /L0–L2/);
  assert.match(source, /L3–L59/);
  assert.match(source, /同一个 Û 同时进入 Router、Routed Experts 与 Shared Expert/);
  assert.match(source, /symbolicShape/);
  assert.match(source, /Nₕ\/TP/);
  assert.match(source, /E\/EP/);
  assert.match(source, /完整 config\.json/);
  assert.match(source, /sparse_attention_config/);
  assert.match(source, /vision_segment_max_frames/);
  assert.match(source, /image_grid_pinpoints/);
  assert.match(source, /forward \/ forward_native/);
  assert.match(css, /height:100svh/);
  assert.match(css, /overflow:hidden/);
  assert.match(css, /\.tensor-node/);
  assert.match(css, /tensor artifact → compute operator → tensor artifact/);
  assert.match(css, /\.runtime-io/);
  assert.match(css, /\.tensor-input/);
  assert.match(css, /\.tensor-output/);
  assert.match(css, /\.latex-render/);
  assert.match(css, /\.katex-display/);
  assert.match(css, /\.code-symbols/);
  assert.match(css, /\.code-call-chain/);
  assert.match(css, /\.code-section/);
  assert.match(css, /\.io-binding-view/);
  assert.match(css, /\.binding-weight/);
  assert.match(css, /\.binding-external/);
  assert.match(css, /\.decoder-column/);
  assert.match(css, /\.decoder-workbench\.has-zoom\{[^}]*grid-template-columns:360px minmax\(0,1fr\)/);
  assert.doesNotMatch(css, /\.decoder-workbench\.has-zoom\{[^}]*grid-template-columns:170px/);
  assert.match(css, /\.stage-zoom/);
  assert.match(css, /\.parallel-experts/);
  assert.match(css, /\.add-circle/);
  assert.match(css, /\.weighted-op/);
  assert.match(css, /\.input-weighted-op/);
  assert.match(css, /\.co-input-row/);
  assert.match(css, /\.parallel-gate-up/);
  assert.match(css, /\.graph-connectors/);
  assert.doesNotMatch(css, /\.graph-arrowheads/);
  assert.match(css, /\.decoder-node-graph/);
  assert.match(css, /\.decoder-node-graph>\.input-weighted-op\{[^}]*row-gap:18px/);
  assert.match(css, /\.connected-attention-graph/);
  assert.match(css, /\.mlp-node-graph/);
  assert.match(css, /\.moe-node-graph/);
  assert.match(css, /\.shape-rows/);
  assert.match(css, /\.config-reference/);
  assert.match(css, /\.detail-formula\{overflow:hidden/);
  assert.doesNotMatch(css, /\.op-node\{[^}]*border-left/);
});

test("renders every operator equation as valid LaTeX", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const equations = [...page.matchAll(/String\.raw`([^`]*)`/g)].map((match) => match[1]);

  assert.ok(equations.length >= 40, `expected a complete formula set, got ${equations.length}`);
  for (const equation of equations) {
    assert.doesNotThrow(() => katex.renderToString(equation, { throwOnError: true }));
  }
});

test("routes the MoE hidden tensor into the shared expert without crossing its weight tensor", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(
    source,
    /\{from:"moe-u",to:"moe-shared",fromPort:"bottom",toPort:"top"\}/,
  );
  assert.doesNotMatch(
    source,
    /\{from:"moe-u",to:"moe-shared",route:"side-right",fromPort:"right",toPort:"right"\}/,
  );

  const edgeBlock = source.match(/const edges:GraphEdge\[\]=\[\s*\{from:"moe-u"([\s\S]*?)\n\s*\];/)?.[0] ?? "";
  const artifacts = new Set([
    "moe-u", "moe-wrouter", "moe-ids", "moe-rweights", "moe-wexperts",
    "moe-routed", "moe-wshared", "moe-shared-out", "moe-y",
  ]);
  const edges = [...edgeBlock.matchAll(/\{from:"([^"]+)",to:"([^"]+)"/g)];
  assert.equal(edges.length, 15, "expected every MoE graph edge in the invariant check");
  for (const [, from, to] of edges) {
    assert.notEqual(
      artifacts.has(from),
      artifacts.has(to),
      `${from} → ${to} must alternate tensor/weight artifacts and operators`,
    );
  }
});
