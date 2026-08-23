import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import katex from "katex";

test("builds a static GitHub Pages entry", async () => {
  const [html, builtHtml, main, page] = await Promise.all([
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../dist/index.html", import.meta.url), "utf8"),
    readFile(new URL("../app/main.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(html, /<html lang="zh-CN">/);
  assert.match(html, /<div id="root"><\/div>/);
  assert.match(html, /Model Atlas · MiniMax-M3/);
  assert.match(html, /og-tensor-operator-map\.png/);
  assert.match(main, /createRoot/);
  assert.match(main, /katex\/dist\/katex\.min\.css/);
  assert.match(page, /MiniMax-M3/);
  assert.match(page, /模型总参数量/);
  assert.match(page, /Embedding Fusion/);
  assert.match(page, /MiniMax Sparse Attention \+ Partial RoPE \+ Top-4 MoE/);
  assert.match(page, /MiniMax Sparse Attention \+ Partial RoPE/);
  assert.match(page, /SwiGLU-OAI MLP/);
  assert.doesNotMatch(page, />SwiGLU MLP</);
  assert.doesNotMatch(page, /Top-16 blocks · causal mask · KV cache|Routed 与 Shared 两路并行|Gate \/ Up 并行 → ⊙ → Down/);
  assert.match(builtHtml, /\.\/assets\/index-[^"']+\.js/);
  assert.match(builtHtml, /\.\/assets\/index-[^"']+\.css/);
  assert.doesNotMatch(`${html}\n${builtHtml}\n${main}`, /_next|vinext|wrangler|cloudflare/i);
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
  assert.match(source, /实际公式/);
  assert.match(source, /operatorname\{TopK\}_K/);
  assert.match(source, /theta_\{p,j\}/);
  assert.doesNotMatch(source, /权重名称为什么与代码不同/);
  assert.match(source, /SiluAndMulWithClamp/);
  assert.match(source, /self\.act_fn/);
  assert.match(source, /forward_native/);
  assert.match(source, /torch\.clamp\(x\[\.\.\., :d\], max=self\.swiglu_limit\)/);
  assert.match(source, /IMPLEMENTATION TRACE/);
  assert.match(source, /NORM_FORWARD_URL.*#L130-L142/);
  assert.match(source, /FLASHINFER_GEMMA_NORM_URL/);
  assert.match(source, /gemma_fused_add_rmsnorm\(x, residual/);
  assert.match(source, /node\.latex\?\?SIMPLE_FORMULA/);
  assert.match(source, /图中 residual 沿旁路单独保留/);
  assert.match(source, /output:\s*"normalized hidden_states", outputShape:\s*"\[B,S,6144\]"/);
  assert.doesNotMatch(source, /output:"normalized · updated residual"|outputShape:"\[B,S,6144\] ×2"/);
  assert.match(source, /FORMULA_TERMS_BY_ID/);
  assert.match(source, /\["γ","input_layernorm\.weight"\]/);
  assert.match(source, /\["H","hidden_size = 6144"\]/);
  assert.doesNotMatch(source, /<b>x \/ a \/ b<\/b>/);
  assert.match(source, /\["U","上游 Add 节点的输出"\]/);
  assert.doesNotMatch(source, /"d-postnorm":\[\["Xₗ"|"s-postnorm":\[\["Xₗ"/);
  assert.match(css, /\.formula-terms\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
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
  assert.match(page, /aria-label="取消固定"[^>]*>×<\/button>/);
  assert.doesNotMatch(page, /已固定 · 取消/);
  assert.match(source, /onPointerDown=\{\(\)=>onSelect\(node\)\}/);
  assert.match(source, /nextDetailState/);
  assert.match(source, /function AddCircle/);
  assert.match(source, /function InputWeightedOp/);
  assert.match(source, /input_layernorm\.weight/);
  assert.match(source, /replaceAll\("6144","H"\)/);
  assert.match(source, /replaceAll\("\[B,S,24576\]","\[B,S,2H_dense\]"\)/);
  assert.match(source, /replaceAll\("24576","2H_dense"\)/);
  assert.match(source, /function checkpointWeightName/);
  assert.doesNotMatch(source, /label="γ(?:post|q|k)"/);
  assert.doesNotMatch(source, /Tensor name="W(?:gate|up|down|router|routed|shared)"/);
  assert.match(source, /mlp\.gate_proj\.weight/);
  assert.match(source, /id:"d-gatesplit",kind:"split",kicker:"DENSE FFN · TP-LOCAL SPLIT",title:"Split Gate \/ Up"/);
  assert.match(source, /graphId="mlp-packed"/);
  assert.match(source, /graphId="mlp-split"/);
  assert.match(source, /data-graph-id="mlp-gate-act"[\s\S]{0,600}clamp → Ḡ⁽ʳ⁾·σ\(αḠ⁽ʳ⁾\)/);
  assert.match(source, /data-graph-id="mlp-up-act"[\s\S]{0,600}clamp → Ū⁽ʳ⁾ \+ β/);
  assert.match(source, /className="mini-math activation-step"/);
  assert.match(source, /title:"SiluAndMulWithClamp · SwiGLU-OAI"/);
  assert.match(source, /<Tensor name="Z⁽ʳ⁾" shape="\[B,S,H_dense\/TP\]" graphId="mlp-activated"\/>/);
  assert.doesNotMatch(source, /Z⁽ʳ⁾ · activated⁽ʳ⁾/);
  assert.match(source, /CODE_BY_ID\["d-swiglu"\]=\{sections:SWIGLU_SECTIONS,symbols:SWIGLU_SYMBOLS\}/);
  assert.doesNotMatch(source, /graphId="mlp-post"|graphId="mlp-wpost"|graphId="mlp-u"/);
  assert.doesNotMatch(source, /className="lesson-notes"/);
  assert.match(source, /function StageOverviewPanel/);
  assert.doesNotMatch(source, /className="stage-zoom lesson-zoom(?: attention-lesson)?"><header><div><span>/);
  assert.match(source, /<header><span>SWIGLU-OAI MLP · L0–2<\/span>/);
  assert.match(source, /<header><span>TOP-4 MOE \+ SHARED EXPERT · L3–59<\/span>/);
  assert.match(source, /\["α","1\.702","swiglu_alpha"\]/);
  assert.match(source, /\["β","1\.0","swiglu_beta"\]/);
  assert.match(source, /"text_config:swiglu_beta":"β"/);
  assert.match(source, /\["c","7\.0","swiglu_limit"\]/);
  assert.match(source, /\["H_dense","12288","dense_intermediate_size"\]/);
  assert.match(source, /G⁽ʳ⁾ = Û \(W_gate⁽ʳ⁾\)ᵀ/);
  assert.match(source, /Z⁽ʳ⁾ = Ḡ⁽ʳ⁾ ⊙ σ\(αḠ⁽ʳ⁾\) ⊙ \(Ū⁽ʳ⁾ \+ β\)/);
  assert.match(source, /σ 表示 sigmoid，⊙ 表示逐元素相乘/);
  assert.match(source, /id:"d-gateup",kind:"linear",kicker:"DENSE FFN · H=6144 · H_dense=12288"/);
  assert.match(source, /id:"d-gatesplit",kind:"split",kicker:"DENSE FFN · TP-LOCAL SPLIT"/);
  assert.match(source, /本节点只切分 view：前 H_dense\/TP 个通道是 gate/);
  assert.match(source, /CODE_BY_ID\["d-gateup"\]=\{sections:GATE_UP_SECTIONS,symbols:GATE_UP_SYMBOLS\}/);
  assert.match(source, /output_size_per_partition = divide\(output_size, self\.tp_size\)/);
  assert.match(source, /outputShape:"\[B,64\/TP,S,T\]"/);
  assert.match(source, /outputShape:"\[B,S,8192\/TP\]"/);
  assert.match(source, /max\(1,4\/TP\)/);
  assert.match(source, /shape:tpShape\?`\$\{weight\.dtype\} · TP shard \$\{tpShape\}`/);
  assert.doesNotMatch(source, /TP shard \$\{tpShape\} · checkpoint \$\{weight\.shape\}/);
  const latexBlock = source.match(/const LATEX_BY_ID:[\s\S]*?const NORM_SECTIONS/)?.[0] ?? "";
  assert.doesNotMatch(latexBlock, /6144|12288|9856|9216|8704|8192|512|128|64|16|7\.0|1\.702|1\.0|10\^\{(?:29|30|-6)\}/);
  assert.match(source, /id:"d-add1",kind:"add",kicker:"DECODER LAYER · ATTENTION RESIDUAL"/);
  assert.match(source, /source:"nvidia\/model\.py · MiniMaxM3DecoderLayer\.forward · L773–775"/);
  assert.match(source, /id:"d-add2",kind:"add",kicker:"DECODER LAYER · FFN RESIDUAL"/);
  assert.match(source, /下一 Decoder Layer 在 L758–767 的 fused input RMSNorm 中执行实际 add/);
  assert.match(source, /"d-add2":\[\["U","Attention 后的 residual stream/);
  assert.doesNotMatch(source, /title:"\+ MLP Residual"/);
  assert.doesNotMatch(page, /DECODER LAYER TYPE|Dense GQA · SwiGLU MLP|Indexer Attention · Top-4 MoE/);
  assert.match(source, /block_sparse_moe\.gate\.weight/);
  assert.match(source, /name="routed expert weights · w1\/w3\/w2"/);
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
  assert.match(source, /同一个 Û 直接进入 Router、Routed Experts 与 Shared Expert/);
  assert.match(source, /symbolicShape/);
  assert.doesNotMatch(source, /CURRENT OPERATOR|className=\{`io-operator/);
  assert.match(source, /<i>符号<\/i><code title=\{symbolicShape\(shape\)\}>/);
  assert.match(source, /<i>实际<\/i><code title=\{shape\}>/);
  assert.match(source, /replaceAll\("6144","H"\)/);
  assert.match(source, /完整 config\.json/);
  assert.match(source, /CONFIG_SYMBOLS/);
  assert.match(source, /<th>参数<\/th><th>符号<\/th><th>值<\/th>/);
  assert.match(source, /className="config-tabs" role="tablist"/);
  assert.match(source, /role="tabpanel"/);
  assert.doesNotMatch(source, /Shape · TP \/ EP/);
  assert.match(source, /sparse_attention_config/);
  assert.match(source, /vision_segment_max_frames/);
  assert.match(source, /image_grid_pinpoints/);
  assert.match(css, /height:100svh/);
  assert.match(css, /overflow:hidden/);
  assert.match(css, /--font-geist-sans:Consolas,"Microsoft YaHei",monospace;--font-geist-mono:Consolas,"Microsoft YaHei",monospace/);
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
  assert.match(css, /\.layer-type-options button\{[^}]*align-content:center/);
  assert.match(css, /\.decoder-workbench\.has-zoom\{[^}]*grid-template-columns:minmax\(440px,\.85fr\) minmax\(0,2fr\)/);
  assert.match(css, /\.stage-zoom/);
  assert.match(css, /\.stage-zoom>header\{align-items:center\}/);
  assert.match(css, /\.parallel-experts/);
  assert.match(css, /\.add-circle/);
  assert.match(css, /\.weighted-op/);
  assert.match(css, /\.input-weighted-op/);
  assert.match(css, /\.co-input-row/);
  assert.match(css, /\.parallel-gate-up/);
  assert.match(css, /\.graph-connectors/);
  assert.doesNotMatch(css, /\.graph-arrowheads/);
  assert.match(css, /\.decoder-node-graph/);
  assert.match(css, /\.decoder-node-graph>\.input-weighted-op\{[^}]*row-gap:clamp\(18px,2\.4vh,28px\)/);
  assert.match(css, /\.decoder-column\{[^}]*width:min\(680px,96%\)/);
  assert.match(css, /\.decoder-node-graph \.co-input-row\{[^}]*grid-template-columns:max-content max-content/);
  assert.match(css, /\.decoder-node-graph \.co-input-row>\.tensor-node\{[^}]*width:max-content/);
  assert.doesNotMatch(css, /\.decoder-node-graph>\.tensor-node\{[^}]*min-width:260px/);
  assert.match(css, /\.decoder-node-graph \.input-weighted-op>\.op-node\{[^}]*width:max-content[^}]*min-width:0/);
  assert.doesNotMatch(css, /\.decoder-node-graph \.input-weighted-op>\.op-node\{[^}]*min-width:220px/);
  assert.match(css, /\.decoder-node-graph>\.stage-summary\{[^}]*width:max-content/);
  assert.match(css, /\.connected-attention-graph/);
  assert.match(css, /\.mlp-node-graph/);
  assert.match(css, /\.mlp-node-graph\{width:min\(600px,96%\);gap:8px 6px;padding:8px 8px\}/);
  assert.match(css, /\.mlp-node-graph>\[data-graph-id\]\{[^}]*width:max-content[^}]*max-width:260px/);
  assert.match(css, /\.stage-overview-panel\{grid-template-rows:auto minmax\(0,1fr\) 28px\}/);
  assert.match(css, /\.stage-formula-section code\{white-space:pre-line\}/);
  assert.match(css, /\.activation-step\{cursor:pointer\}/);
  assert.match(css, /\.multiply-circle\[aria-pressed="true"\]\{[^}]*outline:2px solid #1f6c4d5c[^}]*border-color:var\(--green\)/);
  assert.match(css, /\.mlp-node-graph \[data-graph-id="mlp-wdown"\]\{grid-area:9\/5\}/);
  assert.match(css, /\.moe-node-graph/);
  assert.match(css, /\.moe-node-graph \[data-graph-id="moe-experts"\]\{grid-area:4\/3\}/);
  assert.match(css, /\.moe-node-graph \[data-graph-id="moe-sum"\]\{grid-area:6\/4\}/);
  assert.match(css, /\.moe-node-graph \[data-graph-id="moe-y"\]\{grid-area:7\/4\}/);
  assert.match(css, /\.stage-zoom \.moe-node-graph \.tensor-weight\{width:240px;max-width:100%;min-height:82px;max-height:none;padding:14px 18px;gap:5px\}/);
  assert.match(css, /\.shape-rows/);
  assert.match(css, /\.shape-rows code\{[^}]*white-space:normal[^}]*overflow:visible[^}]*text-overflow:clip[^}]*overflow-wrap:anywhere/);
  assert.match(css, /@media\(min-width:1160px\)\{\.screen-grid\{grid-template-columns:minmax\(680px,1fr\) 460px\}\}/);
  assert.match(css, /\.shape-rows\{display:flex!important;flex-direction:column;align-items:stretch;gap:6px\}/);
  assert.match(css, /\.shape-rows>span\{width:100%;grid-template-columns:46px minmax\(0,1fr\)\}/);
  assert.match(css, /\.shape-rows code\{white-space:nowrap;overflow-wrap:normal!important\}/);
  assert.doesNotMatch(css, /container-type:inline-size|@container \(max-width:430px\)/);
  assert.match(css, /\.model-facts small\{[^}]*font-size:12px/);
  assert.match(css, /\.tensor-node code\{font-size:12px/);
  assert.match(css, /\.stage-zoom \.graph-surface \.tensor-node code\{font-size:12px/);
  assert.match(css, /\.shape-rows code\{font:12px/);
  assert.match(css, /\.model-overview \.model-step code\{font-size:12px/);
  const fixedPixelFontSizes = [...css.matchAll(/(?:font-size|font):(\d+(?:\.\d+)?)px/g)]
    .map((match) => Number(match[1]));
  assert.ok(fixedPixelFontSizes.every((size) => size >= 12), "all fixed pixel font sizes should be at least 12px");
  assert.match(css, /\.config-reference/);
  assert.match(css, /\.config-reference\{[^}]*overflow-x:hidden/);
  assert.match(css, /\.config-reference table\{[^}]*table-layout:fixed/);
  assert.match(css, /\.config-tabs\{[^}]*flex-wrap:wrap/);
  assert.doesNotMatch(css, /font-weight:(?:750|800)/);
  assert.match(css, /\.detail-formula\{overflow:hidden/);
  assert.match(css, /\.unpin-button\{[^}]*width:28px[^}]*height:28px/);
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
  assert.match(source, /\{from:"moe-u",to:"moe-experts",toPort:"top-right"\}/);
  assert.doesNotMatch(source, /\{from:"moe-u",to:"moe-experts",route:"side-left"/);
  assert.match(source, /name="shared expert weights ×3" shape="gate \/ up \/ down"/);
  assert.doesNotMatch(source, /name="block_sparse_moe\.shared_experts\.\{gate_proj,up_proj,down_proj\}\.weight"/);

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
