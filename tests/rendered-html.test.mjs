import { readModelSource, readStyles } from "./source-helpers.mjs";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import katex from "katex";

test("builds a static GitHub Pages entry", async () => {
  const [html, builtHtml, main, page] = await Promise.all([
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../dist/index.html", import.meta.url), "utf8"),
    readFile(new URL("../app/main.tsx", import.meta.url), "utf8"),
    readModelSource(),
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

test("expanded attention keeps labels clear and vertical spacing uniform", async () => {
  const source = readModelSource();
  const css = readStyles();

  assert.match(css, /\.graph-pan-content \.connected-attention-graph\{[^}]*--attention-row-gap:/);
  for (const path of ["Q", "K", "V"]) assert.match(source, new RegExp(`<header>${path} PATH<\\/header>`));
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.qkv-lanes\{[^}]*grid-template-columns:repeat\(3,max-content\)/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.qkv-lanes>section\{[^}]*grid-template-rows:28px 148px 62px 62px 62px/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.qkv-lanes>section>header\{[^}]*justify-self:start/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.tensor-weight\{[^}]*min-width:150px[^}]*overflow-wrap:anywhere/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.score-pipeline\{[^}]*flex:0 0 auto[^}]*width:min\(920px,100%\)[^}]*background:transparent/);
  assert.match(source, /\{from:"attn-qr",to:"attn-qk",toPort:"top-left",approach:28\}/);
  assert.match(source, /\{from:"attn-paged-k",to:"attn-qk"\}/);
  assert.match(source, /\{from:"attn-p",to:"attn-pv",toPort:"top-left",approach:34\}/);
  assert.match(source, /\{from:"attn-paged-v",to:"attn-pv",toPort:"top-right",approach:34\}/);
  assert.equal(source.match(/graphId="attn-paged-v"/g)?.length, 1);
  assert.match(source, /className="score-pipeline"[\s\S]*graphId="attn-p"[\s\S]*graphId="attn-paged-v"/);
  assert.match(css, /\.score-pipeline \[data-graph-id="attn-paged-v"\]\{grid-area:7\/3\}/);
  assert.match(css, /\.context-pipeline \[data-graph-id="attn-pv"\]\{grid-area:1\/2\/2\/4;justify-self:center\}/);
});

test("expanded sparse attention keeps blocks wide and connector lanes separated", async () => {
  const source = readModelSource();
  const css = readStyles();

  assert.match(css, /\.graph-pan-content \.connected-attention-graph\.sparse-attention\{[^}]*width:max\(100%,1740px\)/);
  assert.match(css, /\.attention-branches:not\(\.dense\)\{[^}]*grid-template-columns:max-content minmax\(600px,1fr\)/);
  assert.match(css, /\.attention-branches:not\(\.dense\)>\.attention-data-path\{[^}]*grid-area:1\/1/);
  assert.match(css, /\.attention-branches:not\(\.dense\)>\.index-ribbon\{[^}]*grid-area:1\/2/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.qkv-lanes \.co-input-row\{[^}]*grid-template-columns:max-content max-content/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.index-ribbon\{[^}]*padding:22px var\(--group-padding-inline\)/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.index-ribbon\{[^}]*grid-template-rows:28px 90px/);
  assert.match(css, /\.graph-pan-content \.connected-attention-graph \.index-ribbon-label\{[^}]*grid-area:1\/3\/2\/4[^}]*transform:translateY\(-4px\)/);
  assert.match(source, /className="index-ribbon-label">LIGHTNING INDEXER<\/header>/);

  for (const edge of [
    '{from:"attn-split",to:"attn-q",fanout:"attn-five-way",departure:32}',
    '{from:"attn-split",to:"attn-k",fanout:"attn-five-way",departure:32}',
    '{from:"attn-split",to:"attn-v",fanout:"attn-five-way",departure:32}',
    '{from:"attn-split",to:"attn-qidx",fanout:"attn-five-way",departure:32}',
    '{from:"attn-split",to:"attn-kidx",fanout:"attn-five-way",departure:32}',
    '{from:"attn-topids",to:"attn-qk",toPort:"top-right",approach:28}',
    '{from:"attn-paged-k",to:"attn-qk"}',
    '{from:"attn-qr",to:"attn-qk",toPort:"top-left",approach:28}',
    '{from:"attn-paged-v",to:"attn-pv",toPort:"top-right",approach:34}',
  ]) assert.ok(source.includes(edge), `missing separated attention edge ${edge}`);
});

test("QKV + Index Projection detail contains only five-way projection evidence", async () => {
  const source = readModelSource();
  assert.match(source, /const QKV_INDEX_PROJECTION_SECTIONS: CodeSection\[\] = \[/);
  assert.match(source, /packed:cloneOp\(packed,\{id:"s-packed",kind:"linear",title:"QKV \+ Index Projection"[^}]*codeSections:QKV_INDEX_PROJECTION_SECTIONS/);

  const start = source.indexOf("const QKV_INDEX_PROJECTION_SECTIONS");
  const end = source.indexOf("\nconst ", start + 1);
  const detail = source.slice(start, end);
  assert.doesNotMatch(detail, /ATTEND|self\.attn\(/);
  assert.match(detail, /(?:q_?(?:idx|index)|(?:idx|index)_q)/i);
  assert.match(detail, /(?:k_?(?:idx|index)|(?:idx|index)_k)/i);
  assert.match(detail, /output_sizes[^\n]*|\[q \| k \| v \| index_q \| index_k\]/i);
});

test("sparse attention sends symbolic Top-K indices and paged KV directly to attention math", async () => {
  const source = readModelSource();
  assert.match(source, /"s-idxmask":String\.raw/);
  assert.match(source, /idxnorm:cloneOp\([^\n]*title:"Index Q\/K Gemma RMSNorm \+ RoPE"/);
  assert.match(source, /idxmask:cloneOp\([^\n]*title:"Mask Future Index Keys"/);
  assert.doesNotMatch(source, /select:cloneOp\([^\n]*title:"Map Top-16 Blocks → KV Views"/);
  assert.match(source, /mask:cloneOp\([^\n]*title:"Apply Token Causal \/ Pad Mask"/);
  assert.match(source, /className="index-ribbon-label">LIGHTNING INDEXER<\/header>/);
  assert.match(source, /Tensor name="block_indices · Top-K_block" shape="\[B,N_idx,S,K_block\]"/);
  assert.doesNotMatch(source, /Tensor name="selected [KV] view"/);
  assert.match(source, /title:"Q × paged Kᵀ · Top-K_block"/);
  assert.match(source, /title:"P × paged V · same Top-K_block"/);
  assert.match(source, /shape=\{dense\?"\[B,Nₕ,S,T\]":"\[B,Nₕ,S,K_sel\]"\}/);
  for (const edge of [
    '{from:"attn-idxscore",to:"attn-idxmask"}',
    '{from:"attn-idxbounds",to:"attn-idxmask",fromPort:"left",toPort:"right"}',
    '{from:"attn-idxmask",to:"attn-blockmax"}',
  ]) assert.ok(source.includes(edge), `missing index mask edge ${edge}`);
  assert.match(source, /const SPARSE_MASK_SECTIONS: CodeSection\[\] = \[/);
});

test("vLLM index branch exposes its independent key-only side cache", async () => {
  const source = readModelSource();
  const css = readStyles();
  assert.match(source, /VLLM_INDEXER_URL/);
  assert.match(source, /const INDEX_CACHE_SECTIONS: CodeSection\[\] = \[/);
  assert.match(source, /MiniMaxM3IndexerCache/);
  assert.match(source, /self\.index_cache\.kv_cache/);
  assert.match(source, /"s-idxcache":String\.raw/);
  assert.match(source, /idxcache:cloneOp\([^\n]*title:"Index K Cache · key-only"/);
  assert.match(source, /Tensor name="Index Q query" shape="\[B,N_idx,S,D_idx\]" graphId="attn-idxquery"/);
  assert.match(source, /Tensor name="index slot_mapping" shape="\[Nq\]" role="side" graphId="attn-idxslots"/);
  for (const edge of [
    '{from:"attn-idxnorm",to:"attn-idxquery",fromPort:"bottom-left",approach:30}',
    '{from:"attn-idxnorm",to:"attn-idxcache"}',
    '{from:"attn-idxslots",to:"attn-idxcache",fromPort:"left",toPort:"right"}',
    '{from:"attn-idxquery",to:"attn-idxscore",toPort:"top-left",approach:30}',
    '{from:"attn-idxcache",to:"attn-idxscore"}',
  ]) assert.ok(source.includes(edge), `missing Index K cache edge ${edge}`);
  assert.match(css, /\[data-graph-id="attn-idxcache"\]\{grid-area:3\/2\}/);
  assert.match(css, /\[data-graph-id="attn-idxscore"\]\{grid-area:4\/2\}/);
});

test("Q and K RoPE inputs use symmetric top ports", async () => {
  const source = readModelSource();
  for (const edge of [
    '{from:"attn-qt",to:"attn-qrope",toPort:"top-left",approach:38}',
    '{from:"attn-posq",to:"attn-qrope",toPort:"top-right",approach:38}',
    '{from:"attn-kt",to:"attn-krope",toPort:"top-left",approach:38}',
    '{from:"attn-posk",to:"attn-krope",toPort:"top-right",approach:38}',
  ]) assert.ok(source.includes(edge), `missing symmetric RoPE edge ${edge}`);
  assert.doesNotMatch(source, /from:"attn-pos[ qk]+",to:"attn-[qk]rope",fromPort:"left",toPort:"right"/);
});

test("attention fan-out is shared and compact labels stay inside their nodes", async () => {
  const source = readModelSource();
  const css = readStyles();

  assert.match(source, /title:"Main Q\/K Gemma RMSNorm"/);
  for (const target of ["attn-q", "attn-k", "attn-v", "attn-qidx", "attn-kidx"]) {
    assert.ok(source.includes(`to:"${target}",fanout:"attn-five-way"`), `missing shared fan-out target ${target}`);
  }
  assert.match(css, /\[data-graph-id="attn-idxbounds"\][^{]*\{[^}]*height:78px/);
  assert.match(css, /\[data-graph-id="attn-topids"\][^{]*\{[^}]*height:82px/);
  assert.match(css, /\[data-graph-id="attn-idxbounds"\] b[^}]*overflow-wrap:anywhere/);
  assert.match(css, /\[data-graph-id="attn-bounds"\][^{]*\{[^}]*width:min\(420px,100%\)/);
  assert.match(css, /\[data-graph-id="attn-bounds"\]\{grid-area:5\/1;justify-self:end\}/);
  assert.match(css, /\.attention-branches:not\(\.dense\)\{[^}]*grid-template-columns:max-content minmax\(600px,1fr\)/);
  assert.match(css, /\.index-ribbon\{[^}]*box-sizing:border-box[^}]*min-width:0[^}]*overflow:hidden/);
  assert.match(css, /\.index-ribbon>\[data-graph-id\][^{]*\{[^}]*box-sizing:border-box[^}]*max-width:100%/);
});

test("diagram geometry prefers straight paths, content-sized cards, and conditional panning", async () => {
  const source = readModelSource();
  const css = readStyles();

  assert.match(source, /\{from:"attn-p",to:"attn-pv",toPort:"top-left",approach:34\}/);
  assert.match(source, /const \[canPan,setCanPan\]=useState\(false\)/);
  assert.match(source, /content\.scrollWidth>viewport\.clientWidth\+1\|\|content\.scrollHeight>viewport\.clientHeight\+1/);
  assert.match(source, /\{canPan&&<span className="graph-pan-hint">/);
  assert.match(css, /\.graph-pan-viewport\.can-pan\{cursor:grab;touch-action:none\}/);
  assert.match(css, /\.graph-pan-content \.graph-surface :is\(\.op-node,\.tensor-node\)\{[^}]*width:fit-content!important[^}]*padding-inline:18px!important/);
  assert.match(css, /\.score-pipeline-label\{[^}]*grid-area:1\/1\/2\/4/);
  assert.match(source, /Tensor name="Qidx" shape="\[B,S,N_idx,D_idx\]"/);
  assert.match(source, /title:"Block Max · B_block keys"/);
});

test("keeps code, checkpoint, formula, and shape evidence together", async () => {
  const [page, modelData, css] = await Promise.all([
    readModelSource(),
    readFile(new URL("../app/models/minimax-m3/data.ts", import.meta.url), "utf8"),
    readStyles(),
  ]);
  const source = `${page}\n${modelData}`;

  assert.match(source, /MinimaxM3QKVParallelLinearWithIndexer/);
  assert.match(source, /block_sparse_moe\.experts\.0\.w1\.weight/);
  assert.match(source, /safetensors/);
  assert.match(source, /inputShape/);
  assert.match(source, /formula/);
  assert.match(source, /weights/);
  assert.match(source, /type OpKind/);
  assert.match(source, /Top-K_block Blocks/);
  assert.match(source, /Q × paged Kᵀ · Top-K_block/);
  assert.match(source, /side:"EXTERNAL"/);
  assert.match(source, /position\(req,i\)=num_computed_tokens\[req\]\+i/);
  assert.match(source, /CommonAttentionMetadata/);
  assert.match(source, /compute_slot_mapping/);
  assert.match(source, /Apply Causal \/ Pad Bounds/);
  assert.match(source, /katex\.renderToString/);
  assert.match(source, /LATEX_BY_ID/);
  assert.match(source, /实际公式/);
  assert.match(source, /operatorname\{TopK\}_K/);
  assert.match(source, /Q_\{\\mathrm\{rot\}\},Q_\{\\mathrm\{pass\}\}/);
  assert.match(source, /operatorname\{Concat\}\(\\operatorname\{RoPE\}/);
  assert.match(source, /先把每个 128 维 Q head 拆成两个 64 维分段/);
  assert.doesNotMatch(source, /权重名称为什么与代码不同/);
  assert.match(source, /SiluAndMulWithClamp/);
  assert.match(source, /self\.act_fn/);
  assert.match(source, /forward_native/);
  assert.match(source, /torch\.clamp\(x\[\.\.\., :d\], max=self\.swiglu_limit\)/);
  assert.match(source, /IMPLEMENTATION TRACE/);
  assert.doesNotMatch(source, /forward → fused kernel → 数学定义/);
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
  assert.match(source, /id: "minimax-m3"/);
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
  assert.match(source, /<Tensor name="G⁽ʳ⁾" shape="\[B,S,H_dense\/TP\]" graphId="mlp-gate"\/>/);
  assert.match(source, /<Tensor name="U⁽ʳ⁾" shape="\[B,S,H_dense\/TP\]" graphId="mlp-up"\/>/);
  assert.match(source, /data-graph-id="mlp-gate-act"[\s\S]{0,700}min\(G⁽ʳ⁾, C\) · σ\(α·min\(G⁽ʳ⁾, C\)\)/);
  assert.match(source, /data-graph-id="mlp-up-act"[\s\S]{0,700}clip\(U⁽ʳ⁾, −C, C\) \+ β/);
  assert.match(source, /title="逐元素相乘"[\s\S]{0,300}<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4\.25 4\.25 11\.75 11\.75M11\.75 4\.25 4\.25 11\.75"\/><\/svg><\/button>/);
  assert.match(source, /className="mini-math activation-step"/);
  assert.match(source, /title:"SwiGLU-OAI"/);
  assert.match(source, /runtime:"vLLM: SiluAndMulWithClamp · layout: swigluoai_uninterleave"/);
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
  const termBlock = source.match(/const FORMULA_TERMS_BY_ID:[\s\S]*?function formulaTerms/)?.[0] ?? "";
  const latexIds = [...latexBlock.matchAll(/"([ds]-[^"]+)":String\.raw/g)].map((match) => match[1]);
  const termIds = new Set([...termBlock.matchAll(/"([ds]-[^"]+)":\[/g)].map((match) => match[1]));
  assert.deepEqual(latexIds.filter((id) => !termIds.has(id)), [], "every dedicated formula needs matching dedicated terms");
  assert.match(source, /"s-shared":\[\["u","Shared Expert 输入/);
  assert.match(source, /\["W₁,s","shared gate_proj\.weight"\]/);
  assert.doesNotMatch(source, /FORMULA_TERMS_BY_ID\[node\.id\]\?\?FORMULA_TERMS_BY_KIND/);
  assert.match(source, /id:"d-add1",kind:"add",kicker:"DECODER LAYER · ATTENTION RESIDUAL"/);
  assert.match(source, /source:"nvidia\/model\.py · MiniMaxM3DecoderLayer\.forward · L773–775"/);
  assert.match(source, /id:"d-add2",kind:"add",kicker:"DECODER LAYER · FFN RESIDUAL"/);
  assert.match(source, /下一 Decoder Layer 在 L758–767 的 fused input RMSNorm 中执行实际 add/);
  assert.match(source, /"d-add2":\[\["U","Attention 后的 residual stream/);
  assert.doesNotMatch(source, /title:"\+ MLP Residual"/);
  assert.doesNotMatch(page, /DECODER LAYER TYPE|Dense GQA · SwiGLU MLP|Indexer Attention · Top-4 MoE/);
  assert.match(source, /block_sparse_moe\.gate\.weight/);
  assert.match(source, /name="routed expert weights · correction bias"/);
  assert.match(source, /title:"Gemma RMSNorm"/);
  assert.match(source, /title:"Post-attn Gemma RMSNorm"/);
  assert.match(source, /title:"Q Gemma RMSNorm · per-head",summary:"对每个 Q head 的 128 维向量独立执行 Gemma 风格 RMSNorm。"/);
  assert.match(source, /title:"K Gemma RMSNorm · per-head",summary:"对每个 K head 的 128 维向量独立执行 Gemma 风格 RMSNorm。"/);
  assert.match(source, /for\(const id of \["d-qnorm","d-knorm","s-mainnorm"\]\) CODE_BY_ID\[id\]=\{sections:QK_NORM_SECTIONS,symbols:QK_NORM_SYMBOLS\};/);
  assert.match(source, /className="score-pipeline-label">ATTENTION SCORE PIPELINE · 候选 blocks 内计算概率 P<\/header>/);
  assert.doesNotMatch(source, /本节点不执行 RoPE 或 Attention/);
  assert.match(source, /"d-qnorm":String\.raw`\\begin\{aligned\}\\operatorname\{RMS\}\(Q_\{b,h,s\}\)[\s\S]*\\tilde Q_\{b,h,s,i\}[\s\S]*\\operatorname\{RMS\}\(Q_\{b,h,s\}\)[\s\S]*\\end\{aligned\}`/);
  assert.match(source, /"d-knorm":String\.raw`\\begin\{aligned\}\\operatorname\{RMS\}\(K_\{b,g,s\}\)[\s\S]*\\tilde K_\{b,g,s,i\}[\s\S]*\\operatorname\{RMS\}\(K_\{b,g,s\}\)[\s\S]*\\end\{aligned\}`/);
  assert.match(source, /title:"Partial RoPE \(Q\)",summary:"仅对每个 Q head 的前 64\/128 维应用 RoPE，后 64 维保持不变。"/);
  assert.match(source, /"d-ropeq":String\.raw`\\begin\{aligned\}\(Q_\{\\mathrm\{rot\}\},Q_\{\\mathrm\{pass\}\}\)&=\\operatorname\{Split\}/);
  assert.match(source, /TRANSFORMERS_MINIMAX_M3_URL/);
  assert.match(source, /const detail=CODE_BY_ID\[values\.id\]/);
  assert.match(source, /const sections=node\.codeSections\?\?\[\];/);
  assert.match(source, /title:"Transformers：Partial RoPE 可读实现"/);
  assert.match(source, /q_rot, q_pass = q\[\.\.\., :rotary_dim\], q\[\.\.\., rotary_dim:\]/);
  assert.match(source, /for\(const id of \["d-ropeq","d-ropek","s-rope"\]\) CODE_BY_ID\[id\]=\{sections:\[VLLM_ROPE_SECTION,TRANSFORMERS_ROPE_SECTION\]/);
  assert.match(source, /title:"Paged KV Cache",summary:"L0–L2 的 Full GQA 从 Paged KV Cache 读取全部因果可见的历史与当前 K\/V。"/);
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
  assert.match(source, /Router 在 Python 层只产生 \[B,S,128\] 的 router_logits/);
  assert.match(source, /expert ids 与 router weights 由 FusedMoE 内部计算/);
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
  assert.match(source, /title="颜色区分算子类型"/);
  assert.match(css, /\.operator-swatch\{[^}]*linear-gradient\(90deg,var\(--linear\).*var\(--norm\).*var\(--split\).*var\(--activation\).*var\(--route\)/);
  assert.doesNotMatch(css, /\.runtime-io/);
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
  assert.match(css, /\.decoder-workbench\.has-zoom\{grid-template-columns:minmax\(0,1fr\);gap:0\}/);
  assert.match(source, /\{!expanded&&<GraphPan><GraphSurface className="decoder-column decoder-node-graph"/);
  assert.match(css, /\.decoder-workbench:not\(\.has-zoom\)>\.graph-pan-viewport/);
  assert.match(source, /\{expanded&&<StageZoom/);
  assert.match(css, /\.stage-zoom/);
  assert.match(css, /\.stage-zoom>header\{align-items:center\}/);
  assert.doesNotMatch(css, /\.parallel-experts/);
  assert.match(css, /\.add-circle/);
  assert.doesNotMatch(css, /\.weighted-op/);
  assert.match(css, /\.input-weighted-op/);
  assert.match(css, /\.co-input-row/);
  assert.doesNotMatch(css, /\.parallel-gate-up/);
  assert.match(css, /\.graph-connectors/);
  assert.doesNotMatch(css, /\.graph-arrowheads/);
  assert.match(css, /\.decoder-node-graph/);
  assert.match(css, /\.decoder-node-graph>\.input-weighted-op\{[^}]*row-gap:clamp\(44px,5vh,64px\)/);
  assert.match(css, /\.decoder-column\{[^}]*width:min\(680px,96%\)/);
  assert.match(css, /\.decoder-node-graph \.co-input-row\{[^}]*grid-template-columns:max-content max-content/);
  assert.match(css, /\.decoder-node-graph \.co-input-row>\.tensor-node\{[^}]*width:max-content/);
  assert.doesNotMatch(css, /\.decoder-node-graph>\.tensor-node\{[^}]*min-width:260px/);
  assert.match(css, /\.decoder-node-graph \.input-weighted-op>\.op-node\{[^}]*width:max-content[^}]*min-width:0/);
  assert.doesNotMatch(css, /\.decoder-node-graph \.input-weighted-op>\.op-node\{[^}]*min-width:220px/);
  assert.match(css, /\.decoder-node-graph>\.stage-summary\{[^}]*width:max-content/);
  assert.match(css, /\.connected-attention-graph/);
  assert.match(css, /\.mlp-node-graph/);
  assert.match(css, /\.mlp-node-graph\{width:min\(600px,96%\);padding:8px 8px\}/);
  assert.match(css, /\.mlp-node-graph>\[data-graph-id\]\{[^}]*width:max-content[^}]*max-width:260px/);
  assert.match(css, /\.stage-overview-panel\{grid-template-rows:auto minmax\(0,1fr\) 28px\}/);
  assert.match(css, /\.stage-formula-section code\{white-space:pre-line\}/);
  assert.match(css, /\.activation-step\{cursor:pointer\}/);
  assert.match(css, /\.multiply-circle\[aria-pressed="true"\]\{[^}]*outline:2px solid var\(--focus-ring\)[^}]*border-color:var\(--green\)/);
  assert.match(css, /\.mlp-node-graph \[data-graph-id="mlp-wdown"\]\{grid-area:9\/5\}/);
  assert.match(css, /\.moe-node-graph/);
  assert.match(css, /\.graph-pan-content \.moe-node-graph>\[data-graph-id="moe-experts"\]\{grid-area:4\/5\}/);
  assert.match(css, /\.graph-pan-content \.moe-node-graph>\[data-graph-id="moe-sum"\]\{grid-area:6\/5\/7\/8\}/);
  assert.match(css, /\.graph-pan-content \.moe-node-graph>\[data-graph-id="moe-y"\]\{grid-area:7\/5\/8\/8\}/);
  assert.match(css, /\.stage-zoom\{container-type:inline-size\}/);
  assert.match(css, /\.stage-zoom \.moe-node-graph\{grid-template-columns:repeat\(5,minmax\(0,1fr\)\);padding:8px clamp\(16px,1\.6vw,32px\)\}/);
  assert.match(css, /\.stage-zoom \.moe-node-graph :is\(\.tensor-node,\.op-node\)\{min-height:46px;max-height:none;padding:6px 9px;gap:2px;line-height:1\.1\}/);
  assert.match(css, /\.stage-zoom \.moe-node-graph \.tensor-weight\{width:210px;max-width:100%;min-height:60px;padding:8px 11px\}/);
  assert.match(source, /const safeClearance=direction==="side-left"/);
  assert.match(source, /Math\.min\(GRAPH_GEOMETRY.clearance,Math\.max\(4,obstacleBounds\.left-GRAPH_GEOMETRY.arrowClearance\)\)/);
  assert.match(css, /\.shape-rows/);
  assert.match(css, /\.shape-rows code\{[^}]*min-width:0[^}]*overflow:visible[^}]*text-overflow:clip/);
  assert.match(css, /@media\(min-width:1160px\)\{\.screen-grid\{grid-template-columns:minmax\(680px,1fr\) 460px\}\}/);
  assert.match(css, /\.shape-rows\{display:flex!important;flex-direction:column;align-items:stretch;gap:6px\}/);
  assert.match(css, /\.shape-rows>span\{width:100%;grid-template-columns:46px minmax\(0,1fr\)\}/);
  assert.match(css, /\.shape-rows code\{white-space:nowrap;overflow-wrap:normal!important\}/);
  assert.doesNotMatch(css, /\.binding,\.output-binding article\{container-type:inline-size\}|@container \(max-width:430px\)/);
  assert.match(css, /\.model-facts small\{[^}]*font-size:12px/);
  assert.match(css, /\.tensor-node code\{font-size:12px/);
  assert.match(css, /\.stage-zoom \.graph-surface \.tensor-node code\{font-size:12px/);
  assert.match(css, /\.shape-rows code\{font:12px/);
  assert.match(css, /\.model-overview \.model-step code\{font-size:12px/);
  assert.match(css, /@container \(max-width:1000px\)\{[^}]*font-size:10px!important/);
  assert.match(css, /column-gap:15px/);
  const fixedPixelFontSizes = [...css.matchAll(/(?:font-size|font):(\d+(?:\.\d+)?)px/g)]
    .map((match) => Number(match[1]));
  assert.ok(fixedPixelFontSizes.every((size) => size >= 10), "compact graph text may shrink to 10px but no smaller");
  assert.match(css, /\.config-reference/);
  assert.match(css, /\.config-reference\{[^}]*overflow-x:hidden/);
  assert.match(css, /\.config-reference table\{[^}]*table-layout:fixed/);
  assert.match(css, /\.config-tabs\{[^}]*flex-wrap:wrap/);
  assert.doesNotMatch(css, /font-weight:(?:750|800)/);
  assert.match(css, /\.detail-formula\{overflow:auto/);
  assert.match(css, /\.unpin-button\{[^}]*width:28px[^}]*height:28px/);
  assert.doesNotMatch(css, /\.op-node\{[^}]*border-left/);
});

test("renders every operator equation as valid LaTeX", async () => {
  const page = readModelSource();
  const equations = [...page.matchAll(/String\.raw`([^`]*)`/g)].map((match) => match[1]);

  assert.ok(equations.length >= 40, `expected a complete formula set, got ${equations.length}`);
  for (const equation of equations) {
    assert.doesNotThrow(() => katex.renderToString(equation, { throwOnError: true }));
  }
});

test("routes MoE branches on symmetric rails without crossing nodes", async () => {
  const source = readModelSource();
  const css = readStyles();

  assert.match(
    source,
    /\{from:"moe-u",to:"moe-shared",fromPort:"bottom-right",toPort:"top",approach:28\}/,
  );
  assert.doesNotMatch(
    source,
    /\{from:"moe-u",to:"moe-shared",route:"side-right",fromPort:"right",toPort:"right"\}/,
  );
  assert.doesNotMatch(source, /graphId="moe-expert-input"/);
  assert.match(source, /\{from:"moe-u",to:"moe-experts"\}/);
  assert.match(source, /\{from:"moe-router-logits",to:"moe-experts",toPort:"top-left",approach:28\}/);
  assert.doesNotMatch(source, /className="moe-expert-branches"/);
  assert.match(source, /\{from:"moe-routed",to:"moe-sum",toPort:"top-left",approach:24\}/);
  assert.match(source, /\{from:"moe-shared-out",to:"moe-sum",toPort:"top-right",approach:24\}/);
  assert.match(source, /name="shared expert weights ×3" shape="gate \/ up \/ down"/);
  assert.doesNotMatch(source, /name="block_sparse_moe\.shared_experts\.\{gate_proj,up_proj,down_proj\}\.weight"/);

  const edgeBlock = source.match(/const edges:GraphEdge\[\]=\[\s*\{from:"moe-u"([\s\S]*?)\n\s*\];/)?.[0] ?? "";
  const artifacts = new Set([
    "moe-u", "moe-wrouter", "moe-router-logits", "moe-wexperts",
    "moe-routed", "moe-wshared", "moe-shared-out", "moe-y",
  ]);
  const edges = [...edgeBlock.matchAll(/\{from:"([^"]+)",to:"([^"]+)"/g)];
  assert.equal(edges.length, 13, "expected every MoE graph edge in the invariant check");
  for (const [, from, to] of edges) {
    assert.notEqual(
      artifacts.has(from),
      artifacts.has(to),
      `${from} → ${to} must alternate tensor/weight artifacts and operators`,
    );
  }
});
