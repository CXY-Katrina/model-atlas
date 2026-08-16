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
  assert.match(html, /MiniMax Sparse Attention \+ MoE/);
  assert.match(html, /QKV \+ Index Projection/);
  assert.match(html, /INPUT/);
  assert.match(html, /OPERATOR/);
  assert.match(html, /OUTPUT/);
  assert.match(html, /TENSOR/);
  assert.match(html, /packed_5/);
  assert.match(html, /Top-16 block ids/);
  assert.match(html, /selected K · V/);
  assert.match(html, /ATTENTION RUNTIME I\/O/);
  assert.match(html, /Build Position IDs/);
  assert.match(html, /Build Attention Metadata/);
  assert.match(html, /query_start_loc · seq_lens · causal/);
  assert.match(html, /implicit · 非稠密 \[S,T\]/);
  assert.match(html, /slot_mapping · block_table → KV Cache/);
  assert.match(html, /Xₗ₊₁ · hidden_states/);
  assert.match(html, /model-00003-of-00059\.safetensors/);
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
  assert.match(source, /SIDE INPUT/);
  assert.match(source, /中间张量/);
  assert.match(source, /position\(req,i\)=num_computed_tokens\[req\]\+i/);
  assert.match(source, /CommonAttentionMetadata/);
  assert.match(source, /compute_slot_mapping/);
  assert.match(source, /Apply Causal \/ Pad Bounds/);
  assert.match(source, /mask 在 vLLM 中不是显式 \[S,T\] 张量/);
  assert.match(source, /katex\.renderToString/);
  assert.match(source, /LATEX_BY_ID/);
  assert.match(source, /LATEX · FULL COMPUTE/);
  assert.match(source, /operatorname\{TopK\}_4/);
  assert.match(source, /theta_\{p,j\}/);
  assert.match(source, /Checkpoint → vLLM runtime/);
  assert.match(source, /type="range"/);
  assert.match(source, /MODEL_REGISTRY/);
  assert.match(css, /height:100svh/);
  assert.match(css, /overflow:hidden/);
  assert.match(css, /\.tensor-node/);
  assert.match(css, /tensor artifact → compute operator → tensor artifact/);
  assert.match(css, /\.runtime-io/);
  assert.match(css, /\.tensor-input/);
  assert.match(css, /\.tensor-output/);
  assert.match(css, /\.latex-render/);
  assert.match(css, /\.katex-display/);
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
