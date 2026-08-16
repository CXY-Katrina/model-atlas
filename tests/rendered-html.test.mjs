import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

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
  assert.match(html, /model-00003-of-00059\.safetensors/);
  assert.match(html, /查看参数和符号说明/);
  assert.match(html, /og-operator-map\.png/);
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
  assert.match(source, /Checkpoint → vLLM runtime/);
  assert.match(source, /type="range"/);
  assert.match(source, /MODEL_REGISTRY/);
  assert.match(css, /height:100svh/);
  assert.match(css, /overflow:hidden/);
  assert.doesNotMatch(css, /border-left/);
});
