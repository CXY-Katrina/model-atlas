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
  assert.match(html, /从一张图追到一个张量/);
  assert.match(html, /END-TO-END FORWARD/);
  assert.match(html, /CHECKPOINT → RUNTIME/);
  assert.match(html, /59 safetensors shards/);
  assert.match(html, /q\/k\/v \+ index_q\/index_k/);
  assert.doesNotMatch(html, /Building your site|SkeletonPreview|codex-preview/);
});

test("keeps code, checkpoint, formula, and shape evidence together", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /MinimaxM3QKVParallelLinearWithIndexer/);
  assert.match(page, /block_sparse_moe\.experts\.0\.w1\.weight/);
  assert.match(page, /model\.safetensors\.index\.json/);
  assert.match(page, /safetensors/);
  assert.match(page, /shape/);
  assert.match(page, /formula/);
  assert.match(page, /weight/);
});
