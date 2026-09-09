import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { createModelRegistry } from "../app/models/registry.ts";

const model = id => ({ id, name: id, resources: [], facts: [], View: () => createElement("p", null, `${id} architecture`) });

test("model registry resolves registered models and rejects ambiguous registrations", () => {
  const first = model("first-model"), second = model("second-model");
  const registry = createModelRegistry([first, second]);
  assert.equal(registry.resolve("second-model"), second);
  assert.equal(registry.resolve("missing"), first);
  assert.equal(registry.resolve(), first);
  assert.throws(() => createModelRegistry([]), /at least one/);
  assert.throws(() => createModelRegistry([first, first]), /Duplicate/);
  assert.throws(() => createModelRegistry([model("bad id")]), /Invalid/);
});

test("shell renders independent model adapters without MiniMax assumptions", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { default: Home } = await server.ssrLoadModule("/app/page.tsx");
    const first = model("encoder-only"), second = model("decoder-only");
    for (const selected of [first, second]) {
      const registry = createModelRegistry([selected, selected === first ? second : first]);
      const html = renderToStaticMarkup(createElement(Home, { registry }));
      assert.ok(html.includes(`${selected.id} architecture`));
      assert.match(html, /encoder-only/);
      assert.match(html, /decoder-only/);
      assert.doesNotMatch(html, /MiniMax|QKV|428B|待添加/);
    }
  } finally { await server.close(); }
});

test("shared styling uses tokens and model geometry is scoped", () => {
  const read = file => readFileSync(new URL(`../app/${file}`, import.meta.url), "utf8");
  const tokens = read("styles/tokens.css"), shared = read("styles/ui.css"), layout = read("models/minimax-m3/layout.css");
  assert.match(tokens, /--graph-node-padding-inline:\s*var\(--space-18\)/);
  assert.match(tokens, /--graph-group-padding-inline:\s*var\(--space-24\)/);
  assert.match(tokens, /\.atlas-app.dark/);
  assert.doesNotMatch(shared + layout, /#[a-f\d]{3,8}\b/i);
  assert.match(shared, /padding-inline:\s*var\(--graph-node-padding-inline\)/);
  assert.match(layout, /--group-padding-inline:\s*var\(--graph-group-padding-inline\)/);
  assert.doesNotMatch(layout, /^\.(?:qkv-lanes|index-ribbon|connected-attention-graph|moe-node-graph)/m);
  assert.match(layout, /:where\(\[data-model="minimax-m3"\]\)/);
  assert.doesNotMatch(shared + layout, /\.runtime-io|\.parallel-gate-up|\.selection-chain/);
});
