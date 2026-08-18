import assert from "node:assert/strict";
import test from "node:test";

import { nextDetailState } from "../app/detail-selection.ts";

test("click pins a module after the pointer leaves it", () => {
  const qkv = { id: "qkv", title: "QKV + Index Projection" };
  let state = { hovered: null, pinned: null };

  state = nextDetailState(state, { type: "hover", node: qkv });
  state = nextDetailState(state, { type: "pin", node: qkv });
  state = nextDetailState(state, { type: "leave" });

  assert.equal(state.hovered, null);
  assert.equal(state.pinned, qkv);
  assert.equal(state.pinned ?? state.hovered, qkv);
});

test("a later click replaces the pinned module", () => {
  const qkv = { id: "qkv", title: "QKV + Index Projection" };
  const router = { id: "router", title: "FP32 Router → Top-4" };
  let state = { hovered: null, pinned: qkv };

  state = nextDetailState(state, { type: "hover", node: router });
  state = nextDetailState(state, { type: "pin", node: router });
  state = nextDetailState(state, { type: "leave" });

  assert.equal(state.pinned, router);
  assert.equal(state.pinned ?? state.hovered, router);
});
