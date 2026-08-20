import assert from "node:assert/strict";
import test from "node:test";

import { routeGraphEdge } from "../app/graph-routing.ts";

test("side routes clear every node before turning toward the target", () => {
  const route = routeGraphEdge({
    source: { x: 180, y: 90 },
    target: { x: 210, y: 330 },
    direction: "side-right",
    obstacleBounds: { left: 40, right: 286 },
    clearance: 24,
  });

  assert.equal(route.rail, 310);
  assert.match(route.path, /^M 180 90 L/);
  assert.match(route.path, / Q /);
  assert.match(route.path, /310/);
  assert.ok(route.rail > 286, "connector rail must sit outside the rightmost node");
});

test("direct routes keep the arrow on one continuous path", () => {
  const route = routeGraphEdge({
    source: { x: 120, y: 60 },
    target: { x: 150, y: 150 },
    direction: "vertical",
    obstacleBounds: { left: 0, right: 300 },
    clearance: 24,
  });

  assert.equal(route.rail, null);
  assert.match(route.path, /^M 120 60 C 120 105, 150 105, 150 150$/);
});
