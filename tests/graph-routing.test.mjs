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
  assert.match(route.path, /^M 120 60 C 120 98, 150 98, 150 136 L 150 150$/);
  assert.match(route.path, /150 136 L 150 150$/, "the last segment must enter the arrowhead through the center of its horizontal base");
});

test("horizontal routes finish perpendicular to the arrowhead base", () => {
  const route = routeGraphEdge({
    source: { x: 60, y: 120 },
    target: { x: 150, y: 150 },
    direction: "horizontal",
    obstacleBounds: { left: 0, right: 300 },
    clearance: 24,
  });

  assert.equal(route.path, "M 60 120 C 98 120, 98 150, 136 150 L 150 150");
});
