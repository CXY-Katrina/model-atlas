import assert from "node:assert/strict";
import test from "node:test";

import { routeGraphEdge, routeGraphFanout } from "../app/graph/routing.ts";

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
  assert.match(route.path, /^M 120 60 L 120 /);
  assert.match(route.path, / Q /, "offset vertical routes use rounded orthogonal corners");
  assert.match(route.path, /Q 150 136, 150 [\d.]+ L 150 150$/, "the last segment must enter the arrowhead through the center of its horizontal base");
  assert.match(route.path, /136/, "the horizontal turn stays in the target's clear approach lane");
});

test("aligned vertical routes stay straight from top to bottom", () => {
  for (const targetX of [150, 150.01, 149.99]) {
    const route = routeGraphEdge({
      source: { x: 150, y: 60 },
      target: { x: targetX, y: 150 },
      direction: "vertical",
      obstacleBounds: { left: 0, right: 300 },
      clearance: 24,
    });

    assert.equal(route.path, "M 150 60 L 150 150");
  }
});

test("long approaches keep a straight segment between the final curve and arrow", () => {
  const route = routeGraphEdge({
    source: { x: 120, y: 60 },
    target: { x: 200, y: 180 },
    direction: "vertical",
    obstacleBounds: { left: 0, right: 300 },
    clearance: 24,
    approach: 48,
  });

  assert.match(route.path, /Q 200 132, 200 156 L 200 180$/);
});

test("bus routes share a smooth outer rail without crossing the middle", () => {
  const route = routeGraphEdge({
    source: { x: 180, y: 90 },
    target: { x: 80, y: 330 },
    direction: "bus-right",
    obstacleBounds: { left: 40, right: 286 },
    clearance: 24,
    approach: 42,
  });

  assert.equal(route.rail, 310);
  assert.match(route.path, /^M 180 90 L 180 /);
  assert.match(route.path, /Q 310 /, "the bus uses rounded entry and exit corners");
  assert.match(route.path, /L 80 330$/, "the final segment enters the target vertically");
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

test("vertical fan-out honors one shared departure height", () => {
  const options = {
    source: { x: 100, y: 20 },
    direction: "vertical",
    obstacleBounds: { left: 0, right: 200 },
    clearance: 24,
    approach: 40,
    departure: 36,
  };
  const left = routeGraphEdge({ ...options, target: { x: 20, y: 200 } });
  const right = routeGraphEdge({ ...options, target: { x: 180, y: 240 } });
  const firstTurnY = (path) => Number(path.match(/Q 100 ([\d.]+),/)?.[1]);

  assert.equal(firstTurnY(left.path), 56);
  assert.equal(firstTurnY(right.path), 56);
});

test("shared fan-out uses one trunk and one aligned rail", () => {
  const routes = routeGraphFanout({
    source: { x: 300, y: 40 },
    targets: [
      { x: 80, y: 180 },
      { x: 190, y: 180 },
      { x: 300, y: 180 },
      { x: 410, y: 180 },
      { x: 520, y: 180 },
    ],
    departure: 64,
  });

  assert.equal(routes.filter(route => route.role === "trunk").length, 1);
  assert.equal(routes.filter(route => route.role === "rail").length, 2);
  assert.equal(routes.filter(route => route.role === "drop").length, 5);
  for (const route of routes.filter(route => route.role === "drop")) {
    assert.match(route.path, /^M ([\d.]+) (?:104|128) L \1 180$/);
    assert.equal(route.arrow, true);
  }
  assert.ok(routes.filter(route => route.role !== "drop").every(route => route.arrow === false));
  assert.match(routes.find(route => route.role === "rail" && route.path.includes("80"))?.path ?? "", / Q /);
});
