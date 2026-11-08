import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { ComparisonRow } from "../api/types";
import { configAccuracy, errorExtent, relativeError } from "./accuracy.ts";

const row = (kernel_index: number, real_ns: number | null, sim_ns: Record<string, number | null>): ComparisonRow => ({
  kernel_index,
  kernel_name: `k${kernel_index}`,
  real_ns,
  sim_ns,
});

const rows = [row(0, 100, { a: 120, b: 90 }), row(1, 200, { a: 150, b: null }), row(2, null, { a: 999, b: 999 }), row(3, 50, { a: 50, b: 60 })];

describe("relativeError", () => {
  it("is signed against the real time and undefined without one", () => {
    assert.equal(relativeError(120, 100), 0.2);
    assert.equal(relativeError(80, 100), -0.2);
    assert.equal(relativeError(null, 100), null);
    assert.equal(relativeError(10, null), null);
    assert.equal(relativeError(10, 0), null);
  });
});

describe("configAccuracy", () => {
  it("averages only kernels with both times", () => {
    const a = configAccuracy(rows, "a");
    assert.ok(a);
    assert.equal(a.matched, 3);
    assert.ok(Math.abs(a.mape - (0.2 + 0.25 + 0) / 3) < 1e-12);
    assert.ok(Math.abs(a.bias - (0.2 - 0.25 + 0) / 3) < 1e-12);
    assert.deepEqual(a.total, { sim: 320, real: 350, error: (320 - 350) / 350 });
    assert.deepEqual(a.worst, { index: 1, name: "k1", error: -0.25 });
  });

  it("skips missing sims for that config only", () => {
    const b = configAccuracy(rows, "b");
    assert.ok(b);
    assert.equal(b.matched, 2);
    assert.deepEqual(b.worst, { index: 3, name: "k3", error: 0.2 });
  });

  it("is null when nothing matched yet", () => {
    assert.equal(configAccuracy([row(0, 100, { a: null })], "a"), null);
    assert.equal(configAccuracy(rows, "missing"), null);
  });
});

describe("errorExtent", () => {
  it("is the largest absolute error over the given configs", () => {
    assert.equal(errorExtent(rows, ["a", "b"]), 0.25);
    assert.equal(errorExtent(rows, ["b"]), 0.2);
    assert.equal(errorExtent([], ["a"]), 0);
  });
});
