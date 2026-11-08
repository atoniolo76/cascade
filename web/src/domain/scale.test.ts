import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { formatErrorTick, formatTimeTick, niceTicks, percentAlong, symmetricTicks, timeUnitFor } from "./scale.ts";

describe("niceTicks", () => {
  it("steps by 1, 2, 2.5 or 5 and ends at or above the max", () => {
    assert.deepEqual(niceTicks(13_900, 4), [0, 5000, 10_000, 15_000]);
    assert.deepEqual(niceTicks(4, 4), [0, 1, 2, 3, 4]);
    assert.deepEqual(niceTicks(0.37, 4), [0, 0.1, 0.2, 0.3, 0.4]);
    assert.deepEqual(niceTicks(212_000, 4), [0, 50_000, 100_000, 150_000, 200_000, 250_000]);
  });

  it("gives a unit axis when there is nothing to scale", () => {
    assert.deepEqual(niceTicks(0), [0, 1]);
    assert.deepEqual(niceTicks(Number.NaN), [0, 1]);
  });
});

describe("symmetricTicks", () => {
  it("centers on zero with the same steps on both sides", () => {
    assert.deepEqual(symmetricTicks(0.52, 2), [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]);
    assert.deepEqual(symmetricTicks(0.08, 2), [-0.1, -0.05, 0, 0.05, 0.1]);
  });

  it("keeps a usable axis when every error is zero", () => {
    assert.deepEqual(symmetricTicks(0, 2), [-1, -0.5, 0, 0.5, 1]);
  });
});

describe("timeUnitFor and formatTimeTick", () => {
  it("picks the largest unit the value reaches", () => {
    assert.equal(timeUnitFor(950).unit, "ns");
    assert.equal(timeUnitFor(1_000).unit, "µs");
    assert.equal(timeUnitFor(250_000).unit, "µs");
    assert.equal(timeUnitFor(2_500_000).unit, "ms");
    assert.equal(timeUnitFor(3e9).unit, "s");
  });

  it("labels every tick in the axis unit", () => {
    const u = timeUnitFor(15_000);
    assert.deepEqual([0, 5000, 10_000, 15_000].map((v) => formatTimeTick(v, u)), ["0", "5 µs", "10 µs", "15 µs"]);
    assert.equal(formatTimeTick(500, u), "0.5 µs");
  });
});

describe("formatErrorTick", () => {
  it("signs both sides and leaves zero bare", () => {
    assert.deepEqual([-0.4, 0, 0.25, 0.125].map(formatErrorTick), ["−40%", "0", "+25%", "+12.5%"]);
  });
});

describe("percentAlong", () => {
  it("maps into 0–100 and clamps outside the axis", () => {
    assert.equal(percentAlong(0, -0.5, 0.5), 50);
    assert.equal(percentAlong(0.25, 0, 1), 25);
    assert.equal(percentAlong(2, 0, 1), 100);
    assert.equal(percentAlong(1, 1, 1), 0);
  });
});
