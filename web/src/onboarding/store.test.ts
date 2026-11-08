import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { activity, canAdvance, checklist, fleetLabel, footer, initialState, MAX_PER_GPU, reducer, replayPhase, scriptedResult, summary, type Action, type OnboardingState } from "./store.ts";

const run = (...actions: Action[]): OnboardingState => actions.reduce(reducer, initialState);

describe("onboarding store", () => {
  it("shows a connecting source as a pending row, then confirms it", () => {
    const pending = run({ type: "connect", id: "modal" });
    assert.deepEqual(
      activity(pending).map((r) => [r.subject, r.detail, r.pending]),
      [
        ["cascade-demo", null, false],
        ["Modal", "cascade-demo", true],
      ],
    );
    assert.equal(activity(reducer(pending, { type: "confirm", id: "modal" }))[1]?.pending, false);
  });

  it("keeps a source gone when it is disconnected before it confirms", () => {
    const s = run({ type: "connect", id: "vllm" }, { type: "disconnect", id: "vllm" }, { type: "confirm", id: "vllm" });
    assert.deepEqual(s.sources, []);
  });

  it("names an upload by its file", () => {
    const s = run({ type: "connect", id: "upload", detail: "prod-week.parquet" }, { type: "confirm", id: "upload" });
    assert.equal(activity(s)[1]?.subject, "prod-week.parquet");
    assert.equal(summary(s), "cascade-demo · 1 trace source");
  });

  it("needs a connected source before step 1 advances", () => {
    assert.equal(run({ type: "next" }).step, 1);
    assert.equal(run({ type: "connect", id: "sample" }, { type: "next" }).step, 1);
    const s = run({ type: "connect", id: "sample" }, { type: "confirm", id: "sample" }, { type: "next" });
    assert.equal(s.step, 2);
    assert.equal(reducer(s, { type: "back" }).direction, -1);
  });
});

const ready = (): OnboardingState =>
  run(
    { type: "connect", id: "sample" },
    { type: "confirm", id: "sample" },
    { type: "next" },
    { type: "setCount", gpu: "h100", count: 2 },
    { type: "setCount", gpu: "l40s", count: 2 },
    { type: "next" },
    { type: "chooseRouter", router: "least-loaded" },
  );

describe("onboarding fleet, router and replay", () => {
  it("adds, changes and removes GPUs in the order they were added", () => {
    const s = run({ type: "setCount", gpu: "l40s", count: 2 }, { type: "setCount", gpu: "h100", count: 1 }, { type: "setCount", gpu: "h100", count: 3 });
    assert.equal(fleetLabel(s.fleet), "2× L40S, 3× H100");
    assert.deepEqual(reducer(s, { type: "setCount", gpu: "l40s", count: 0 }).fleet, [{ gpu: "h100", count: 3 }]);
    assert.equal(reducer(s, { type: "setCount", gpu: "h100", count: 99 }).fleet[1]?.count, MAX_PER_GPU);
  });

  it("writes the fleet and router into the activity timeline", () => {
    const rows = activity(ready()).slice(2);
    assert.deepEqual(
      rows.map((r) => `${r.lead} ${r.subject} · ${r.detail ?? ""}`),
      ["You added 2× H100 · measured", "You added 2× L40S · estimated", "You chose least-loaded · engine defaults"],
    );
    const custom = reducer(ready(), { type: "setEngine", engine: { batchSize: 16 } });
    assert.equal(activity(custom).at(-1)?.detail, "custom engine");
  });

  it("starts the replay on step 4 and finishes it after 25 ticks", () => {
    let s = reducer(ready(), { type: "next" });
    assert.equal(s.step, 4);
    assert.equal(s.replay.status, "running");
    assert.equal(canAdvance(s), false);
    for (let i = 0; i < 25; i++) s = reducer(s, { type: "tick" });
    assert.equal(s.replay.status, "done");
    assert.equal(activity(s).at(-1)?.detail, "2 hotspots on gpu2 and gpu1");
    assert.equal(footer(s), "1 replay");
    assert.equal(checklist(s).at(-1)?.detail, "p99 TTFT 1,105 ms · 2 hotspots · twin error ±6.8%");
    assert.equal(s.step, "done");
  });

  it("drops the replay when the user backs out of step 4", () => {
    const s = reducer(reducer(ready(), { type: "next" }), { type: "back" });
    assert.equal(s.step, 3);
    assert.equal(s.replay.status, "idle");
  });

  it("keeps hotspots on lanes that exist for a small fleet", () => {
    assert.deepEqual(scriptedResult([{ gpu: "a100", count: 1 }]).hotspots, [{ gpu: 0, cell: 17 }]);
  });

  it("narrates the replay phase by progress", () => {
    assert.equal(replayPhase(0.3), "Replaying request 600 of 1,200 · 00:47");
    assert.equal(replayPhase(0.5), "Replaying request 1,000 of 1,200 · 01:18");
    assert.equal(replayPhase(0.7), "Detecting hotspots");
    assert.equal(replayPhase(0.9), "Comparing to the real run");
  });
});
