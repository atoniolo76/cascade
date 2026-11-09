import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  activity,
  canAdvance,
  checklist,
  entriesLabel,
  errorPct,
  footer,
  initialState,
  MAX_REPLICAS,
  reducer,
  resultLine,
  runPhase,
  simStates,
  summary,
  type Action,
  type OnboardingState,
} from "./store.ts";

const run = (...actions: Action[]): OnboardingState => actions.reduce(reducer, initialState);

const ready = (): OnboardingState =>
  run(
    { type: "chooseTrace", trace: "opt125m-h100" },
    { type: "next" },
    { type: "setCount", config: "SM90_H100", count: 2 },
    { type: "setCount", config: "SM90_H200", count: 1 },
  );

describe("onboarding store", () => {
  it("needs a trace before step 1 advances", () => {
    assert.equal(run({ type: "next" }).step, 1);
    const s = run({ type: "chooseTrace", trace: "opt125m-a100" }, { type: "next" });
    assert.equal(s.step, 2);
    assert.equal(summary(s), "cascade-demo · OPT-125m on A100");
  });

  it("only takes configs the trace has recorded results for", () => {
    const s = run({ type: "chooseTrace", trace: "opt125m-a100" }, { type: "setCount", config: "SM90_H100", count: 2 });
    assert.deepEqual(s.entries, []);
  });

  it("adds, caps and removes configs in the order they were added", () => {
    const s = ready();
    assert.equal(entriesLabel(s.entries), "2× SM90_H100, 1× SM90_H200");
    assert.equal(reducer(s, { type: "setCount", config: "SM90_H100", count: 500 }).entries[0]?.count, MAX_REPLICAS);
    assert.equal(entriesLabel(reducer(s, { type: "setCount", config: "SM90_H100", count: 0 }).entries), "1× SM90_H200");
  });

  it("starts the GPU list over when the trace changes", () => {
    assert.deepEqual(reducer(ready(), { type: "chooseTrace", trace: "opt125m-a100" }).entries, []);
  });

  it("writes each choice into the activity timeline", () => {
    assert.deepEqual(
      activity(ready()).map((r) => `${r.lead} ${r.subject} · ${r.detail ?? ""}`),
      ["You created cascade-demo · ", "You opened OPT-125m on H100 · 712 kernels", "You added 2× SM90_H100 · 1980 MHz", "You added 1× SM90_H200 · 1980 MHz"],
    );
  });

  it("launches on step 3 and lands on done with the recorded result", () => {
    let s = reducer(ready(), { type: "next" });
    assert.equal(s.step, 3);
    assert.equal(s.run.status, "running");
    assert.equal(canAdvance(s), false);
    for (let i = 0; i < 30; i++) s = reducer(s, { type: "tick" });
    assert.equal(s.step, "done");
    assert.equal(activity(s).at(-1)?.detail, "+3.0% e2e vs real");
    assert.equal(checklist(s).at(-1)?.detail, "sim e2e 261.11 ms vs 253.44 ms real · +3.0%");
    assert.equal(footer(s), "1 run");
  });

  it("says when only H200 ran, which has no end-to-end comparison", () => {
    const s = reducer(ready(), { type: "setCount", config: "SM90_H100", count: 0 });
    assert.equal(resultLine(s), "Per-kernel times ready · no end-to-end comparison for H200");
  });

  it("drops the run when the user backs out of step 3", () => {
    const s = reducer(reducer(ready(), { type: "next" }), { type: "back" });
    assert.equal(s.step, 2);
    assert.equal(s.run.status, "idle");
  });

  it("forks, then finishes the sims one by one", () => {
    assert.deepEqual(simStates(3, 0.1), ["forking", "forking", "forking"]);
    assert.deepEqual(simStates(3, 0.5), ["done", "running", "running"]);
    assert.deepEqual(simStates(3, 0.9), ["done", "done", "done"]);
    assert.equal(runPhase(3, 0.1), "Forking 3 Sail sandboxes from the workspace checkpoint");
    assert.equal(runPhase(3, 0.5), "Replaying recorded Accel-Sim results · 1 of 3 done");
    assert.equal(runPhase(3, 0.9), "Comparing to the real run");
  });

  it("matches the samples' READMEs", () => {
    assert.equal(errorPct({ realE2eMs: 253.44, simE2eMs: 261.11 }), "+3.0%");
    assert.equal(errorPct({ realE2eMs: 222.08, simE2eMs: 229.58 }), "+3.4%");
  });
});
