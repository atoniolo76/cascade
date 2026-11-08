import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { activity, initialState, reducer, summary, type Action, type OnboardingState } from "./store.ts";

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
