import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { SAMPLE_RUN_BLOCKED, noOwnTraces, runBlocker, sampleRunLine } from "./samples.ts";

describe("runBlocker", () => {
  it("lets a ready trace of your own run", () => {
    assert.equal(runBlocker({ state: "ready", sample: false }), null);
  });

  it("blocks a ready sample trace with the reason", () => {
    assert.equal(runBlocker({ state: "ready", sample: true }), SAMPLE_RUN_BLOCKED);
  });

  it("names the state of a trace that is not ready", () => {
    assert.equal(runBlocker({ state: "processing", sample: false }), "Trace is processing; runs need a ready trace");
  });
});

describe("noOwnTraces", () => {
  it("is true for an empty workspace and one holding only samples", () => {
    assert.equal(noOwnTraces([]), true);
    assert.equal(noOwnTraces([{ sample: true }]), true);
  });

  it("is false once a trace of your own arrives", () => {
    assert.equal(noOwnTraces([{ sample: true }, { sample: false }]), false);
  });
});

describe("sampleRunLine", () => {
  it("names the model and configs", () => {
    const run = { trace: { id: "t", model: "facebook/opt-125m" }, entries: [{ gpu_config: "SM90_H100", count: 1 }] };
    assert.equal(sampleRunLine(run), "Sample run: real Accel-Sim results for facebook/opt-125m on SM90_H100, recorded earlier.");
  });
});
