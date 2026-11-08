import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { perfettoMessage, startHandshake, stepHandshake, type Effect, type Handshake, type HandshakeEvent } from "./perfetto.ts";

function run(events: HandshakeEvent[], start: Handshake = startHandshake(0, 1000)): { state: Handshake; effects: Effect["kind"][] } {
  let state = start;
  const effects: Effect["kind"][] = [];
  for (const e of events) {
    const [next, effect] = stepHandshake(state, e);
    state = next;
    effects.push(effect.kind);
  }
  return { state, effects };
}

const buffer = new TextEncoder().encode('{"traceEvents":[]}').buffer;
const tick = (now: number, closed = false): HandshakeEvent => ({ type: "tick", now, closed });

describe("perfetto handshake", () => {
  it("pings on every tick until the UI answers", () => {
    const { state, effects } = run([tick(0), tick(250), tick(500)]);
    assert.deepEqual(effects, ["ping", "ping", "ping"]);
    assert.equal(state.phase, "waiting");
  });

  it("posts the trace once both the PONG and the bytes are in, in either order", () => {
    for (const order of [
      [{ type: "pong" }, { type: "loaded", buffer }],
      [{ type: "loaded", buffer }, { type: "pong" }],
    ] satisfies HandshakeEvent[][]) {
      const { state, effects } = run([tick(0), ...order]);
      assert.deepEqual(effects, ["ping", "none", "post"]);
      assert.equal(state.phase, "sent");
    }
  });

  it("stops pinging after the PONG while the trace is still loading", () => {
    const { effects } = run([tick(0), { type: "pong" }, tick(250), tick(5000)]);
    assert.deepEqual(effects, ["ping", "none", "none", "none"]);
  });

  it("never posts twice when more PONGs arrive", () => {
    const { effects } = run([{ type: "loaded", buffer }, { type: "pong" }, { type: "pong" }, tick(250)]);
    assert.deepEqual(effects, ["none", "post", "none", "none"]);
  });

  it("gives up when the UI never answers", () => {
    const { state } = run([tick(0), tick(999), tick(1000)]);
    assert.equal(state.phase, "failed");
  });

  it("fails when the tab is closed or the trace can't load", () => {
    assert.equal(run([tick(0), tick(250, true)]).state.phase, "failed");
    const failed = run([{ type: "pong" }, { type: "loadFailed", reason: "404 Not Found" }]).state;
    assert.deepEqual(failed, { phase: "failed", reason: "404 Not Found" });
  });

  it("builds the documented message shape", () => {
    const msg = perfettoMessage(buffer, { title: "Cascade run-1", fileName: "run-1-perfetto.json", url: "https://cascade.example/runs/run-1" });
    assert.deepEqual(Object.keys(msg), ["perfetto"]);
    assert.equal(msg.perfetto.buffer, buffer);
    assert.equal(msg.perfetto.title, "Cascade run-1");
    assert.equal(msg.perfetto.url, "https://cascade.example/runs/run-1");
  });
});
