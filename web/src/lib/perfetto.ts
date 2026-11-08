// Opens ui.perfetto.dev and hands it a trace with its postMessage protocol:
// https://perfetto.dev/docs/visualization/deep-linking-to-perfetto-ui

export const PERFETTO_ORIGIN = "https://ui.perfetto.dev";
const PING_MS = 250;
const PONG_TIMEOUT_MS = 30_000;

export interface PerfettoTrace {
  title: string;
  fileName: string;
  url: string;
}

export type Handshake =
  | { phase: "waiting"; deadline: number; ponged: boolean; buffer: ArrayBuffer | null }
  | { phase: "sent" }
  | { phase: "failed"; reason: string };

export type HandshakeEvent =
  | { type: "tick"; now: number; closed: boolean }
  | { type: "pong" }
  | { type: "loaded"; buffer: ArrayBuffer }
  | { type: "loadFailed"; reason: string };

export type Effect = { kind: "ping" } | { kind: "post"; buffer: ArrayBuffer } | { kind: "none" };

export const startHandshake = (now: number, timeoutMs = PONG_TIMEOUT_MS): Handshake => ({ phase: "waiting", deadline: now + timeoutMs, ponged: false, buffer: null });

const NONE: Effect = { kind: "none" };

/** The UI drops messages that arrive before its listener is up, so PING until it PONGs, then post the trace once. */
export function stepHandshake(state: Handshake, event: HandshakeEvent): [Handshake, Effect] {
  if (state.phase !== "waiting") return [state, NONE];
  let next = state;
  switch (event.type) {
    case "tick":
      if (event.closed) return [{ phase: "failed", reason: "The Perfetto tab was closed before the trace loaded" }, NONE];
      if (state.ponged) return [state, NONE];
      if (event.now >= state.deadline) return [{ phase: "failed", reason: "ui.perfetto.dev didn't answer; check your connection and try again" }, NONE];
      return [state, { kind: "ping" }];
    case "loadFailed":
      return [{ phase: "failed", reason: event.reason }, NONE];
    case "pong":
      next = { ...state, ponged: true };
      break;
    case "loaded":
      next = { ...state, buffer: event.buffer };
      break;
  }
  return next.ponged && next.buffer !== null ? [{ phase: "sent" }, { kind: "post", buffer: next.buffer }] : [next, NONE];
}

export const perfettoMessage = (buffer: ArrayBuffer, trace: PerfettoTrace) => ({ perfetto: { buffer, ...trace } });

/** Call straight from the click handler: the tab must open inside the user gesture, before `load` is awaited. */
export function openInPerfetto(load: () => Promise<ArrayBuffer>, trace: PerfettoTrace): Promise<void> {
  const win = window.open(PERFETTO_ORIGIN);
  if (win === null) return Promise.reject(new Error("The browser blocked the Perfetto tab; allow pop-ups for this site and try again"));
  return new Promise((resolve, reject) => {
    let state = startHandshake(Date.now());
    const apply = (event: HandshakeEvent) => {
      const [next, effect] = stepHandshake(state, event);
      state = next;
      if (effect.kind === "ping") win.postMessage("PING", PERFETTO_ORIGIN);
      if (effect.kind === "post") win.postMessage(perfettoMessage(effect.buffer, trace), PERFETTO_ORIGIN);
      if (state.phase === "waiting") return;
      clearInterval(timer);
      window.removeEventListener("message", onMessage);
      if (state.phase === "sent") resolve();
      else {
        if (!win.closed) win.close();
        reject(new Error(state.reason));
      }
    };
    const onMessage = (e: MessageEvent) => {
      if (e.source === win && e.origin === PERFETTO_ORIGIN && e.data === "PONG") apply({ type: "pong" });
    };
    window.addEventListener("message", onMessage);
    const timer = setInterval(() => {
      apply({ type: "tick", now: Date.now(), closed: win.closed });
    }, PING_MS);
    load().then(
      (buffer) => {
        apply({ type: "loaded", buffer });
      },
      (error: unknown) => {
        apply({ type: "loadFailed", reason: error instanceof Error ? error.message : "Couldn't load the trace" });
      },
    );
  });
}
