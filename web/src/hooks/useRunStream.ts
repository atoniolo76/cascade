import { useEffect, useReducer, useRef, useState } from "react";
import { ApiError, apiJson, eventsUrl } from "../api/client";
import { fetchRunResults } from "../api/queries";
import type { RunDetail, RunEvent } from "../api/types";
import { initialRunLive, runLiveReducer } from "../domain/runLive";
import { deriveRunState } from "../domain/status";

export type Connection = "connecting" | "live" | "reconnecting" | "ended" | "missing";

const EVENT_TYPES = ["snapshot", "sim", "log", "kernel"] as const;
const BACKOFF_MS = [1000, 2000, 4000, 8000, 15000];

function isEventType(value: unknown): value is RunEvent["type"] {
  return typeof value === "string" && (EVENT_TYPES as readonly string[]).includes(value);
}

/** SSE boundary: accepts both unnamed `data:` messages and named events carrying a RunEvent body. */
function parseRunEvent(data: string, name: string): RunEvent | null {
  let body: unknown;
  try {
    body = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof body !== "object" || body === null) return null;
  const type = "type" in body ? body.type : name;
  if (!isEventType(type)) return null;
  return { ...body, type } as RunEvent;
}

export function useRunStream(runId: string) {
  const [state, dispatch] = useReducer(runLiveReducer, initialRunLive);
  const [connection, setConnection] = useState<Connection>("connecting");
  const terminal = state.kind === "live" && deriveRunState(state.run.sims.map((s) => s.state)) !== "active";
  const terminalRef = useRef(terminal);

  useEffect(() => {
    terminalRef.current = terminal;
  }, [terminal]);

  useEffect(() => {
    let source: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let disposed = false;

    const seedKernels = (run: RunDetail) => {
      if (!run.sims.some((s) => s.kernels_done > 0)) return;
      fetchRunResults(runId)
        .then((results) => {
          if (!disposed) dispatch({ type: "kernels", stats: results.sims.flatMap((s) => s.kernels) });
        })
        .catch(() => undefined);
    };

    const handle = (data: string, name: string) => {
      const event = parseRunEvent(data, name);
      if (!event) return;
      dispatch(event);
      if (event.type === "snapshot") seedKernels(event.run);
    };

    const connect = () => {
      setConnection(attempt === 0 ? "connecting" : "reconnecting");
      const es = new EventSource(eventsUrl(runId));
      source = es;
      es.onopen = () => {
        attempt = 0;
        setConnection("live");
      };
      es.onmessage = (e: MessageEvent<string>) => {
        handle(e.data, "message");
      };
      for (const name of EVENT_TYPES) {
        es.addEventListener(name, (e: MessageEvent<string>) => {
          handle(e.data, name);
        });
      }
      es.onerror = () => {
        es.close();
        if (disposed) return;
        if (terminalRef.current) {
          setConnection("ended");
          return;
        }
        setConnection("reconnecting");
        apiJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`).catch((err: unknown) => {
          if (disposed || !(err instanceof ApiError) || err.status !== 404) return;
          clearTimeout(retry);
          setConnection("missing");
        });
        retry = setTimeout(connect, BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)]);
        attempt++;
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(retry);
      source?.close();
    };
  }, [runId]);

  return { state, dispatch, connection };
}
