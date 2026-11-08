const enc = encodeURIComponent;

export const paths = {
  traces: "/",
  trace: (id: string) => `/traces/${enc(id)}`,
  runs: "/runs",
  runsForTrace: (traceId: string) => `/runs?trace=${enc(traceId)}`,
  run: (id: string) => `/runs/${enc(id)}`,
  sim: (runId: string, simId: string) => `/runs/${enc(runId)}?sim=${enc(simId)}`,
  compare: (runIds: readonly string[] = []) => (runIds.length === 0 ? "/compare" : `/compare?runs=${runIds.map(enc).join(",")}`),
  newRun: (traceId?: string) => (traceId === undefined ? "/runs/new" : `/runs/new?trace=${enc(traceId)}`),
};
