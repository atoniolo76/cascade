const enc = encodeURIComponent;

export const paths = {
  traces: "/",
  trace: (id: string) => `/traces/${enc(id)}`,
  runs: "/runs",
  run: (id: string) => `/runs/${enc(id)}`,
  newRun: (traceId?: string) => (traceId === undefined ? "/runs/new" : `/runs/new?trace=${enc(traceId)}`),
};
