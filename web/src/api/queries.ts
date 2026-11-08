import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiJson } from "./client";
import type {
  Comparison,
  GpuConfig,
  RunCreate,
  RunDetail,
  RunResults,
  RunSummary,
  TraceDetail,
  TraceSummary,
} from "./types";

const POLL_MS = 3000;
const enc = encodeURIComponent;

export const keys = {
  traces: ["traces"] as const,
  trace: (id: string) => ["traces", id] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["runs", id] as const,
  comparison: (id: string) => ["runs", id, "comparison"] as const,
  results: (id: string) => ["runs", id, "results"] as const,
  gpuConfigs: ["gpu-configs"] as const,
  simLog: (id: string) => ["sims", id, "log"] as const,
};

export function useTraces() {
  return useQuery({
    queryKey: keys.traces,
    queryFn: () => apiJson<TraceSummary[]>("/api/traces"),
    refetchInterval: POLL_MS,
  });
}

export function useTrace(id: string) {
  return useQuery({
    queryKey: keys.trace(id),
    queryFn: () => apiJson<TraceDetail>(`/api/traces/${enc(id)}`),
    refetchInterval: (q) => {
      const state = q.state.data?.state;
      return state === "ready" || state === "failed" ? false : POLL_MS;
    },
  });
}

export function useRuns() {
  return useQuery({
    queryKey: keys.runs,
    queryFn: () => apiJson<RunSummary[]>("/api/runs"),
    refetchInterval: POLL_MS,
  });
}

export function useGpuConfigs() {
  return useQuery({
    queryKey: keys.gpuConfigs,
    queryFn: () => apiJson<GpuConfig[]>("/api/gpu-configs"),
    staleTime: Infinity,
  });
}

export function useComparison(runId: string, { live }: { live: boolean }) {
  return useQuery({
    queryKey: keys.comparison(runId),
    queryFn: () => apiJson<Comparison>(`/api/runs/${enc(runId)}/comparison`),
    refetchInterval: live ? 5000 : false,
  });
}

export function fetchRunResults(runId: string) {
  return apiJson<RunResults>(`/api/runs/${enc(runId)}/results`);
}

export function useCreateRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunCreate) =>
      apiJson<RunDetail>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: (run) => {
      qc.setQueryData(keys.run(run.id), run);
      void qc.invalidateQueries({ queryKey: keys.runs });
    },
  });
}

export function useCancelRun(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiJson<RunDetail>(`/api/runs/${enc(runId)}/cancel`, { method: "POST" }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.runs });
    },
  });
}
