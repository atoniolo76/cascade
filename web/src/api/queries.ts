import { useEffect, useRef } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiJson } from "./client";
import type {
  Comparison,
  GpuConfig,
  KernelStat,
  RunCreate,
  RunDetail,
  RunFile,
  RunSummary,
  SimFile,
  SimStepTimeline,
  TraceDetail,
  TraceSummary,
} from "./types";

const POLL_MS = 3000;
const enc = encodeURIComponent;

export interface Health {
  ok: boolean;
  auth: boolean;
  workspace: string | null;
}

export const keys = {
  health: ["health"] as const,
  traces: ["traces"] as const,
  trace: (id: string) => ["traces", id] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["runs", id] as const,
  comparison: (id: string) => ["runs", id, "comparison"] as const,
  kernels: (id: string) => ["runs", id, "kernels"] as const,
  stepTimelines: (id: string) => ["runs", id, "step-timelines"] as const,
  runFiles: (id: string) => ["runs", id, "files"] as const,
  simFiles: (id: string) => ["sims", id, "files"] as const,
  gpuConfigs: ["gpu-configs"] as const,
  simLog: (id: string) => ["sims", id, "log"] as const,
};

export function useHealth() {
  return useQuery({
    queryKey: keys.health,
    queryFn: () => apiJson<Health>("/api/health"),
    staleTime: Infinity,
  });
}

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

/** Sims post step timing as they finish, so an active run keeps asking. */
export function useStepTimelines(runId: string, { live }: { live: boolean }) {
  return useQuery({
    queryKey: keys.stepTimelines(runId),
    queryFn: () => apiJson<SimStepTimeline[]>(`/api/runs/${enc(runId)}/step-timelines`),
    refetchInterval: live ? 5000 : false,
  });
}

/** A sim uploads its files just before its final heartbeat, so the poll ends with one last fetch once it settles. */
function usePollUntilSettled<T>(queryKey: readonly unknown[], path: string, live: boolean) {
  const q = useQuery({ queryKey, queryFn: () => apiJson<T>(path), refetchInterval: live ? 5000 : false });
  const { refetch } = q;
  const wasLive = useRef(live);
  useEffect(() => {
    if (wasLive.current && !live) void refetch();
    wasLive.current = live;
  }, [live, refetch]);
  return q;
}

export function useRunFiles(runId: string, { live }: { live: boolean }) {
  return usePollUntilSettled<RunFile[]>(keys.runFiles(runId), `/api/runs/${enc(runId)}/files`, live);
}

export function useSimFiles(simId: string, { live }: { live: boolean }) {
  return usePollUntilSettled<SimFile[]>(keys.simFiles(simId), `/api/sims/${enc(simId)}/files`, live);
}

/** Everything the Compare timeline needs per run; active runs refresh on the comparison cadence. */
export function useRunsForTimeline(ids: readonly string[], active: (id: string) => boolean) {
  const runs = useQueries({
    queries: ids.map((id) => ({
      queryKey: keys.run(id),
      queryFn: () => apiJson<RunDetail>(`/api/runs/${enc(id)}`),
      refetchInterval: active(id) ? POLL_MS : false,
    })),
  });
  const kernels = useQueries({
    queries: ids.map((id) => ({
      queryKey: keys.kernels(id),
      queryFn: () => apiJson<KernelStat[]>(`/api/runs/${enc(id)}/kernels`),
      refetchInterval: active(id) ? POLL_MS : false,
    })),
  });
  const comparisons = useQueries({
    queries: ids.map((id) => ({
      queryKey: keys.comparison(id),
      queryFn: () => apiJson<Comparison>(`/api/runs/${enc(id)}/comparison`),
      refetchInterval: active(id) ? 5000 : false,
    })),
  });
  return { runs, kernels, comparisons };
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
