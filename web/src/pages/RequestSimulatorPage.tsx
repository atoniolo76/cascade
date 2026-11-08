import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { ApiError } from "../api/client";
import { useGpuConfigs, useScenario, useScenarioBaseRuns, useTrace, useTraces } from "../api/queries";
import { PageHeader } from "../components/AppShell";
import { ScenarioResults } from "../components/ScenarioResults";
import { Button, Card, ErrorNote, PanelHeader } from "../components/ui";
import { buildScenario, DEFAULT_SCENARIO, supportsRequests, type ScenarioForm } from "../domain/scenario";
import { paths } from "../paths";

const INPUT = "mt-1 block w-full rounded-[10px] border border-border bg-surface px-3 py-2 text-[14px] text-fg outline-none focus:ring-2 focus:ring-accent";

export function RequestSimulatorPage() {
  const [params] = useSearchParams();
  const trace = params.get("trace") ?? "";
  const gpu = params.get("gpu") ?? "";
  return <RequestSimulator key={`${trace}|${gpu}`} initialTrace={trace} initialGpu={gpu} />;
}

function RequestSimulator({ initialTrace, initialGpu }: { initialTrace: string; initialGpu: string }) {
  const traces = useTraces();
  const configs = useGpuConfigs();
  const ready = traces.data?.filter((t) => t.state === "ready") ?? [];
  const [selection, setSelection] = useState(initialTrace);
  const traceId = selection || ready[0]?.id || "";
  const trace = ready.find((t) => t.id === traceId);
  const detail = useTrace(traceId);
  const [gpuSelection, setGpuSelection] = useState(initialGpu);
  const compatible = configs.data?.filter((g) => supportsRequests(g.name) && g.arch === trace?.trace_gpu_arch) ?? [];
  const gpu = gpuSelection || compatible.find((g) => g.name === trace?.suggested_sim_config)?.name || compatible[0]?.name || "";
  const supported = compatible.some((g) => g.name === gpu);
  const [form, setForm] = useState<ScenarioForm>({ ...DEFAULT_SCENARIO });
  const [attempted, setAttempted] = useState(false);
  const scenario = useScenario();
  const active = useScenarioBaseRuns(scenario.variables, scenario.isPending);
  const modelLimit = readModelLimit(detail.data?.metadata);
  const validation = buildScenario(form, traceId, gpu, modelLimit);
  const update = (key: keyof ScenarioForm, value: string | boolean) => { setForm((f) => ({ ...f, [key]: value })); };
  const numeric = (key: Exclude<keyof ScenarioForm, "enable_chunked_prefill">, label: string, hint?: string) => (
    <label className="block text-[13px] text-fg-2">
      {label}
      <input name={key} type="number" step={key === "arrival_rate_rps" ? "any" : "1"} value={form[key]} onChange={(e) => { update(key, e.target.value); }} className={INPUT} />
      {hint && <span className="mt-1 block text-[12px] text-muted">{hint}</span>}
    </label>
  );
  return <>
    <PageHeader crumbs={[{ label: "Request simulator" }]}>
      <p className="max-w-3xl text-[14px] leading-6 text-muted">Explore prompt sizes, output lengths and continuous batching on a single GPU. Predictions reuse a recorded base simulation; this is not model inference or measured concurrency.</p>
    </PageHeader>
    <Card>
      <PanelHeader title="Request simulator" subtitle="Transient scenario · results are not saved" />
      <form noValidate className="space-y-5 p-5" onSubmit={(e) => {
        e.preventDefault(); setAttempted(true);
        if (validation.body && trace && supported) scenario.mutate(validation.body);
      }}>
        <fieldset disabled={scenario.isPending} className="space-y-5 disabled:opacity-60">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-[13px] text-fg-2">Trace
              <select value={traceId} className={INPUT} onChange={(e) => { setSelection(e.target.value); setGpuSelection(""); scenario.reset(); }}>
                {!trace && <option value={traceId}>{traceId || "Select a ready trace"}</option>}
                {ready.map((t) => <option key={t.id} value={t.id}>{t.model ?? t.id} · {t.source_gpu ?? t.trace_gpu_arch}{t.sample ? " · sample" : ""}</option>)}
              </select>
            </label>
            <label className="text-[13px] text-fg-2">GPU config
              <select value={gpu} className={INPUT} onChange={(e) => { setGpuSelection(e.target.value); scenario.reset(); }}>
                {!supported && <option value={gpu}>{gpu ? `${gpu} — unsupported or incompatible` : "No compatible request-sizing config"}</option>}
                {compatible.map((g) => <option key={g.name}>{g.name}</option>)}
              </select>
            </label>
          </div>
          {!supported && <p role="status" className="text-[14px] text-muted">Request sizing supports SM90_H100, SM90_H200 and SM80_A100 only, with a matching trace architecture. Choose another trace/config.</p>}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {numeric("prompt_tokens", "Prompt tokens")}
            {numeric("output_tokens", "Output tokens", "0 = prefill only")}
            {numeric("num_requests", "Number of requests", "Up to 4096")}
            {numeric("arrival_rate_rps", "Arrival rate (req/s)", "0 = all at once; otherwise uniform spacing")}
          </div>
          <details className="rounded-[12px] border border-border px-4 py-3">
            <summary className="cursor-pointer text-[14px] font-medium text-fg">Advanced scheduler assumptions</summary>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {numeric("max_num_seqs", "Max sequences")}
              {numeric("max_num_batched_tokens", "Max batched tokens")}
              {numeric("block_size", "KV block size (tokens)")}
              {numeric("num_kv_blocks", "KV blocks override", "Leave blank to estimate from GPU memory")}
              <label className="flex items-center gap-2 text-[14px] text-fg-2"><input type="checkbox" checked={form.enable_chunked_prefill} onChange={(e) => { update("enable_chunked_prefill", e.target.checked); }} />Chunked prefill</label>
            </div>
            <p className="mt-3 text-[12px] text-muted">Defaults are assumptions, not captured vLLM settings. Memory utilization comes from the profile (otherwise 0.8). Capacity subtracts weights and a fixed 1 GiB activation reserve.</p>
          </details>
        </fieldset>
        <p className="text-[12px] text-muted">Limits: 1,000,000 total prompt/output tokens; 100,000 executed steps. {modelLimit ? `Model limit: ${modelLimit} tokens.` : "Model-length limits are checked by the server (2048 if not captured)."} Only the first missing base simulation may take minutes; closing this page does not cancel it.</p>
        {traces.isError && <ErrorNote error={traces.error} />}
        {configs.isError && <ErrorNote error={configs.error} />}
        {detail.isError && <ErrorNote error={detail.error} />}
        {attempted && validation.errors.length > 0 && <ul role="alert" className="list-inside list-disc text-[13px] text-red-500">{validation.errors.map((error) => <li key={error}>{error}</li>)}</ul>}
        {scenario.isError && <div role="alert"><ErrorNote error={scenario.error} />{scenario.error instanceof ApiError && scenario.error.runId && <Link className="text-link underline" to={paths.run(scenario.error.runId)}>Open base run (not cancelled)</Link>}</div>}
        <Button type="submit" variant="primary" disabled={scenario.isPending || !trace || !supported}>{scenario.isPending ? "Simulating…" : "Simulate requests"}</Button>
      </form>
    </Card>
    {scenario.isPending && <div role="status" aria-live="polite" className="mt-4 rounded-[14px] border border-border bg-accent-soft p-4 text-[14px] text-fg">
      {active[0] ? <>Matching base sim is {active[0].state}. <Link className="text-link underline" to={paths.sim(active[0].run_id, active[0].id)}>Open base run</Link>.</> : <>Resolving cached timing or launching a base sim. <Link className="text-link underline" to={paths.runsForTrace(scenario.variables.trace_id)}>Watch runs for this trace</Link>.</>}
      <p className="mt-1 text-[12px] text-muted">This can wait up to the server’s CASCADE_OPENAI_WAIT_S budget. The request-layer calculation runs after base timing is available.</p>
    </div>}
    {scenario.data && <ScenarioResults key={`${scenario.data.sim_id}|${scenario.submittedAt}`} result={scenario.data} submitted={scenario.variables} />}
  </>;
}

function readModelLimit(metadata: Record<string, unknown> | null | undefined): number | undefined {
  const sources = [metadata, metadata?.run, metadata?.workload, metadata?.llm_kwargs, metadata?.effective_llm_kwargs];
  for (const source of sources) {
    if (typeof source === "object" && source !== null && "max_model_len" in source && typeof source.max_model_len === "number" && Number.isSafeInteger(source.max_model_len) && source.max_model_len > 0) return source.max_model_len;
  }
  return undefined;
}
