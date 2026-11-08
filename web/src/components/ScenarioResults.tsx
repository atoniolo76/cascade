import { useState } from "react";
import { Link } from "react-router";
import type { ScenarioCreate, ScenarioRequestMetrics, ScenarioResult } from "../api/types";
import { paths } from "../paths";
import { ExtrapolationBanner } from "./ExtrapolationBanner";
import { Button, Card, PanelHeader } from "./ui";

const number = (n: number | null) => n === null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 3 });

export function ScenarioResults({ result, submitted }: { result: ScenarioResult; submitted: ScenarioCreate | undefined }) {
  const a = result.aggregate;
  const cards: [string, number | null, string][] = [
    ["p50 TTFT", a.p50_ttft_ms, "ms"], ["p99 TTFT", a.p99_ttft_ms, "ms"],
    ["p50 ITL", a.p50_itl_ms, "ms"], ["p50 E2E", a.p50_e2e_ms, "ms"],
    ["p99 E2E", a.p99_e2e_ms, "ms"], ["Output throughput", a.throughput_output_tokens_s, "tok/s"],
  ];
  return <section className="mt-7 space-y-5" aria-label="Scenario results">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 className="text-[18px] font-medium text-fg">Results {submitted && <span className="ml-2 text-[13px] font-normal text-muted">{submitted.num_requests} requests · {submitted.prompt_tokens}/{submitted.output_tokens} tokens · {submitted.arrival_rate_rps} req/s</span>}</h2>
      <Link to={paths.sim(result.run_id, result.sim_id)} className="text-[13px] text-link underline">{result.gpu_config} · {result.cached ? "cached base" : "base simulation"}</Link>
    </div>
    <ExtrapolationBanner counts={result.provenance_counts} />
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {cards.map(([label, value, unit]) => <Card key={label} className="p-4"><div className="text-[12px] text-muted">{label}</div><div className="mt-1 text-[24px] font-medium tracking-tight text-fg tabular-nums">{number(value)} <span className="text-[12px] font-normal text-muted">{unit}</span></div></Card>)}
    </div>
    <Card>
      <PanelHeader title="Provenance & assumptions" subtitle="Extrapolation is an estimate, not an accuracy claim" />
      <div className="space-y-3 p-5 text-[13px] text-fg-2">
        <div className="flex flex-wrap gap-2">{Object.entries(result.provenance_counts).map(([label, count]) => <span key={label} className="rounded-lg bg-surface-2 px-3 py-1 font-mono">{label}: {number(count)}</span>)}</div>
        <p>Calibration: prefill {result.calibration_range.prefill_tokens.join("–")} tokens; decode context {result.calibration_range.decode_context.join("–")} tokens. “Measured” means a matching simulated shape, not independent hardware measurement. Host timing is assumed.</p>
        <p>Resolved scheduler: {result.scheduler.max_num_seqs} sequences · {result.scheduler.max_num_batched_tokens} token budget · {result.scheduler.block_size} tokens/block · {number(result.scheduler.num_kv_blocks)} KV blocks · {result.scheduler.gpu_memory_utilization} memory utilization.</p>
        <details><summary className="cursor-pointer font-medium text-fg">All assumptions ({result.assumptions.length})</summary><ul className="mt-3 list-disc space-y-2 pl-5">{result.assumptions.map((s, i) => <li key={i}>{s}</li>)}</ul></details>
      </div>
    </Card>
    <StepChart result={result} />
    <RequestTable rows={result.requests} />
  </section>;
}

function StepChart({ result }: { result: ScenarioResult }) {
  const source = result.steps;
  const steps = source.length <= 500 ? source : Array.from({ length: 500 }, (_, i) => source[Math.floor(i * (source.length - 1) / 499)]).filter((s) => s !== undefined);
  const start = steps[0]?.t_start_s ?? 0;
  const end = steps.at(-1)?.t_end_s ?? 1;
  const series = [
    { title: "Batch (sequences)", color: "#5689e8", values: steps.map((s) => s.num_seqs) },
    { title: "Scheduled tokens", color: "#22a294", values: steps.map((s) => s.num_tokens) },
    { title: "Step wall (ms)", color: "#a47be0", values: steps.map((s) => s.wall_s * 1000) },
  ];
  return <Card>
    <PanelHeader title="Scheduler steps over time" subtitle={`${result.steps_total.toLocaleString()} executed steps · ${steps.length.toLocaleString()} plotted${result.steps_downsampled ? ` · API downsampled (${result.steps_sampling})` : ""}. Independent vertical scales; all metrics use all steps.`} />
    <div className="overflow-x-auto p-4">
      <svg viewBox="0 0 900 310" className="min-w-[600px] w-full" role="img" aria-label="Batch sequences, scheduled tokens and step wall duration over time">
        <title>Scheduler step chart with independent vertical scales</title>
        {series.map((line, row) => {
          const max = Math.max(1, ...line.values);
          const top = row * 88 + 24;
          const points = steps.map((step, i) => `${180 + (step.t_start_s - start) / Math.max(1e-9, end - start) * 690},${top + 55 - (line.values[i] ?? 0) / max * 50}`).join(" ");
          return <g key={line.title}>
            <text x="8" y={top + 15} fill="currentColor" className="text-fg" fontSize="13">{line.title}</text>
            <text x="8" y={top + 35} fill="currentColor" className="text-muted" fontSize="12">0–{number(max)}</text>
            <line x1="180" x2="870" y1={top + 55} y2={top + 55} stroke="currentColor" opacity=".15" />
            <polyline points={points} fill="none" stroke={line.color} strokeWidth="2" />
            {steps.length === 1 && <circle cx="180" cy={top + 55 - (line.values[0] ?? 0) / max * 50} r="3" fill={line.color} />}
          </g>;
        })}
        <text x="180" y="303" fill="currentColor" className="text-muted" fontSize="12">{number(start)} s</text>
        <text x="870" y="303" textAnchor="end" fill="currentColor" className="text-muted" fontSize="12">{number(end)} s since scenario start</text>
      </svg>
    </div>
  </Card>;
}

const COLUMNS: [keyof ScenarioRequestMetrics, string][] = [["id", "Request"], ["arrival_s", "Arrival (s)"], ["ttft_ms", "TTFT (ms)"], ["itl_ms_mean", "Mean ITL (ms)"], ["e2e_ms", "E2E (ms)"], ["queue_time_ms", "Queue (ms)"], ["output_tokens", "Output tokens"], ["preemptions", "Preemptions"]];

function RequestTable({ rows }: { rows: ScenarioRequestMetrics[] }) {
  const [sort, setSort] = useState<{ key: keyof ScenarioRequestMetrics; ascending: boolean }>({ key: "arrival_s", ascending: true });
  const [page, setPage] = useState(0);
  const sorted = [...rows].sort((a, b) => {
    const left = a[sort.key], right = b[sort.key];
    if (left === null || right === null) return left === right ? 0 : left === null ? 1 : -1;
    return (typeof left === "string" && typeof right === "string" ? left.localeCompare(right) : Number(left) - Number(right)) * (sort.ascending ? 1 : -1);
  });
  const pages = Math.max(1, Math.ceil(rows.length / 20));
  return <Card>
    <PanelHeader title="Per-request metrics" subtitle="Latency measured from arrival; null TTFT/ITL shown as —" />
    <div className="overflow-x-auto">
      <table className="w-full whitespace-nowrap text-left text-[12px] tabular-nums"><thead className="bg-surface-2 text-muted"><tr>
        {COLUMNS.map(([key, label]) => <th key={key} className="px-4 py-3 font-medium" aria-sort={sort.key === key ? sort.ascending ? "ascending" : "descending" : "none"}><button type="button" onClick={() => { setSort({ key, ascending: sort.key === key ? !sort.ascending : true }); setPage(0); }}>{label}{sort.key === key ? sort.ascending ? " ↑" : " ↓" : " ↕"}</button></th>)}
      </tr></thead><tbody>{sorted.slice(page * 20, (page + 1) * 20).map((r) => <tr key={r.id} className="border-t border-border text-fg-2">{COLUMNS.map(([key]) => <td key={key} className="px-4 py-3">{typeof r[key] === "string" ? r[key] : number(r[key])}</td>)}</tr>)}</tbody></table>
    </div>
    <div className="flex items-center justify-between gap-3 border-t border-border p-3 text-[12px] text-muted"><span>{rows.length} requests · Page {page + 1} of {pages}</span><div className="flex gap-2"><Button disabled={page === 0} onClick={() => { setPage(page - 1); }}>Previous</Button><Button disabled={page + 1 >= pages} onClick={() => { setPage(page + 1); }}>Next</Button></div></div>
  </Card>;
}
