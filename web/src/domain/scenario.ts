import type { ScenarioCreate } from "../api/types.ts";

export const REQUEST_GPUS = ["SM90_H100", "SM90_H200", "SM80_A100"] as const;
export const supportsRequests = (gpu: string) => REQUEST_GPUS.some((name) => name === gpu);
export const DEFAULT_SCENARIO = {
  prompt_tokens: "512", output_tokens: "128", num_requests: "8", arrival_rate_rps: "0",
  max_num_seqs: "256", max_num_batched_tokens: "8192", block_size: "16",
  enable_chunked_prefill: true, num_kv_blocks: "",
};
export type ScenarioForm = typeof DEFAULT_SCENARIO;

export function buildScenario(form: ScenarioForm, trace: string, gpu: string, modelLimit?: number): { body: ScenarioCreate | null; errors: string[] } {
  const errors: string[] = [];
  const integer = (key: Exclude<keyof ScenarioForm, "enable_chunked_prefill" | "arrival_rate_rps">, min: number, max: number) => {
    const value = Number(form[key]);
    if (!form[key].trim() || !Number.isSafeInteger(value) || value < min || value > max) errors.push(`${key} must be an integer from ${min} to ${max}.`);
    return value;
  };
  const prompt_tokens = integer("prompt_tokens", 1, 1_000_000);
  const output_tokens = integer("output_tokens", 0, 1_000_000);
  const num_requests = integer("num_requests", 1, 4096);
  const arrival_rate_rps = Number(form.arrival_rate_rps);
  if (!form.arrival_rate_rps.trim() || !Number.isFinite(arrival_rate_rps) || arrival_rate_rps < 0 || arrival_rate_rps > 1e9 || (arrival_rate_rps > 0 && (num_requests - 1) / arrival_rate_rps > 1e9)) errors.push("Arrival rate must be finite, between 0 and 1e9, with final arrival at most 1e9 seconds.");
  const scheduler: ScenarioCreate["scheduler"] = {
    max_num_seqs: integer("max_num_seqs", 1, 4096),
    max_num_batched_tokens: integer("max_num_batched_tokens", 1, 1_000_000),
    block_size: integer("block_size", 1, 1_000_000),
    enable_chunked_prefill: form.enable_chunked_prefill,
  };
  if (form.num_kv_blocks.trim()) scheduler.num_kv_blocks = integer("num_kv_blocks", 1, 1_000_000);
  if (!trace) errors.push("Select a ready trace.");
  if (!supportsRequests(gpu)) errors.push("Select a supported, architecture-compatible GPU config.");
  if (num_requests * (prompt_tokens + output_tokens) > 1_000_000) errors.push("Scenario exceeds 1,000,000 total prompt/output tokens.");
  if (modelLimit !== undefined && prompt_tokens + output_tokens > modelLimit) errors.push(`Prompt plus output must fit max_model_len=${modelLimit}.`);
  if (!scheduler.enable_chunked_prefill && prompt_tokens > scheduler.max_num_batched_tokens) errors.push("Prompt exceeds the token budget with chunked prefill disabled.");
  return { errors, body: errors.length ? null : { trace_id: trace, gpu_config: gpu, prompt_tokens, output_tokens, num_requests, arrival_rate_rps, scheduler } };
}

export const EXTRAPOLATION_BANNER = "Extrapolated beyond the profiled shape; not a measured result";
