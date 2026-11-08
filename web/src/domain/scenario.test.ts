import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ExtrapolationBanner } from "../components/ExtrapolationBanner.ts";
import { buildScenario, DEFAULT_SCENARIO, EXTRAPOLATION_BANNER } from "./scenario.ts";

describe("request simulator shorthand", () => {
  it("builds the API payload with numeric values and omits automatic KV capacity", () => {
    const { body, errors } = buildScenario(DEFAULT_SCENARIO, "trace", "SM90_H100", 2048);
    assert.deepEqual(errors, []);
    assert.deepEqual(body, {
      trace_id: "trace", gpu_config: "SM90_H100", num_requests: 8, prompt_tokens: 512,
      output_tokens: 128, arrival_rate_rps: 0,
      scheduler: { max_num_seqs: 256, max_num_batched_tokens: 8192, block_size: 16, enable_chunked_prefill: true },
    });
  });
  it("includes explicit capacity and uniformly spaced arrivals, including zero outputs", () => {
    const result = buildScenario({ ...DEFAULT_SCENARIO, num_kv_blocks: "32", arrival_rate_rps: "2.5", output_tokens: "0" }, "trace", "SM80_A100");
    assert.ok(result.body);
    assert.equal(result.body.scheduler.num_kv_blocks, 32);
    assert.equal(result.body.arrival_rate_rps, 2.5);
    assert.equal(result.body.output_tokens, 0);
  });
  for (const change of [
    { prompt_tokens: "" }, { prompt_tokens: "0" }, { output_tokens: "-1" },
    { num_requests: "4097" }, { num_requests: "1.2" }, { arrival_rate_rps: "Infinity" },
    { arrival_rate_rps: "-1" }, { arrival_rate_rps: "1e-20" }, { num_kv_blocks: "0" },
    { block_size: "1.5" }, { max_num_seqs: "4097" }, { max_num_batched_tokens: "0" },
    { num_requests: "4096" }, { prompt_tokens: "2000", output_tokens: "128" },
    { enable_chunked_prefill: false, max_num_batched_tokens: "1" },
  ]) it(`rejects invalid form ${JSON.stringify(change)}`, () => {
    const result = buildScenario({ ...DEFAULT_SCENARIO, ...change }, "trace", "SM90_H100", 2048);
    assert.equal(result.body, null);
    assert.ok(result.errors.length > 0);
  });
  it("rejects absent trace or unsupported GPU", () => {
    assert.equal(buildScenario(DEFAULT_SCENARIO, "", "SM90_H100").body, null);
    assert.equal(buildScenario(DEFAULT_SCENARIO, "trace", "SM86_RTX3070").body, null);
  });
  it("leaves an uncaptured model limit to server validation", () => {
    assert.ok(buildScenario({ ...DEFAULT_SCENARIO, num_requests: "1", prompt_tokens: "10000" }, "trace", "SM90_H100").body);
  });
});

describe("extrapolation banner rendering", () => {
  it("renders a prominent alert for any extrapolated step", () => {
    const html = renderToStaticMarkup(createElement(ExtrapolationBanner, { counts: { "extrapolated:roofline": 1, "assumed-host": 2 } }));
    assert.ok(html.includes('role="alert"'));
    assert.ok(html.includes(EXTRAPOLATION_BANNER));
  });
  it("does not label measured/interpolated or empty results as extrapolated", () => {
    for (const counts of [{}, { measured: 2, interpolated: 3, "extrapolated:roofline": 0 }]) {
      assert.equal(renderToStaticMarkup(createElement(ExtrapolationBanner, { counts })), "");
    }
  });
});
