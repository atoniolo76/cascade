import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { kernelLabel } from "./kernelLabel.ts";

describe("kernelLabel", () => {
  it("keeps the nested names of a mangled kernel", () => {
    assert.equal(kernelLabel("_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii"), "vllm::rms_norm_kernel");
    assert.equal(kernelLabel("_ZN2at6native29vectorized_elementwise_kernelILi4E"), "at::native::vectorized_elementwise_kernel");
  });

  it("passes plain names through", () => {
    assert.equal(kernelLabel("sm90_xmma_gemm_f16f16"), "sm90_xmma_gemm_f16f16");
  });
});
