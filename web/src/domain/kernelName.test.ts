import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { kernelDisplayName, simplifyKernelName } from "./kernelName.ts";

const cases: [string, string][] = [
  ["_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii", "vllm::rms_norm_kernel"],
  ["_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS4_iiiiii", "vllm::rotary_embedding_kernel"],
  ["_ZN4vllm30reshape_and_cache_flash_kernelIN3c104HalfES2_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S7_PT0_S9_PKlliiiiiPKfSD_", "vllm::reshape_and_cache_flash_kernel"],
  [
    "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64ELi64ELi128ELi4ELb0ELb0EN7cutlass6half_tELb0EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv13Flash_fwd_params",
    "flash::flash_fwd_splitkv_kernel",
  ],
  ["_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEESt5arrayIPcLm3EEEEviT0_T1_", "at::native::vectorized_elementwise_kernel"],
  [
    "_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE8_clEvEUlN3c104HalfEE_St5arrayIPcLm2EELi4E23TrivialOffsetCalculatorILi1EjESC_NS0_6memory12LoadWithCastILi1EEENSD_13StoreWithCastILi1EEEEEviT_T0_T2_T3_T4_T5_",
    "at::native::unrolled_elementwise_kernel",
  ],
  ["_ZN7cutlass7Kernel2I64cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x4_tn_align8EEvNT_6ParamsE", "cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x4_tn_align8>"],
  ["_Z10add_kernelPfS_S_i", "add_kernel"],
  ["_Z6reduceILi256EEvPKfPfi", "reduce"],
  ["_ZN12_GLOBAL__N_116softmax_warp_fwdEPKfPfi", "softmax_warp_fwd"],
  ["_ZNSt6vectorIiSaIiEE9push_backEOi", "std::vector::push_back"],
  ["_ZN3foo3BarC2Ev", "foo::Bar::Bar"],
];

describe("simplifyKernelName", () => {
  for (const [mangled, expected] of cases) {
    it(`reads ${expected}`, () => {
      assert.equal(kernelDisplayName(mangled), expected);
    });
  }

  it("splits the namespaces from the function and keeps the recorded name", () => {
    const name = "_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii";
    assert.deepEqual(simplifyKernelName(name), { scope: "vllm::", base: "rms_norm_kernel", full: name });
  });

  it("keeps unmangled names whole", () => {
    const gemm = "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas";
    assert.deepEqual(simplifyKernelName(gemm), { scope: "", base: gemm, full: gemm });
    assert.equal(kernelDisplayName("triton_poi_fused_mul_silu_0"), "triton_poi_fused_mul_silu_0");
  });

  it("drops the return type, template arguments, and parameters from demangled text", () => {
    assert.equal(
      kernelDisplayName("void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float const*, int)"),
      "at::native::cunn_SoftMaxForward",
    );
  });

  it("keeps what it read before grammar it does not know", () => {
    assert.equal(kernelDisplayName("_ZN2ns6kernelIXadL_Z1fvEEEEvv"), "ns::kernel");
  });

  it("falls back to the recorded name when nothing parses", () => {
    assert.equal(kernelDisplayName("_Zgarbage"), "_Zgarbage");
    assert.equal(kernelDisplayName("_ZN99tooshortE"), "_ZN99tooshortE");
  });
});
