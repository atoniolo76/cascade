import pytest

from cascade.kernel_names import kernel_display_name

# Every expected value below came from running kernelDisplayName in web/src/domain/kernelName.ts.
KERNEL_NAME_TEST_TS = [
    ("_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii", "vllm::rms_norm_kernel"),
    (
        "_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS4_iiiiii",
        "vllm::rotary_embedding_kernel",
    ),
    (
        "_ZN4vllm30reshape_and_cache_flash_kernelIN3c104HalfES2_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S7_PT0_S9_PKlliiiiiPKfSD_",
        "vllm::reshape_and_cache_flash_kernel",
    ),
    (
        "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64ELi64ELi128ELi4ELb0ELb0EN7cutlass6half_tELb0EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv13Flash_fwd_params",
        "flash::flash_fwd_splitkv_kernel",
    ),
    (
        "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEESt5arrayIPcLm3EEEEviT0_T1_",
        "at::native::vectorized_elementwise_kernel",
    ),
    (
        "_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE8_clEvEUlN3c104HalfEE_St5arrayIPcLm2EELi4E23TrivialOffsetCalculatorILi1EjESC_NS0_6memory12LoadWithCastILi1EEENSD_13StoreWithCastILi1EEEEEviT_T0_T2_T3_T4_T5_",
        "at::native::unrolled_elementwise_kernel",
    ),
    (
        "_ZN7cutlass7Kernel2I64cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x4_tn_align8EEvNT_6ParamsE",
        "cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x4_tn_align8",
    ),
    ("_Z10add_kernelPfS_S_i", "add_kernel"),
    ("_Z6reduceILi256EEvPKfPfi", "reduce"),
    ("_ZN12_GLOBAL__N_116softmax_warp_fwdEPKfPfi", "softmax_warp_fwd"),
    ("_ZNSt6vectorIiSaIiEE9push_backEOi", "std::vector::push_back"),
    ("_ZN3foo3BarC2Ev", "foo::Bar::Bar"),
    (
        "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
        "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
    ),
    ("triton_poi_fused_mul_silu_0", "triton_poi_fused_mul_silu_0"),
    (
        "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, "
        "at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float const*, int)",
        "at::native::cunn_SoftMaxForward",
    ),
    ("_ZN2ns6kernelIXadL_Z1fvEEEEvv", "ns::kernel"),
    ("_Zgarbage", "_Zgarbage"),
    ("_ZN99tooshortE", "_ZN99tooshortE"),
]

OPT125M_H100_ACCELSIM = [
    ("nvjet_sm90_hsh_64x8_64x16_4x1_v_bz_bias_TNT", "nvjet_sm90_hsh_64x8_64x16_4x1_v_bz_bias_TNT"),
    (
        "_ZN4vllm30reshape_and_cache_flash_kernelIttLNS_18Fp8KVCacheDataTypeE0EEEvPKT_S4_PT0_S6_PKlllllliiiPKfSA_i",
        "vllm::reshape_and_cache_flash_kernel",
    ),
    (
        "_ZN7cutlass13device_kernelIN5flash20enable_sm90_or_laterINS1_16FlashAttnFwdSm90INS1_25CollectiveMainloopFwdSm90ILi2EN4cute5tupleIJNS5_1CILi1EEES8_S8_EEENS6_IJNS7_ILi64EEENS7_ILi192EEESA_EEELi64ENS_6half_tEfNS_4arch4Sm90ELb1ELb0ELb0ELb1ELb1ELb0ELb0ELb1ELb1ELb1ELb0ELb0ENS_10bfloat16_tELi1EEENS1_21CollectiveEpilogueFwdINS6_IJSA_SA_SB_EEES9_SD_SF_Li128ELb1ELb1ELb0ELb0ELi1EEENS1_36VarlenDynamicPersistentTileSchedulerILi64ELi192ELi128ELi128ELb0ELb1ELb1ELb1ELb0ELb1EEEEEEEEEvNT_6ParamsE",
        "cutlass::device_kernel",
    ),
    (
        "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEviT0_T1_",
        "at::native::vectorized_elementwise_kernel",
    ),
    (
        "_ZN7cutlass13device_kernelIN5flash20enable_sm90_or_laterINS1_16FlashAttnFwdSm90INS1_25CollectiveMainloopFwdSm90ILi2EN4cute5tupleIJNS5_1CILi1EEES8_S8_EEENS6_IJNS7_ILi64EEENS7_ILi192EEESA_EEELi64ENS_6half_tEfNS_4arch4Sm90ELb0ELb0ELb0ELb1ELb1ELb0ELb0ELb1ELb1ELb1ELb0ELb0ENS_10bfloat16_tELi1EEENS1_21CollectiveEpilogueFwdINS6_IJSA_SA_SB_EEES9_SD_SF_Li128ELb1ELb1ELb0ELb0ELi1EEENS1_36VarlenDynamicPersistentTileSchedulerILi64ELi192ELi128ELi128ELb0ELb1ELb1ELb0ELb0ELb1EEEEEEEEEvNT_6ParamsE",
        "cutlass::device_kernel",
    ),
]

# Where Python and JavaScript strings disagree: trim's whitespace set and UTF-16 code-unit lengths.
JS_STRING_SEMANTICS = [
    ("\x1c _Z3fooi", "\x1c _Z3fooi"),
    ("﻿_Z3fooi\n", "foo"),
    ("_Z3f\U0001f600ov", "f\U0001f600"),
    ("_ZNKS", "std"),
]


@pytest.mark.parametrize(
    ("name", "expected"),
    KERNEL_NAME_TEST_TS + OPT125M_H100_ACCELSIM + JS_STRING_SEMANTICS,
    ids=lambda v: repr(v[:40]),
)
def test_matches_kernel_name_ts(name, expected):
    assert kernel_display_name(name) == expected
