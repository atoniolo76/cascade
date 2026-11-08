# Measured sweep tables

Cycles and microseconds are simulated, not hardware measurements.

## SM90_H100

Accel-Sim wall seconds: 804.428951
Counts: `{'aligned': 712, 'selected': 712, 'ok': 712, 'failed': 0, 'timeout': 0, 'propagated': 16376, 'unique_accelsim_runs': 31, 'traced_kernel_occurrences': 712, 'unmatched_untraced': 0}`

### Prefill/local step 0

| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |
|---|---:|---:|---:|---:|
| KV write | 12 | 8229.00 | 4.156061 | 49.872727 |
| MLP down GEMM | 12 | 10273.00 | 5.188384 | 62.260606 |
| MLP down reduction | 12 | 7269.00 | 3.671212 | 44.054545 |
| MLP up GEMM | 12 | 10821.00 | 5.465152 | 65.581818 |
| QKV GEMM | 12 | 9786.00 | 4.942424 | 59.309091 |
| attention | 12 | 14501.00 | 7.323737 | 87.884848 |
| layer norm | 25 | 10106.00 | 5.104040 | 127.601010 |
| lm_head | 1 | 52002.00 | 26.263636 | 26.263636 |
| other setup/index/reduction | 5 | 13534.60 | 6.835657 | 34.178283 |
| out-proj | 12 | 8832.00 | 4.460606 | 53.527273 |
| small elementwise | 63 | 6328.49 | 3.196208 | 201.361111 |

### Decode/local step 2 (context 8)

| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |
|---|---:|---:|---:|---:|
| KV write | 12 | 8022.00 | 4.051515 | 48.618182 |
| MLP down GEMM | 12 | 10273.00 | 5.188384 | 62.260606 |
| MLP down reduction | 12 | 7269.00 | 3.671212 | 44.054545 |
| MLP up GEMM | 12 | 10821.00 | 5.465152 | 65.581818 |
| QKV GEMM | 12 | 9786.00 | 4.942424 | 59.309091 |
| attention | 12 | 13286.00 | 6.710101 | 80.521212 |
| layer norm | 25 | 10103.00 | 5.102525 | 127.563131 |
| lm_head | 1 | 52002.00 | 26.263636 | 26.263636 |
| other setup/index/reduction | 5 | 13534.60 | 6.835657 | 34.178283 |
| out-proj | 12 | 8832.00 | 4.460606 | 53.527273 |
| small elementwise | 63 | 6420.97 | 3.242913 | 204.303535 |

| Step kind | N | GPU busy µs mean | p50 | min | max |
|---|---:|---:|---:|---:|---:|
| prefill | 3 | 822.678949 | 822.678949 | 822.646949 | 822.710949 |
| decode | 93 | 816.980001 | 816.933313 | 816.805313 | 819.269313 |

| Repeat | E2E ms | TTFT ms | ITL mean ms |
|---|---:|---:|---:|
| 1 | 261.133918 | 8.986973 | 8.140983 |
| 2 | 262.302298 | 8.710049 | 8.186943 |
| 3 | 262.987898 | 9.301620 | 8.190507 |

## SM90_H200

Accel-Sim wall seconds: 777.060434
Counts: `{'aligned': 712, 'selected': 712, 'ok': 712, 'failed': 0, 'timeout': 0, 'propagated': 16376, 'unique_accelsim_runs': 31, 'traced_kernel_occurrences': 712, 'unmatched_untraced': 0}`

### Prefill/local step 0

| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |
|---|---:|---:|---:|---:|
| KV write | 12 | 8227.00 | 4.155051 | 49.860606 |
| MLP down GEMM | 12 | 10282.00 | 5.192929 | 62.315152 |
| MLP down reduction | 12 | 7269.00 | 3.671212 | 44.054545 |
| MLP up GEMM | 12 | 10684.00 | 5.395960 | 64.751515 |
| QKV GEMM | 12 | 9771.00 | 4.934848 | 59.218182 |
| attention | 12 | 14503.00 | 7.324747 | 87.896970 |
| layer norm | 25 | 10090.00 | 5.095960 | 127.398990 |
| lm_head | 1 | 55296.00 | 27.927273 | 27.927273 |
| other setup/index/reduction | 5 | 13486.80 | 6.811515 | 34.057576 |
| out-proj | 12 | 8831.00 | 4.460101 | 53.521212 |
| small elementwise | 63 | 6328.13 | 3.196024 | 201.349495 |

### Decode/local step 2 (context 8)

| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |
|---|---:|---:|---:|---:|
| KV write | 12 | 8022.00 | 4.051515 | 48.618182 |
| MLP down GEMM | 12 | 10282.00 | 5.192929 | 62.315152 |
| MLP down reduction | 12 | 7269.00 | 3.671212 | 44.054545 |
| MLP up GEMM | 12 | 10684.00 | 5.395960 | 64.751515 |
| QKV GEMM | 12 | 9771.00 | 4.934848 | 59.218182 |
| attention | 12 | 13286.00 | 6.710101 | 80.521212 |
| layer norm | 25 | 10103.00 | 5.102525 | 127.563131 |
| lm_head | 1 | 55296.00 | 27.927273 | 27.927273 |
| other setup/index/reduction | 5 | 13486.80 | 6.811515 | 34.057576 |
| out-proj | 12 | 8831.00 | 4.460101 | 53.521212 |
| small elementwise | 63 | 6420.02 | 3.242432 | 204.273232 |

| Step kind | N | GPU busy µs mean | p50 | min | max |
|---|---:|---:|---:|---:|---:|
| prefill | 3 | 823.135515 | 823.135515 | 823.103515 | 823.167515 |
| decode | 93 | 817.619900 | 817.573212 | 817.445212 | 819.909212 |

| Repeat | E2E ms | TTFT ms | ITL mean ms |
|---|---:|---:|---:|
| 1 | 261.154211 | 8.987430 | 8.141623 |
| 2 | 262.322591 | 8.710506 | 8.187583 |
| 3 | 263.008191 | 9.302077 | 8.191146 |
