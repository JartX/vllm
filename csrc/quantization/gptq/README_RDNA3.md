# RDNA3 W4A16 GPTQ Kernels (gfx1100)

Drop-in replacement for `ExllamaLinearKernel` / `TritonW4A16LinearKernel`
on AMD RDNA3 (RX 7900 XT/XTX, gfx1100/gfx1101/gfx1102). Provides:

* a hand-tuned scalar dot-product kernel for decode + small batches, and
* a `v_wmma_f32_16x16x16` matrix-instruction kernel for bf16 prefill.

Registered as `RDNA3W4A16LinearKernel` in
`vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py` ahead of
the Triton kernel for the ROCm path; falls through on non-RDNA3 ROCm
devices via `can_implement()` gating on `vllm.platforms.rocm.on_gfx11()`.

## Files

| File | Role |
| --- | --- |
| `q_gemm_rdna3.cu` | Scalar dot-product kernel + C++ dispatch entry `gptq_gemm_rdna3` |
| `q_gemm_rdna3_wmma.cu` | WMMA prefill kernel + 3 diagnostic ops |
| `qdq_4_rdna3.cuh` | int4 dequant helpers (fp16 bit-trick, bf16 fp32-output) |

## Dispatch (decision tree)

The `torch.ops._C.gptq_gemm_rdna3` op is the **single Python entry point**.
All branching happens in C++ to keep `apply_weights` torch.compile-friendly:

```
gptq_gemm_rdna3(a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format)
│
├── if dtype == bf16 and M >= 16 and N % 16 == 0 and K % 16 == 0
│       → gptq_gemm_rdna3_wmma  (v_wmma_f32_16x16x16_bf16_w32, prefill)
│
└── else
        → scalar dot-product kernel (decode + fp16 prefill + small batches)
```

**Why bf16-only WMMA?** Microbench across 5 Qwen-class shapes × M ∈ {1..256}
showed the fp16 scalar path (with the exllama bit-trick) beats the current
WMMA implementation at every M because fp16 dequant is essentially free
and the kernel becomes memory-bound at ~26% HBM peak. The WMMA kernel
underperforms because:

* 1 wave per block / 16×16 output tile → ~10% CU occupancy at typical
  prefill grid sizes
* No LDS-shared A across waves; each wave reloads from global
* No K-pipelining; `__syncthreads()` twice per K iteration

For bf16 the picture is different: `bf16` scalar pays a tax for the
missing `v_pk_fma_bf16` on gfx11 (the kernel uses fp32 widening to
sidestep, see below), but WMMA wins consistently at M ≥ 16 by 1.5-2×
because it avoids per-element FMAs entirely.

## Kernel architecture

### Compute paths overview

```
================================================================================
                    RDNA3 GPTQ W4A16 COMPUTE PATHS (gfx1100)
================================================================================

                       [ C++ ENTRY: gptq_gemm_rdna3 ]
                            (single Python entry)
                                      │
                ┌─────────────────────┴─────────────────────┐
                │  dtype == bf16                            │
                │  AND M >= 16                              │
                │  AND N % 16 == 0  AND  K % 16 == 0        │
                └────────┬─────────────────────────┬────────┘
                         │ FALSE                   │ TRUE
                         ▼                         ▼
================================================================================
        SCALAR PATH                       WMMA PATH
        gemm_q4_kernel_rdna3<T,M_COUNT>   gemm_q4_wmma_kernel<T>
        (decode + fp16 prefill            (bf16 prefill, M >= 16,
         + small M < 16)                   compute-bound win)
================================================================================

 BLOCK STRUCTURE                          BLOCK STRUCTURE
  - 256 threads (8 waves x 32 lanes)       - 32 threads = 1 wave-group
  - all 256 lanes active                   - lanes 0..15 unique inputs
                                           - lanes 16..31 hold COPIES
                                             (RDNA3 doubled wave32 conv.)

 TILE PER BLOCK                           TILE PER BLOCK
  M_COUNT rows x 1024 N x 256 K            16 M rows x 16 N x full-K
  ┌─────────────────────────────┐          ┌──────────┐
M │      1024 N columns         │ K=256  M │ 16 N     │ K = full
  │   (4 N cols per thread)     │ (1/16    │  cols    │ (no K-split,
  └─────────────────────────────┘  of K)   └──────────┘  one block
                                                          owns the
                                                          16x16 tile)

 GRID                                     GRID
  (N/1024, M/M_COUNT, K/256)               (N/16, M/16, 1)
   gridDim.z splits K -> atomic acc         NO K-split, NO atomics

 CORE EXECUTION  (per outer K=32 iter)    CORE EXECUTION  (per K=16 step)
  for k in 0..256 step 32:                  for k_tile in 0..K step 16:
   ┌─────────────────────────────┐           ┌─────────────────────────────┐
   │ uint4 weight load (4 ints)  │           │ Cooperative dequant:        │
   │ bit-trick dequant 32 weights│           │   32 lanes do 32 dequants   │
   │   fp16: 0x6400 + FMA        │           │   -> 16 N x 16 K B-tile in  │
   │   bf16: 0x4300 + fp32 widen │           │      LDS                    │
   │ 4 dot22_8 vs LDS A          │           │ build a_frag, b_frag        │
   │   (4 packed FMAs each)      │           │ v_wmma_f32_16x16x16         │
   └─────────────────────────────┘           │   (one hw matrix instr)     │
                                             └─────────────────────────────┘

 OUTPUT WRITE                             OUTPUT WRITE
  K/BLOCK_KN_SIZE blocks contend per        Single block owns the 16x16
  output position                           output tile
   atomicCAS-pk (uint32 CAS retry           direct store to global C
   loop -- gfx11 has no v_global_           (lanes 0..15 store EVEN rows
   atomic_pk_add_{f16,bf16})                 lanes 16..31 store ODD rows)


================================================================================
                            COMPUTE DENSITY COMPARISON
================================================================================
| Metric                       | Scalar                   | WMMA                |
|------------------------------|--------------------------|---------------------|
| Threads per block            | 256 (8 waves)            | 32 (1 wave)         |
| Lanes doing distinct work    | 256                      | 16 (rest duplicate) |
| Block muladds per K=16 step  | ~4096 (256 thr x 16 mad) | 4096 (1 v_wmma op)  |
| Cycles for those muladds     | ~32-64 (multi-issue)     | ~16                 |
| Per-wave peak throughput     | 64 packed-FMA muladds/c  | 256 muladds/cycle   |
| Atomics per output position  | K / BLOCK_KN_SIZE        | 0                   |
|                              | (16 for K=4096)          |                     |
| Memory pattern (per block)   | vec int4 weight loads    | coalesced weight    |
|                              | + LDS A + atomic CAS     | + LDS B broadcast   |
| Best regime                  | memory-bound, small M,   | compute-bound bf16  |
|                              | fp16 always              | prefill (M >= 16)   |
================================================================================

BEST FOR
  Scalar:                                  WMMA:
    - decode (M=1) — high concurrency        - bf16 prefill (M >= 16) — bypass
      across 256 threads x many blocks         the slow __hfma2(bf162_t,...)
    - fp16 always — bit-trick + free           on gfx11 (no v_pk_fma_bf16 in
      v_pk_fma_f16 keep it memory-bound        HW; the scalar path falls
      at ~26 % HBM peak; WMMA can't beat       back to a fp32 widen path)
      that on this kernel                    - 1.5-2x scalar at M=16..256
    - any small M < 16 — too little          - direct store to C, no atomic
      work to fill a WMMA tile                 CAS retries
                                             - fp16 prefill DOESN'T win here
                                               yet — see "Why WMMA helps
                                               bf16..." below
================================================================================
```

### RDNA3 wave32 fragment layout (lane × slot mapping)

This is the part that took the longest to nail down. AMD's wave32 WMMA
uses a "doubled" input convention — lanes 16..31 hold a copy of lanes
0..15. The output, in contrast, is split between halves.

```
A frag input (16 fp16/bf16 elements per lane):
═══════════════════════════════════════════════
  lane t holds:   a_frag[i] = A[lane_lo][k = i]
                  with lane_lo = t & 15

  Lane axis encodes M (A's row index)
  Slot axis encodes K (depth axis aligned with B)

       slot 0  slot 1  slot 2  ... slot 15
       ──────  ──────  ──────  ── ──────
  L 0  A[0][0] A[0][1] A[0][2] ... A[0][15]   ┐
  L 1  A[1][0] A[1][1] A[1][2] ... A[1][15]   │ lanes 0..15
  ...                                          │ hold rows 0..15
  L15  A[15][0] ......         ... A[15][15]  ┘
  L16  A[0][0] ........        ... A[0][15]   ┐
  L17  A[1][0]                     A[1][15]   │ lanes 16..31
  ...                                          │ DUPLICATE 0..15
  L31  A[15][0] .....          ... A[15][15]  ┘


B frag input (16 elements per lane, COL-major):
════════════════════════════════════════════════
  lane t holds:   b_frag[i] = B[k = i][lane_lo]
                  with lane_lo = t & 15

  Lane axis encodes N (B's column index)
  Slot axis encodes K (same as A — enables per-lane inner products)

       slot 0  slot 1  slot 2  ... slot 15
       ──────  ──────  ──────  ── ──────
  L 0  B[0][0] B[1][0] B[2][0] ... B[15][0]   ┐
  L 1  B[0][1] B[1][1] B[2][1] ... B[15][1]   │ cols 0..15
  ...                                          │
  L15  B[0][15] ........        ... B[15][15] ┘
  L16  B[0][0] ............     ... B[15][0]  ┐ DUPLICATE
  ...                                          │
  L31  B[0][15] ............    ... B[15][15] ┘


C frag output after WMMA (8 fp32 values per lane):
═══════════════════════════════════════════════════
  lane t, slot i  →  C[m = 2*i + lane_hi][n = lane_lo]
  with lane_hi = t >> 4 ∈ {0, 1}, lane_lo = t & 15

  Lane axis encodes N (output column)
  Slot axis encodes M (output row, with hi-bit interleave)

       slot 0   slot 1   slot 2   ...  slot 7
       ──────   ──────   ──────   ──   ──────
  L 0  C[0][0]  C[2][0]  C[4][0]  ...  C[14][0]    ┐ lane_hi=0
  L 1  C[0][1]  C[2][1]  C[4][1]  ...  C[14][1]    │ → EVEN rows
  ...                                               │ {0,2,4,...,14}
  L15  C[0][15] C[2][15] C[4][15] ...  C[14][15]   ┘
  L16  C[1][0]  C[3][0]  C[5][0]  ...  C[15][0]    ┐ lane_hi=1
  L17  C[1][1]  C[3][1]  C[5][1]  ...  C[15][1]    │ → ODD rows
  ...                                               │ {1,3,5,...,15}
  L31  C[1][15] C[3][15] C[5][15] ...  C[15][15]   ┘
```

### Why this layout matters (mode 1 vs the alternatives)

The probe op (`gptq_gemm_rdna3_wmma_probe`) tests four hypotheses for
how A and B fragments map to memory:

```
mode 0:  A row-major, B row-major    → computes  A @ B^T
mode 1:  A row-major, B col-major    → computes  A @ B    ✓
mode 2:  A col-major, B row-major    → computes  A^T @ B^T
mode 3:  A col-major, B col-major    → computes  A^T @ B
```

Tests with `A = identity` pass for both mode 0 and mode 1 because the
K-axis sum collapses, masking the bug. Random-A tests reveal mode 1
as the unique correct choice. Hardcoded across the kernel — flipping
B's load axis would silently corrupt prefill output.

### Why WMMA helps bf16 but not fp16 (current state)

```
bf16 scalar (gfx11):       fp16 scalar (gfx11):
═══════════════════        ═══════════════════
  v_pk_fma_bf16  ✗           v_pk_fma_f16  ✓ full rate
   → fallback to             → 1 cycle per packed FMA
     v_fma_f32 + cvt          → kernel becomes memory-bound
   → ~2× slower per FMA       → ~26 % HBM peak (already good)
   → COMPUTE-BOUND
   → WMMA wins by avoiding   bit-trick dequant 0x6400:
     per-element FMAs         → 5 ops + 4 FMAs per int32
     (1.5-2× scalar)          → essentially free

bf16 dispatch  → WMMA at M ≥ 16  ✓
fp16 dispatch  → ALWAYS scalar (WMMA underperforms at
                 ~10 % CU occupancy with the current
                 1-wave-per-block / 16×16 tile)
```

### Concrete scaling — qkv-square (K = N = 4096) on a 96-CU GPU

```
Decode (M=1):
  Scalar: 64 blocks x 8 waves = 512 waves  -> ~17 % of 96 CUs at peak
  WMMA:   256 blocks x 1 wave  = 256 waves -> only 8 CUs busy + wastes
                                              15/16 of the M tile
  -> SCALAR wins easily on tiny M (matches microbench)

Prefill (M=64):
  Scalar: 4 col-blocks x 8 row-blocks (M_COUNT=8) x 16 K-splits
          = 512 blocks x 8 waves
  WMMA:   256 col-blocks x 4 row-blocks x 1 wave
          = 1024 blocks x 1 wave
  -> WMMA wins on bf16 (compute-bound), ties or loses on fp16
     (kernel still memory-bound; bit-trick keeps scalar competitive)
```

## Performance reference (RX 7900 XTX, Qwen3.6-27B-GPTQ-W4A16-G32, tp=2)

End-to-end throughput on `evalscope perf openqa` (50 reqs, max-num-seqs=8):

| Config                                      | Concurrency 1 | Concurrency 100 |
| ---                                         | ---           | ---             |
| ExLlama fp16 (no bf16 support)              | (baseline)    | 255 tk/s        |
| Triton W4A16 fp16                           | —             | 83 tk/s         |
| Triton W4A16 bf16                           | —             | 82 tk/s         |
| **RDNA3 W4A16 fp16**                        | par fp16      | **247 tk/s**    |
| **RDNA3 W4A16 bf16** (initial branch)       | par fp16      | ~100 tk/s       |
| **RDNA3 W4A16 bf16** (after this PR)        | par fp16      | **234 tk/s**    |

The bf16 path under load was the last gap. Three stacked changes closed it
(see lessons 7–8 below): packed atomic CAS-64, factored scale/zb fold for
M_COUNT=1 decode, and col-outer `#pragma unroll 1` to free VGPR pressure
in the M_COUNT=4/8 templates (decode batched). Net: bf16 gained +134%
under load and now sits at ~95% of the best fp16 baseline (ExLlama).

VGPR / spill audit at gfx1100 (`.note.AMDGPU.metadata` from the .so):

| Kernel       | VGPR before | VGPR after | Spill before     | Spill after |
| ---          | ---         | ---        | ---              | ---         |
| bf16 M=1     | 82          | 41         | 0                | 0           |
| bf16 M=2     | 138         | 56         | 0                | 0           |
| bf16 M=4     | 192 (cap)   | 82         | 38 VGPR / 156 B  | **0**       |
| bf16 M=8     | 192 (cap)   | 130        | 144 VGPR / 580 B | **0**       |
| fp16 M=*     | unchanged   | unchanged  | 0                | 0           |

bf16 M=8 (the decode-batched template under load) went from 5 waves/SIMD
with 144 VGPRs spilled to scratch every K-iteration, to ~7 waves/SIMD
with zero scratch traffic. The fp16 path was already register-clean
thanks to the `0x6400` bit-trick + `v_pk_fma_f16` (1 cycle/packed-FMA),
and only the atomic-64 swap applies there.

WMMA bf16 prefill (`gptq_gemm_rdna3_wmma`, M ≥ 16):

| Shape (M=32, K=N=4096)        | bf16 WMMA throughput |
| ---                           | ---                  |
| Initial branch                | 187 tk/s             |
| After single-wave latency fix | **365 tk/s** (+95%)  |
| fp16 scalar reference         | 500 tk/s             |

The fp16 scalar reference is the same workload going through the scalar
M_COUNT=8 template at gridDim.y=4 (= 4×) — the WMMA path still trails it
because of single-wave-per-block giving only ~17% CU saturation vs the
scalar's ~67%. Closing that gap requires K-split (next round, untested).

## Diagnostic ops (registered, callable from Python)

For kernel correctness debugging only — not used in production paths:

* `torch.ops._C.gptq_gemm_rdna3_wmma_probe(a, b, mode)` — runs one WMMA on
  fp16 16×16 inputs under `mode ∈ {0..3}` fragment-load hypotheses, dumps
  per-lane c_acc to fp32[32, 8]. Used to identify the wave32 fragment
  layout (mode 1: A row-major + B col-major + output `[m=2*i+lane_hi]
  [n=lane_lo]`).
* `torch.ops._C.gptq_gemm_rdna3_wmma_dump(...)` — full-pipeline c_acc
  dump from a single 16×16 output tile after dequant + LDS + WMMA.
* `torch.ops._C.gptq_gemm_rdna3_wmma_lds_check(...)` — dequant + LDS-write
  only, dumps the b_lds tile back as fp16 for visual sanity check.

## Tools (`tools/` subdirectory)

Scripts used during development of these kernels. Preserved here as
backup so future regression work can reuse them without rebuilding from
scratch. All run inside the container where vLLM is installed
(`python3 tools/<script>.py`).

### Benchmarks

| Script | Purpose |
| --- | --- |
| `bench_wmma_threshold.py` | M-sweep `{1,4,8,16,24,32,48,64,96,128,256}` × 5 Qwen-class shapes × `{fp16, bf16}`. Calls both `gptq_gemm_rdna3` (scalar path for fp16, dispatches to WMMA for bf16 M≥16) and `gptq_gemm_rdna3_wmma` directly to compare. Median of 50 iters with `cuda.Event` timing. Produces the table that justifies the bf16-only WMMA gating decision and lets us decide if a kernel change wins or regresses. |
| `bench_fp16_decode.py` | Focused single-dtype, M=1 only, 5 shapes × 200 iters. Designed as a stable, repeatable target for `rocprof` (large stable hot loop with no shape variation noise per iteration). Use this when you want HW counter data on fp16 decode specifically. |
| `bench_scalar.py` | Minimal scalar-only microbench (one shape, hardcoded). Bypasses vLLM dispatch entirely via `torch.ops._C.gptq_gemm_rdna3`. Useful as a smoke test that the C++ extension loaded and the kernel runs at all. |
| `bench_full_decode.py` | Simulates one full decode token (~32 layers × 7 linears = 224 GEMM calls). Times three call paths: direct `torch.ops._C`, the Python `ops.gptq_gemm_rdna3` wrapper, and the full `RDNA3W4A16LinearKernel.apply_weights`. Used historically to localise a 7× decode regression to the apply_weights dispatch logic vs. the kernel itself. |

For shape exploration the benchmarks call `gptq_gemm_rdna3_wmma`
directly (regardless of dtype) so the WMMA path can be timed independently
of the dispatch in `gptq_gemm_rdna3`.

### Profiling

| Script | Purpose |
| --- | --- |
| `rocprof_counters.txt` | PMC counter spec for gfx1100 (`pmc:` lines). Lists the SQ_* counters known to register on RDNA3. Used as `-i` input to rocprofv2 / rocprofv3. NOTE: most non-SQ_BUSY_CYCLES / SQ_WAVES counters return 0 on gfx1100 even when listed by `--list-counters` — see "Known limitations" below. |
| `analyze_profile.py` | Aggregates a rocprof v1 trace CSV (`--hip-trace --hsa-trace`) by kernel basename, prints top 20 by total time. Works even without HW counters — pure timing. Useful to confirm where wall-clock time goes. |
| `analyze_counters.py` | Reads rocprofv2 `results_pmc1.csv` + `results_pmc2.csv` (wide format) and computes derived metrics: VALU%, LDS%, WaitRatio%, VMEM%. Loosely v2-specific; the on-disk paths v2 generates are unpredictable, so check `find` first. |
| `analyze_v3.py` | Reads the rocprofv3 long-format `<pid>_counter_collection.csv` (one row per `(dispatch, counter)` tuple). Pivots to `(kernel, grid)` keys and prints the same derived metrics as `analyze_counters.py`. Preferred — rocprofv3 is the only profiler that worked reliably for us on gfx1100 (v1 is unsupported on RDNA3, v2 has output-path bugs). |

### Recommended profiling workflow on gfx1100

1. Run `bench_fp16_decode.py` once standalone to confirm the bench
   finishes without errors.
2. Single-pass with up to 4 SQ counters (avoids hardware multiplexing):
   ```
   rocprofv3 --pmc SQ_BUSY_CYCLES SQ_WAVES SQ_WAVE_CYCLES SQ_WAIT_ANY \
             -d /tmp/profile_v3 \
             --output-format csv \
             -- python3 tools/bench_fp16_decode.py
   ```
3. Analyse with `python3 tools/analyze_v3.py /tmp/profile_v3/<host>/<pid>_counter_collection.csv`.
4. For a second pass with instruction mix:
   ```
   rocprofv3 --pmc SQ_INSTS_WAVE32 SQ_INSTS_WAVE32_VALU SQ_INSTS_WAVE32_LDS SQ_INST_CYCLES_VMEM \
             -d /tmp/profile_v3_p2 \
             --output-format csv \
             -- python3 tools/bench_fp16_decode.py
   ```
   These counters are documented to exist on gfx1100 but in practice may
   return 0 — verify with the analysis script before trusting them.

### 1. The bf16 fp32 dequant fix is essential

`__hfma2(bf162_t, bf162_t, bf162_t)` lowers to a slow fallback on gfx11
(no `v_pk_fma_bf16` instruction). Per-element it's ~2× the cycle count of
`v_pk_fma_f16`. The bf16 scalar path was decode-bound on this until two
fixes:

* `dot22_8_f` widens bf16 → fp32 via free `(bits<<16)` left-shift and
  accumulates with `v_fma_f32` (full rate on RDNA3).
* `dequant_4bit_8_bf16_f32` produces fp32 output directly from the int4
  unpack, bypassing a second round of slow bf16 FMAs.

Both bf16 paths use fp32 throughout for compute; only the I/O is bf16.
Net: bf16 decode 36 → 50 tk/s on Qwen3.6-27B.

The fp16 path keeps `__hfma2(half2,...)` because `v_pk_fma_f16` IS native
and full rate; widening to fp32 would just cost more VGPRs without speed.

### 2. Dispatch lives in C++, not Python

An earlier attempt put `if x.size(0) >= 16: wmma_op(...) else: scalar(...)`
inside `apply_weights`. torch.compile / Dynamo broke decode 7× — the
size-comparison guard triggered graph break/recompile on every layer.
Fix: branch in `gptq_gemm_rdna3`'s C++ entry; Python sees a single op.

The same constraint kills `print()` for debugging — Dynamo can't trace
builtin print in fullgraph mode. Use vLLM's startup logs or
`process_weights_after_loading` (called outside compile) for diagnostics.

### 3. WMMA fragment layout is mode 1, not mode 0

A diagonal-A test passes for both `B row-major` and `B col-major` fragment
loadings because A=I makes the K-axis sum collapse. Random-A reveals only
mode 1 (`A row, B col, output [m=2*i+lane_hi][n=lane_lo]`) implements
A·B; the others compute A·Bᵀ or worse. The `gptq_gemm_rdna3_wmma_probe`
op was built specifically to disambiguate this empirically.

### 4. WMMA TU lives in its own file

A monolithic TU with both scalar and WMMA kernels miscompiled the M=1
scalar path even when the WMMA template was never instantiated for M=1.
hipcc appears to scope some optimizer decisions (register file / SGPR
pressure heuristics) at the TU level. Splitting into separate
translation units (linked via standard cross-TU calls) restores scalar
binary identity to its tuned baseline.

### 5. BLOCK_KN_SIZE = 256 is the sweet spot

Tried 512 to halve atomic CAS count per output. bf16 gained 5-10% at
large M, but fp16 decode regressed up to 40% on `qkv-square` (M=1, only
~8 of 96 CUs saturated due to 16-wave blocks). Reverted; comment in
`q_gemm_rdna3.cu` records the experiment.

### 6. Atomic accumulation via packed CAS, no fp32 buffer

gfx11 has no `v_global_atomic_pk_add_{f16,bf16}`. The kernel writes to
fp16/bf16 output directly via an `atomicCAS`-retry loop. Saves M*N*4
bytes allocation + memset + epilogue cast pass that an fp32-accumulator
design would need (~5-10 μs/call, 11-22% of decode budget at 50 tk/s).
The CAS retries are rarely triggered because each output position's
K-splits finish at different rates.

The CAS targets a 64-bit word (`global_atomic_cmpswap_b64`) covering all
4 output lanes per row in a single atomic operation. Earlier versions
issued two `b32` CAS calls (one per (lo, hi) half2/bf162); merging them
halves the atomic instruction count and the contention window per
output position. Alignment is guaranteed by `can_implement()` requiring
`partition_weight_shape[1] % 8 == 0` and `n` always a multiple of 4.

### 7. bf16 M_COUNT=1 decode: factor scale/zb out of dequant

The default per-col dequant computes `dq[i] = q_f32[i] * scale + zb`
(8 fp32 FMAs per int32 weight × 4 N cols = 32 dequant FMAs), then dot
adds 8 fp32 FMAs per (m, n_col). At M_COUNT=1 this reduces to:

```
accum = sum_i (q_f32[i] * scale + zb) * a[i]
      = scale * sum_i (q_f32[i] * a[i]) + zb * sum_i a[i]
```

Compute `sum_a = Σa[i]` once per K=8 step (shared across all 4 N cols)
and a per-col `partial = Σ(q_f32 * a)`, then fold scale and zb_neg into
the accumulator with 2 FMAs per col. Drops 64 → 47 FMAs per int32
weight at M_COUNT=1 (−27%). Break-even at M_COUNT=2, so guarded behind
`if constexpr (M_COUNT == 1)`.

The new `dequant_4bit_8_bf16_q_only` produces unscaled fp32 q-values
directly from the bf16 bit-trick (0x4300 magic) — zero FMAs in the
dequant itself. Numerically safe because the entire factored path runs
in fp32 (the gfx11 widen sidestep means we already pay fp32 acc).

This trick does **not** apply to fp16: the +1024 bias in the `0x6400`
bit-trick would cause catastrophic cancellation when subtracting
`(1024+zero)*scale*sum_a` from `scale*Σ(q_h2*a)` in fp16-precision
accumulators. Keep fp16 with its native bit-trick + `v_pk_fma_f16`.

### 8. VGPR pressure trumps ILP at the cap — `#pragma unroll 1` on col-loops

The bf16 j-loop originally declared `float dq[4][8]` outside the m-loop
(all 4 cols' dequant results alive simultaneously across the m-loop).
At M_COUNT=4/8 this hit the 192-VGPR cap with 38 / 144 VGPRs spilled
to scratch, costing ~580B of v_writelane/v_readlane traffic per
K-iteration and pinning occupancy at 5 waves/SIMD.

Restructured to col-outer with `dq` declared inside the col loop —
expecting the compiler to free the registers between iterations.
**It didn't.** With `#pragma unroll`, the AMDGPU optimizer expanded the
4 cols into a straight-line block where all 4 dq arrays remained
alive simultaneously to maximize ILP across cols. VGPR count was
unchanged.

Adding `#pragma unroll 1` on the col loop forces a real 4-iteration
loop. The register allocator is then bound to recycle VGPRs across
iterations — `dq` lives only within one col-iter. Inner loops (m, i)
stay `#pragma unroll`d for FMA pipelining within a col.

Outcome: bf16 M_COUNT=8 went from 192 VGPRs + 144 spilled / 580 B
scratch to 130 VGPRs / 0 spilled / 0 scratch. ~7 waves/SIMD vs 5
previously; no more scratch IO in the inner loop. End-to-end bf16
under load went from ~100 tk/s to 234 tk/s.

The same trick applied to the M_COUNT=1 factored path (`q_f32[8]`
inside the col loop instead of `q_f32[4][8]` outside): VGPRs dropped
82 → 41. Generally: when scope-inside-loop alone doesn't reduce
pressure on AMDGPU, force a real loop with `#pragma unroll 1`.

**Caveat — does not apply universally:** tried the same col-outer +
`#pragma unroll 1` on the fp16 scalar path expecting the +12 VGPRs
(110 → ~98 at M_COUNT=8) to cross an occupancy threshold. Wall-time
regressed 247 → 200 tk/s (−19%) in the same evalscope bench. fp16 was
register-clean (no spills), so the trade-off was pure ILP loss vs
marginal occupancy gain — and the AMD compiler had been interleaving
the 4 independent col-dot accumulator chains, which `unroll 1` killed.
Reverted. Heuristic: only apply when the audit shows
`.vgpr_spill_count > 0` or `.private_segment_fixed_size > 0`. Without
spills, trust the compiler's ILP scheduling.

### 9. WMMA single-wave block: every stall blocks the whole block

`gemm_q4_wmma_kernel` launches with `dim3 block(32)` = exactly one
wave32. There is no second wave to overlap with stalls, which makes the
kernel uniquely sensitive to long-latency operations in the K-loop.
Two such operations were costing 95% of the wall-time:

* **`__syncthreads()` × 2 per K-iteration.** A single-wave block has
  no inter-wave concurrency to synchronize, but the explicit barrier
  still emits `s_barrier` and stalls the wave (~30 cycles). The
  compiler-inserted `s_waitcnt lgkmcnt(0)` already orders dependent
  ds_write/ds_read pairs within a wave (including across-lane reads —
  e.g., lane 0 reading what lane 16 wrote into b_lds[8..15][0]). Over
  256 K-iterations on K=4096 that's ~15K wasted cycles per block.

* **Sequential A-row global loads.** Each lane loaded its 16
  fp16/bf16 A elements with 16 separate `global_load_b16`. Each load
  has ~200 cycles of HBM latency. With one wave there is no other
  wave to schedule during the wait — every load is a serial stall
  against the same wave. That's ~3200 cycles/K-tile × 256 K-tiles =
  ~819K cycles per block. Replacing with a 32-byte bulk
  `__builtin_memcpy` lowered to two `global_load_b128` instructions
  drops this to ~400 cycles/K-tile (~102K cycles per block).

Combined, the two changes took bf16 WMMA from 187 to 365 tk/s — same
algorithm, same VGPR class, just removing the serial stalls that the
single-wave layout had no way to hide.

**Implementation note for the A vectorise change:** memcpy into the
whole vector (`__builtin_memcpy(&a_frag, ptr, 32)`), not into individual
elements. `&a_frag[0]` on `ext_vector_type` is not reliably a valid C
pointer across compiler versions; building this on hipcc 7.2.1 fails
with the per-element form. The whole-vector address works.

The remaining gap to fp16 scalar (365 vs 500 tk/s) is CU saturation:
WMMA is 256 N-tiles × 1 wave = 512 waves at M=32, vs fp16 scalar's
2048 waves. The scalar path has gridDim.y splitting M into 4 row-blocks
and gridDim.z splitting K into 16 K-blocks (with atomic write-back).
Replicating the K-split in the WMMA kernel would 4× the wave count
and is the next attack.

## Known limitations / future work

* **fp16 prefill is scalar.** WMMA fp16 is gated off until the kernel is
  competitive with scalar. Likely needs: 16M×128N tile, LDS-shared A
  across waves, K-pipelining (double-buffered LDS), `uint4` vector loads
  for weights. Estimated 3-5× to reach scalar parity at M≥64.

* **No fp16 → fp32-output-buffer experiment.** Replacing `atomicCAS` with
  fp32 atomic-add + epilogue cast might save 5-15% on fp16 decode if CAS
  retries are dominant. Untested.

* **`rocprof` on gfx1100 returns 0 for most useful counters.**
  Attempted: `rocprof v1` (unsupported on RDNA3), `rocprofv2` (works
  but writes CSVs to unpredictable paths and silently drops some
  counters), `rocprofv3` (cleanest, but only `SQ_BUSY_CYCLES` and
  `SQ_WAVES` return non-zero values on the GPTQ kernel — the
  instruction-mix and cycle-accounting counters listed in
  `--list-counters` register but report 0). Without compute-vs-memory
  telemetry, optimisation decisions are blind. The remaining ~74% gap
  to HBM peak (fp16 scalar at ~26%) could be atomic CAS contention,
  LDS bank conflicts, instruction-issue stalls, or occupancy — but we
  can't tell which without working counters. See `tools/rocprof_counters.txt`
  for the counter list we tested and `tools/analyze_v3.py` for the parser.

* **fp16 decode CU saturation.** At `K=N=4096, M=1` the scalar fp16
  kernel launches 64 blocks (`(N/1024) * (M/M_COUNT) * (K/256)`)
  against 96 CUs — only ~17% of wave slots utilized. The kernel is
  memory-bound at ~26% HBM peak, and increasing block count would
  give the load/store scheduler more outstanding requests in flight.
  Reducing `THREADS_X` from 256 to 128 (with `BLOCK_KN_SIZE=256`
  unchanged, each thread loading 2 K elements) would double
  `gridDim.x`. Expected ~+10-20% on fp16 decode; no expected
  regression on prefill (M_COUNT=8 still has plenty of work per
  block). Not yet implemented because the M_COUNT=1/2/4/8 templates
  would need re-tuning together.
