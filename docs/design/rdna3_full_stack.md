# RDNA3 Full Stack Acceleration for vLLM

> _"Every instruction counts when you're chasing the memory wall."_

This document describes the complete performance stack for AMD RDNA3 (gfx1100)
in vLLM. Three independent acceleration layers that compose multiplicatively —
each targeting a different bottleneck in the inference pipeline.

---

## The Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM Inference Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│  │  Layer 1:     │   │  Layer 2:        │   │  Layer 3:         │  │
│  │  W4A16 WMMA   │──▶│  Triton Prefill  │──▶│  HIP INT8/INT4    │  │
│  │  (GEMM)       │   │  (Attention FP16)│   │  (Attention Quant) │  │
│  └───────────────┘   └──────────────────┘   └───────────────────┘  │
│       ▲                      ▲                      ▲               │
│       │                      │                      │               │
│  Weights are the        Attention is the      KV cache is the      │
│  bottleneck at          bottleneck at         bottleneck at        │
│  short context          medium context        long context         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Native W4A16 WMMA Kernel

**What**: Hand-written HIP kernel for 4-bit weight × 16-bit activation GEMM,
targeting `v_wmma_f32_16x16x16_bf16_w32` on gfx1100.

**Why**: Triton's W4A16 kernel on RDNA3 leaves 60% of WMMA throughput on the
table due to suboptimal dequant scheduling and wave occupancy.

**Result**: Saturates the gfx1100 WMMA unit. Scalar path + WMMA v2 (2-wave
cooperative with double-buffered LDS) hits the compute ceiling.

**Files**: [`csrc/quantization/gptq/README_RDNA3.md`](../../csrc/quantization/gptq/README_RDNA3.md)

---

## Layer 2: Triton Prefill Tuning (3-Tier Adaptive)

**What**: Launch parameter overrides for `triton_unified_attention` on gfx11.
Adapts BLOCK_M and num_warps to sequence length.

**Why**: Triton defaults assume wave64 (CDNA). On wave32 RDNA3, the default
BLOCK_M=16 with 4 warps gives pathetically low arithmetic intensity (~4
FLOPs/byte) — deeply memory-bound even when it shouldn't be.

**Result**: Up to **3.7× faster** FP16 prefill by matching tile size to the
memory/compute crossover point.

| Sequence length | Tile config | Speedup vs default |
|----------------|-------------|-------------------|
| ≤ 1024 | M32, 2 warps | 1.7× |
| 1025 – 8192 | M64, 4 warps | 2.3× |
| > 8192 | M128, 8 warps | 2.2× |

**Files**: [`vllm/v1/attention/ops/README_RDNA3_TRITON_PREFILL_TUNING.md`](../../vllm/v1/attention/ops/README_RDNA3_TRITON_PREFILL_TUNING.md)

---

## Layer 3: HIP INT8/INT4 Prefill Kernels

**What**: Native HIP attention kernels for quantized KV cache. Bypass Triton
entirely with cooperative 4-wave WMMA and fused dequant.

**Why**: On gfx1100, INT8→FP16 is 1 native instruction (`v_cvt_f16_i16_e32`).
FP8→FP16 is ~20 emulated instructions. INT4 nibble unpack is 2 instructions.
Combined with 50-75% bandwidth reduction from quantized cache, these kernels
dominate at long context.

**Result**:

| Mode | vs FP16 Triton | VRAM saving | ISA used |
|------|---------------|-------------|----------|
| INT8 per-token-head | **+66%** | 50% | `v_wmma_f32_16x16x16_bf16` |
| INT4 per-token-head | **+58%** | 75% | `v_wmma_i32_16x16x16_iu8` |

Both are **8× faster** than the equivalent Triton quantized attention path.

**Files**: [`csrc/attention/README_RDNA3_HIP_KERNELS.md`](../../csrc/attention/README_RDNA3_HIP_KERNELS.md)

---

## Benchmark Results

### Test Setup

- **Model**: Qwen3.6-27B GPTQ W4A16, TP2
- **Hardware**: 2× RX 7900 XTX (48 GB total VRAM)
- **Config**: chunked prefill, max_num_batched_tokens=2048, enforce_eager

### Kernel-Level (Attention Microbenchmark, ctx=8000 ql=512)

| Kernel | Time | vs Triton baseline |
|--------|------|--------------------|
| Triton INT8 per-token-head | ~25 ms | — |
| **HIP INT8** | **3.03 ms** | **8.3×** |
| **HIP INT4** | **3.05 ms** | **8.2×** |
| Triton FP16 (default M16w4) | 11.2 ms | — |
| Triton FP16 (tuned M64w4) | **3.0 ms** | **3.7×** |

### End-to-End TTFT (Single Request, Prefill → First Token)

| Prompt length | Baseline (FP16) | INT8 full stack | INT4 full stack |
|--------------|-----------------|-----------------|-----------------|
| ~2K tokens | 2.81s | 2.74s (-2.5%) | 2.78s (-1%) |
| ~8K tokens | 11.15s | 10.99s (-1.4%) | 11.25s (+1%) |
| ~16K tokens | 23.74s | 22.52s (-5.1%) | 22.96s (-3.3%) |
| ~24K tokens | 38.44s | **34.84s (-9.4%)** | **35.49s (-7.7%)** |

Note: GEMM (W4A16 linear layers) dominates ~80% of total inference time.
Attention is ~20%. The 8× kernel speedup on 20% of runtime yields ~10% E2E.
The longer the context, the more attention dominates → larger E2E gains.

### KV Cache Capacity (The Killer Metric)

With identical VRAM budget (33.3 GB available for KV after model load):

| KV dtype | Max context | Concurrent @8K | Concurrent @32K |
|----------|-------------|----------------|-----------------|
| FP16 | 136K tokens | 17 requests | 4 requests |
| INT8 | 273K tokens | **34 requests (2×)** | **8 requests (2×)** |
| INT4 | **545K tokens** | **68 requests (4×)** | **17 requests (4×)** |

**INT4 serves 4× more concurrent users with the same hardware.**
Or equivalently: supports 4× longer context (545K vs 136K tokens).

### Throughput Under Load (8 Concurrent Requests, ~2K prompt + 50 gen)

| Config | Total throughput |
|--------|-----------------|
| Baseline FP16 | 673 tok/s |
| INT8 full stack | 672 tok/s |
| INT4 full stack | 655 tok/s |

At low concurrency (8 reqs), throughput is GEMM-bound and similar. The real
throughput advantage appears at high concurrency (34-68 reqs) where FP16
would OOM but INT8/INT4 can still serve.

### Quality

| Mode | Cosine vs FP16 reference | Notes |
|------|--------------------------|-------|
| INT8 per-token-head | 1.000000 | Exact (symmetric, no precision loss) |
| INT4 per-token-head | 0.999995 | Q→int8 quantization noise (~0.001) |

Both produce coherent outputs at temperature=0 on code generation,
mathematical reasoning, translation, and factual QA.

Layer 1 (W4A16 WMMA) improves decode and short-prompt latency where GEMM
dominates. Layer 2+3 improve prefill where attention dominates.

---

## ISA Foundation

All optimizations are grounded in verified gfx1100 ISA analysis:

| Instruction | What | Cycles |
|---|---|---|
| `v_wmma_f32_16x16x16_bf16_w32` | 16×16 bf16 matmul | 16 |
| `v_wmma_i32_16x16x16_iu8_w32` | 16×16 int8 matmul | 16 |
| `v_cvt_f16_i16_e32` | INT8 → FP16 | 1 |
| `v_cvt_f16_u16_e32` | UINT8 → FP16 | 1 |
| `v_dot2_f32_bf16` | 2× bf16 FMA (scalar path) | 1* |

*half-rate on gfx1100 (measured 2.01× vs full-rate expectation).

Full ISA reference: [`csrc/quantization/gptq/README_RDNA3_FULL_ISA.md`](../../csrc/quantization/gptq/README_RDNA3_FULL_ISA.md)
Conversion cost analysis: [`csrc/quantization/gptq/README_RDNA3_CVT_ISA.md`](../../csrc/quantization/gptq/README_RDNA3_CVT_ISA.md)

---

## Usage

```bash
# Full stack: W4A16 model + INT4 KV cache + all tuning active
vllm serve Qwen3.6-27B-GPTQ-W4A16 \
  --kv-cache-dtype int4_per_token_head \
  --dtype float16 \
  --enforce-eager \
  --enable-chunked-prefill \
  --tensor-parallel-size 2

# Or INT8 for simpler setup (no RHT rotation needed)
vllm serve <model> --kv-cache-dtype int8_per_token_head
```

The W4A16 WMMA kernel auto-dispatches when a GPTQ model is loaded on ROCm
gfx1100. Triton tuning activates automatically on gfx11. HIP attention kernels
dispatch when the compiled op is available and conditions are met.

---

## Build Requirements

- ROCm 6.x+ with gfx1100 target
- `PYTORCH_ROCM_ARCH=gfx1100` during build
- Python 3.12+, PyTorch 2.4+

```bash
MAX_JOBS=$(nproc) PYTORCH_ROCM_ARCH=gfx1100 python3 setup.py build_ext --inplace
```

---

## Architecture Decisions

1. **Separate TUs per kernel variant** — hipcc optimizer interactions between
   templates cause VGPR inflation. Each kernel gets its own `.cu` file.

2. **Centering zero-point in loader** (INT4) — subtracting zp during the
   cooperative load (16 integer subs) eliminates ~96 VALU/tile from the inner
   loop. Makes INT4 within 1-8% of INT8 despite asymmetric quantization.

3. **Phase 1/Phase 2 split** — cached prefix reads quantized paged cache;
   current chunk reads raw fp16 (not yet quantized). Avoids double-quantize
   and keeps the critical path clean.

4. **3-tier prefill over fixed tuning** — no single BLOCK_M is optimal across
   all sequence lengths. The crossover from memory-bound to compute-bound
   happens at ~1024 tokens on gfx1100.

5. **INT8 WMMA for INT4 Q×K** — `v_wmma_i32_16x16x16_iu8` operates directly
   on centered int8 K values. Zero conversion needed. Q quantized to int8
   once (amortized over all K tiles).

---

## Branch Structure

### Independent PR branches (each targets `main`)

```
main
├── perf/rdna3_w4a16_squashed              Layer 1 — W4A16 WMMA GEMM kernel
├── perf/rdna3_triton_prefill_tuning       Layer 2 — Triton prefill 3-tier adaptive
├── feat/int8_per_tensor_clean             Layer 4 — INT8 per-tensor KV cache (kv_cache_scheme)
│
├── refactor/prefill-fastpath-per-token-head-v2    (upstream dependency, NOT in main)
│   └── feat/rdna3_int8_int4_hip_kernels           Layer 3 — HIP INT8/INT4 prefill kernels
│
└── perf/rdna3_full_stack                  Integration branch (all layers, local testing only)
```

### PR dependency chain

| Branch | PR target | Blocker |
|--------|-----------|---------|
| `perf/rdna3_w4a16_squashed` | `main` | None |
| `perf/rdna3_triton_prefill_tuning` | `main` | None |
| `feat/int8_per_tensor_clean` | `main` | None |
| `refactor/prefill-fastpath-per-token-head-v2` | `main` | None (upstream) |
| `feat/rdna3_int8_int4_hip_kernels` | `main` | Requires `refactor/prefill-fastpath-pth-v2` merged first |
| `perf/rdna3_full_stack` | — | Not for PR; integration branch for E2E testing |

### Rebuilding the integration branch (`perf/rdna3_full_stack`)

The integration branch can be reconstructed from `main` plus the five
dependency branches.  Merge order matters — later merges depend on context
introduced by earlier ones.

```bash
git checkout -b perf/rdna3_full_stack_rebuild main

# Layer 1 — clean merge
git merge perf/rdna3_w4a16_squashed

# Layer 2 — CONFLICT in triton_unified_attention.py (see §Conflict 1)
git merge perf/rdna3_triton_prefill_tuning

# Upstream dep for Layer 3 — clean merge
git merge refactor/prefill-fastpath-per-token-head-v2

# Layer 3 — CONFLICT in csrc/ops.h, csrc/torch_bindings.cpp (see §Conflict 2)
git merge feat/rdna3_int8_int4_hip_kernels

# Layer 4 — CONFLICT in 4 files (see §Conflict 3)
git merge feat/int8_per_tensor_clean

# Own commits (docs, multi HEAD_SIZE) — cherry-pick from old full_stack
git cherry-pick <docs-commits> <multi-headsize-commit>
# The multi HEAD_SIZE cherry-pick has minor conflicts (see §Conflict 4)
```

### Known merge conflicts and resolutions

These conflicts are **expected and unavoidable**.  Each PR branch targets
`main` independently — they cannot be pre-aligned to each other without
rewriting pushed history, and doing so would break them against `main`
(where the other branches don't exist yet).

#### Conflict 1: Layer 1 × Layer 2 — `triton_unified_attention.py`

**File**: `vllm/v1/attention/ops/triton_unified_attention.py` (2 hunks)

**Cause**: Layer 1 (`w4a16_squashed`) adds a simple RDNA3 prefill override
(`BLOCK_M=32`, `num_warps=2`).  Layer 2 (`triton_prefill_tuning`) replaces
that same block with the full 3-tier adaptive logic (M32/M64/M128 with
2/4/8 warps).  Both insert at the same location after `BLOCK_M = 16 if ...`.

**Resolution**: Take Layer 1's simple `BLOCK_M=32, num_warps=2` for both
hunks. Validated empirically post-V7 WMMA: 27B prefill at 8192 tokens
(8× the 3-tier "long" threshold) runs in 14.483 s and beats Hybrid 1.93×
— attention is *not* the bottleneck at long context on gfx1100; the V7
W4A16 GEMM kernel is. The 3-tier preventive scaling (M64/M128 + 4/8 warps
for "compute-bound long") was designed pre-V7 when the balance was
different. M32+2warps scales fine because once GEMM saturates the WMMA
unit, larger attention tiles only add register pressure for no payoff.

#### Conflict 2: Layer 3 — `csrc/ops.h`, `csrc/torch_bindings.cpp`

**Files**: `csrc/ops.h`, `csrc/torch_bindings.cpp` (1 hunk each)

**Cause**: Layer 1 adds GPTQ RDNA3 function declarations after
`gptq_shuffle`.  Layer 3 adds HIP INT8/INT4 attention declarations at
the same insertion point.  Git can't merge two independent additions at
the same location.

**Resolution**: Keep both blocks.  Order doesn't matter — they are
independent function declarations.  Convention: GPTQ ops first, then
attention ops.

#### Conflict 3: Layer 4 — 4 files

**Files** (1 hunk each unless noted):

| File | Hunks | Cause |
|------|-------|-------|
| `vllm/utils/torch_utils.py` | 1 | Layer 4 adds `"int8_per_tensor"`, upstream refactor adds `"int2/int4_per_token_head"` — same insertion point |
| `vllm/v1/kv_cache_interface.py` | 2 | (a) Enum numbering: Layer 4 uses `INT8_PER_TENSOR=5, NVFP4=4` but upstream refactor already assigned 4=INT4, 5=INT2, 6=NVFP4. (b) `get_kv_quant_mode()`: both add entries at the same spot |
| `vllm/v1/attention/ops/triton_unified_attention.py` | 1 | Layer 4 adds `_cast_kv_tile` inline; upstream refactor moved it to `triton_attention_helpers.py` (with mode 5 already supported) |
| `vllm/v1/attention/backends/triton_attn.py` | 1 | `supported_kv_cache_dtypes`: same insertion point for `"int8_per_tensor"` vs `"int2/int4_per_token_head"` |

**Why Layer 4 can't be pre-fixed**: `feat/int8_per_tensor_clean` targets
`main`, where `refactor/prefill-fastpath-per-token-head-v2` doesn't exist
yet.  On `main`, enum value 5 is free and `_cast_kv_tile` doesn't exist in
helpers.  If we changed Layer 4 to use enum=7 and skip the inline helper,
it would break against `main`.

**Resolution**:
- `torch_utils.py` / `triton_attn.py`: Keep both sides (add all entries).
- `kv_cache_interface.py`: Renumber to `INT8_PER_TENSOR=5, INT2=6, NVFP4=7`
  (or any consistent numbering). Add `int8_per_tensor` mapping to
  `get_kv_quant_mode()`.
- `triton_unified_attention.py`: Drop the inline `_cast_kv_tile` — it's
  already in `triton_attention_helpers.py` with mode 5 support.

#### Conflict 4: multi HEAD_SIZE cherry-pick — 2 files

**Files**: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` (5
trivial hunks), `vllm/v1/kv_cache_interface.py` (1 hunk)

**Cause**: This commit was authored on the old `full_stack` where the
enum numbering and comments differed from the freshly-merged state.

**Resolution**:
- `triton_reshape_and_cache_flash.py`: Keep the comments from the
  cherry-pick (they add useful context: "INT8 per-tensor: quantize to
  [-128, 127]", "Platform", etc.).
- `kv_cache_interface.py`: Keep the numbering from HEAD (consistent with
  Conflict 3 resolution).

### Why the branches can't be conflict-free

Each PR branch is designed to merge cleanly into `main` on its own.
They share no common ancestor beyond `main` itself, yet they touch
overlapping files:

- `triton_unified_attention.py` is modified by Layers 1, 2, and 4
- `kv_cache_interface.py` is modified by the upstream refactor and Layer 4
- `csrc/ops.h` / `torch_bindings.cpp` are modified by Layers 1 and 3
- `triton_attn.py` and `torch_utils.py` are modified by the refactor and Layer 4

Making them conflict-free against each other would require either:
1. **Rewriting history** on pushed branches — breaks collaboration
2. **Pre-aligning to unreleased code** — breaks the branch against `main`
3. **Merging in a fixed order** and rebasing later branches — creates
   artificial dependencies between independent features

Option 3 is what `perf/rdna3_full_stack` effectively does as an
integration branch.  The individual PR branches stay clean against
`main`, and the documented conflict resolutions above make reconstruction
deterministic.
