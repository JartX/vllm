# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-compiled HIP gather kernels for W4A16 cold-expert offload (arch A).

Two HIP-graph-capturable kernels (no host sync, fixed launch shapes):

  ec_plan : one block. Dedup the step's needed experts, freshen LRU on hits,
            pick a clock-LRU victim on misses (current-step protection is
            automatic — hits and just-loaded slots are freshened to the max
            clock this step, so argmin never selects them while an older slot
            exists), update slot_of_expert / expert_of_slot, emit an
            (expert -> slot) copy plan padded to max_m.
  ec_copy : one block per plan entry. Valid entries zero-copy the expert's row
            block from device-mapped pinned host master into its GPU slot;
            padded entries are skipped by a SIMT branch.

Measured on gfx1100: in-kernel zero-copy reads from device-mapped pinned host
run at ~26 GB/s (== DMA), and the all-hit path is ~0.007 ms/layer.

Compiled at import via torch cpp_extension so we do NOT rebuild vLLM's _rocm_C.
"""

from __future__ import annotations

import functools
import os

import torch

# Decode hit/miss instrumentation (env VLLM_MOE_OFFLOAD_STATS=1). Each cache
# accumulates device-side counters in ec_plan; an aggregated line is printed
# every _STATS_EVERY tokens across all layers in this worker process. Only the
# dump touches the host, so cudagraph capture is unaffected (though the dump
# only fires under eager, where ensure()'s Python runs every step).
_STATS_ON = bool(os.environ.get("VLLM_MOE_OFFLOAD_STATS"))
_STATS_EVERY = 256  # tokens between dumps
_ALL_CACHES: list[ExpertOffloadCache] = []
_ENSURE_CALLS = 0


def _maybe_dump_stats() -> None:
    global _ENSURE_CALLS
    _ENSURE_CALLS += 1
    n_layers = len(_ALL_CACHES)
    if n_layers == 0 or _ENSURE_CALLS % (n_layers * _STATS_EVERY) != 0:
        return
    hits = misses = steps = 0
    for c in _ALL_CACHES:
        h, m, s = (int(x) for x in c.stats.tolist())
        hits += h
        misses += m
        steps += s
        c.stats.zero_()
    req = hits + misses
    rate = 100.0 * misses / req if req else 0.0
    per_step = misses / steps if steps else 0.0
    print(
        f"[offload-stats] decode miss rate {rate:.1f}% "
        f"(hits={hits} misses={misses} over {steps} layer-steps, "
        f"{per_step:.2f} experts copied/layer-step, C={_ALL_CACHES[0].C})",
        flush=True,
    )

_CPP_SRC = r"""
#include <torch/extension.h>
int64_t ec_device_ptr(torch::Tensor pinned);
void ec_plan(torch::Tensor topk_ids, torch::Tensor slot_of_expert,
             torch::Tensor expert_of_slot, torch::Tensor lru_last,
             torch::Tensor clk, int64_t cache_size, torch::Tensor plan_expert,
             torch::Tensor plan_slot, torch::Tensor stats);
void ec_copy(torch::Tensor plan_expert, torch::Tensor plan_slot,
             int64_t master_devptr, torch::Tensor cache, int64_t row_u4);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void ec_plan_k(const int* __restrict__ topk_ids, int n_sel,
                          int* __restrict__ slot_of_expert,
                          int* __restrict__ expert_of_slot,
                          long long* __restrict__ lru_last,
                          long long* __restrict__ clk, int cache_size, int max_m,
                          int* __restrict__ plan_expert,
                          int* __restrict__ plan_slot,
                          long long* __restrict__ stats) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long now = *clk;
  int pc = 0;      // misses this step (unique experts loaded)
  int uhits = 0;   // unique hits this step (resident, first touch)
  for (int i = 0; i < n_sel; i++) {
    int e = topk_ids[i];
    if (e < 0) continue;
    int s = slot_of_expert[e];
    if (s >= 0) {  // hit: count once, freshen
      if (lru_last[s] != now) { uhits++; lru_last[s] = now; }
      continue;
    }
    bool dup = false;
    for (int j = 0; j < pc; j++) if (plan_expert[j] == e) { dup = true; break; }
    if (dup) continue;
    int victim = 0; long long best = lru_last[0];
    for (int c = 1; c < cache_size; c++)
      if (lru_last[c] < best) { best = lru_last[c]; victim = c; }
    int ve = expert_of_slot[victim];
    if (ve >= 0) slot_of_expert[ve] = -1;
    expert_of_slot[victim] = e;
    slot_of_expert[e] = victim;
    lru_last[victim] = now;
    plan_expert[pc] = e; plan_slot[pc] = victim; pc++;
  }
  for (int j = pc; j < max_m; j++) { plan_expert[j] = -1; plan_slot[j] = -1; }
  *clk = now + 1;
  // device-side accumulation (capturable; no per-step host sync)
  stats[0] += (long long)uhits;  // unique hits
  stats[1] += (long long)pc;     // misses (== experts copied)
  stats[2] += 1;                 // steps
}

__global__ void ec_copy_k(const int* __restrict__ plan_expert,
                          const int* __restrict__ plan_slot,
                          const int4* __restrict__ master,
                          int4* __restrict__ cache, int row_u4) {
  int m = blockIdx.x;
  int e = plan_expert[m];
  if (e < 0) return;  // padded: skip
  int s = plan_slot[m];
  const int4* src = master + (long long)e * row_u4;
  int4* dst = cache + (long long)s * row_u4;
  for (int i = threadIdx.x; i < row_u4; i += blockDim.x) dst[i] = src[i];
}

// Resolve a device-accessible pointer for a pinned host tensor (once, at setup).
int64_t ec_device_ptr(torch::Tensor pinned) {
  void* dptr = nullptr;
  cudaHostGetDevicePointer(&dptr, pinned.data_ptr(), 0);
  return reinterpret_cast<int64_t>(dptr);
}

// Plan the step's residency (dedup + evict + update maps + emit copy plan).
// Fully on-device -> HIP-graph capturable. Call ONCE per MoE step; the plan is
// shared by all weight planes.
void ec_plan(torch::Tensor topk_ids, torch::Tensor slot_of_expert,
             torch::Tensor expert_of_slot, torch::Tensor lru_last,
             torch::Tensor clk, int64_t cache_size, torch::Tensor plan_expert,
             torch::Tensor plan_slot, torch::Tensor stats) {
  int n_sel = topk_ids.numel();
  int max_m = plan_expert.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  ec_plan_k<<<1, 64, 0, stream>>>(
      topk_ids.data_ptr<int>(), n_sel, slot_of_expert.data_ptr<int>(),
      expert_of_slot.data_ptr<int>(),
      reinterpret_cast<long long*>(lru_last.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(clk.data_ptr<int64_t>()), (int)cache_size,
      max_m, plan_expert.data_ptr<int>(), plan_slot.data_ptr<int>(),
      reinterpret_cast<long long*>(stats.data_ptr<int64_t>()));
}

// Zero-copy the planned experts' rows for ONE weight plane (contiguous [C,..]).
// row_u4 = bytes_per_expert_row / 16. Call once per plane with the same plan.
void ec_copy(torch::Tensor plan_expert, torch::Tensor plan_slot,
             int64_t master_devptr, torch::Tensor cache, int64_t row_u4) {
  int max_m = plan_expert.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  ec_copy_k<<<max_m, 256, 0, stream>>>(
      plan_expert.data_ptr<int>(), plan_slot.data_ptr<int>(),
      reinterpret_cast<const int4*>(master_devptr),
      reinterpret_cast<int4*>(cache.data_ptr()), (int)row_u4);
}
"""


@functools.lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="vllm_expert_gather",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["ec_device_ptr", "ec_plan", "ec_copy"],
        with_cuda=True,
        verbose=False,
    )


def device_ptr(pinned: torch.Tensor) -> int:
    return _mod().ec_device_ptr(pinned)


def plan(
    topk_ids: torch.Tensor,
    slot_of_expert: torch.Tensor,
    expert_of_slot: torch.Tensor,
    lru_last: torch.Tensor,
    clk: torch.Tensor,
    cache_size: int,
    plan_expert: torch.Tensor,
    plan_slot: torch.Tensor,
    stats: torch.Tensor,
) -> None:
    _mod().ec_plan(
        topk_ids, slot_of_expert, expert_of_slot, lru_last, clk,
        cache_size, plan_expert, plan_slot, stats,
    )


def copy(
    plan_expert: torch.Tensor,
    plan_slot: torch.Tensor,
    master_devptr: int,
    cache: torch.Tensor,
    row_u4: int,
) -> None:
    _mod().ec_copy(plan_expert, plan_slot, master_devptr, cache, row_u4)


class ExpertOffloadCache:
    """Per-layer W4A16 cold-expert cache: pinned master of all E experts +
    fixed GPU cache of C slots, backed by the plan/copy HIP kernels.

    Each per-expert weight plane is stored as a contiguous, uint4-aligned pinned
    master ``[E, row_ints]`` (device-mapped) and a GPU cache ``[C, row_ints]``.
    :meth:`ensure` plans residency for a step's ``topk_ids`` (expert space) and
    returns ``topk_ids`` remapped to slot space ``[0, C)`` for the GEMM.
    """

    def __init__(self, num_experts: int, cache_size: int, max_sel: int,
                 device: torch.device):
        self.E = num_experts
        self.C = min(cache_size, num_experts)
        self.device = device
        self.planes: dict[str, dict] = {}  # name -> {master, cache, dptr, row_u4}
        self.slot_of_expert = torch.full((self.E,), -1, dtype=torch.int32,
                                         device=device)
        self.expert_of_slot = torch.full((self.C,), -1, dtype=torch.int32,
                                         device=device)
        self.lru_last = torch.zeros(self.C, dtype=torch.int64, device=device)
        self.clk = torch.ones(1, dtype=torch.int64, device=device)
        self.plan_e = torch.full((max_sel,), -1, dtype=torch.int32, device=device)
        self.plan_s = torch.full((max_sel,), -1, dtype=torch.int32, device=device)
        # [unique_hits, misses, steps], accumulated on device by ec_plan
        self.stats = torch.zeros(3, dtype=torch.int64, device=device)
        if _STATS_ON:
            _ALL_CACHES.append(self)

    def add_plane(self, name: str, master_pinned: torch.Tensor,
                  cache_gpu: torch.Tensor) -> None:
        """master_pinned [E, *shape] (pinned), cache_gpu [C, *shape] (device).
        Both must be contiguous and uint4 (16B) aligned per expert row."""
        row_bytes = master_pinned[0].numel() * master_pinned.element_size()
        assert row_bytes % 16 == 0, f"{name} row {row_bytes} not 16B-aligned"
        self.planes[name] = {
            "master": master_pinned,
            "cache": cache_gpu,
            "dptr": device_ptr(master_pinned),
            "row_u4": row_bytes // 16,
        }

    def cache_tensor(self, name: str) -> torch.Tensor:
        return self.planes[name]["cache"]

    def warm(self, expert_ids: list[int]) -> None:
        """Preload experts into slots (host-side, one-time at load)."""
        for slot, e in enumerate(expert_ids[: self.C]):
            self.slot_of_expert[e] = slot
            self.expert_of_slot[slot] = e
            self.lru_last[slot] = 0
            for p in self.planes.values():
                p["cache"][slot].copy_(p["master"][e], non_blocking=True)

    def ensure(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Make the step's experts resident; return slot-space topk_ids."""
        flat = topk_ids.reshape(-1).to(torch.int32)
        plan(flat, self.slot_of_expert, self.expert_of_slot, self.lru_last,
             self.clk, self.C, self.plan_e, self.plan_s, self.stats)
        for p in self.planes.values():
            copy(self.plan_e, self.plan_s, p["dptr"], p["cache"], p["row_u4"])
        if _STATS_ON:
            _maybe_dump_stats()
        return self.slot_of_expert[topk_ids.long()].to(topk_ids.dtype)

    def load_group(self, experts: list[int]) -> torch.Tensor:
        """Make EXACTLY ``experts`` (<= C) resident in slots ``[0, len)``.

        Expert-major prefill: instead of reloading experts per token-chunk
        (``O(tokens/C * E)`` copies -> thrashing), tile the forward's unique
        experts into groups of <= C and load each group once. Not
        graph-capturable (host-driven residency), so this is a prefill-only
        path; decode keeps :meth:`ensure`.

        Returns the ``[E]`` global->slot map (``-1`` = not resident) for use as
        ``expert_map`` in ``moe_align_block_size(..., ignore_invalid_experts=True)``.
        """
        n = len(experts)
        assert n <= self.C, f"group {n} > cache {self.C}"
        ge = torch.tensor(experts, dtype=torch.int32, device=self.device)
        ar = torch.arange(n, dtype=torch.int32, device=self.device)
        self.slot_of_expert.fill_(-1)
        self.expert_of_slot.fill_(-1)
        self.slot_of_expert[ge.long()] = ar
        self.expert_of_slot[:n] = ge
        # loaded slots look "old" so a following decode ensure() evicts them
        # first; empty slots ([n:]) are preferred victims.
        self.lru_last[:n] = 1
        self.lru_last[n:] = 0
        self.plan_e.fill_(-1)
        self.plan_s.fill_(-1)
        self.plan_e[:n] = ge
        self.plan_s[:n] = ar
        for p in self.planes.values():
            copy(self.plan_e, self.plan_s, p["dptr"], p["cache"], p["row_u4"])
        return self.slot_of_expert
