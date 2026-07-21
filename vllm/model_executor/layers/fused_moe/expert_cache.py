# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-expert CPU offloading cache for fused W4A16 MoE layers.

A fused MoE layer stores every expert as a single 3D tensor with the expert
index on dim 0 (``w13_weight_packed[E, ...]``, ``w2_weight_packed[E, ...]``,
plus the scale / zero-point planes). When the full expert set does not fit in
VRAM, this cache keeps the master copy of all ``E`` experts in **CPU pinned
memory** and holds only a fixed-size subset of ``C`` experts resident on the
GPU. A per-forward :meth:`ExpertCache.ensure` brings the experts a step needs
into GPU slots and returns ``topk_ids`` remapped into slot space ``[0, C)`` so
the existing expert GEMM kernel runs unchanged.

Design notes (see the RDNA3 offload investigation):

* **Kernel-free.** The cache only slices dim 0 of the fused params; it is
  agnostic to the inner W4A16 layout (packed int32, group scales, synthesized
  qzeros). Any backend that gathers experts by id (the native RDNA3 kernel,
  Triton ``fused_moe``) consumes ``slot_of_expert[topk_ids]`` and the ``[C,...]``
  cache tensors.
* **TP-trivial.** Under tensor parallelism (no EP) every rank holds all experts
  but a shard of each expert's intermediate dim, and the router runs replicated,
  so ``topk_ids`` is identical across ranks. Each rank caches its own shard with
  zero cross-rank coordination; the post-``w2`` all-reduce is unaffected.
* **Bandwidth.** The measured per-expert H2D cost on gfx1100 is ~0.104 ms as one
  contiguous blob (25.5 GB/s) vs ~0.139 ms as the 6 separate planes (19.6 GB/s,
  a 1.30x fragmentation tax). ``pack_blob=True`` stores each expert's planes
  contiguously in the pinned master so a miss is a single memcpy.

This module provides the data structure and a correct **eager LFRU reference**
policy (the correctness gate). The CUDA/HIP-graph-capturable copy path is layered
on top once the A/B architecture is chosen (in-graph gather kernel vs
out-of-capture async DMA + prefetch); both reuse the slot/mapping tensors here.
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


class EvictionPolicy(enum.Enum):
    LRU = "lru"
    # Frequency-weighted LRU (a la vLLM #37190): protects "hub" experts from
    # bursty eviction. score = freq / (clock - last_access + 1); lowest evicted.
    LFRU = "lfru"


@dataclass
class ExpertPlaneSpec:
    """One per-expert parameter plane (e.g. ``w13_weight_packed``).

    ``shape`` is the *per-expert* shape (without the leading expert dim).
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def nbytes_per_expert(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n * self.dtype.itemsize


@dataclass
class ExpertCacheConfig:
    num_experts: int
    cache_size: int
    policy: EvictionPolicy = EvictionPolicy.LFRU
    pack_blob: bool = True  # store each expert's planes contiguously in master
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.cache_size > self.num_experts:
            self.cache_size = self.num_experts


@dataclass
class _Plane:
    spec: ExpertPlaneSpec
    master: torch.Tensor  # pinned CPU [E, *shape]
    cache: torch.Tensor   # GPU [C, *shape]


@dataclass
class ExpertCacheStats:
    hits: int = 0
    misses: int = 0
    steps: int = 0
    max_miss_per_step: int = 0
    miss_hist: dict[int, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0

    def record_step(self, step_misses: int) -> None:
        self.steps += 1
        self.max_miss_per_step = max(self.max_miss_per_step, step_misses)
        self.miss_hist[step_misses] = self.miss_hist.get(step_misses, 0) + 1


class ExpertCache:
    """Fixed-size GPU cache over a CPU-pinned master of all experts.

    Lifecycle:
      1. ``add_plane(spec)`` for each per-expert param, then ``allocate()``.
      2. ``set_master(name, expert_id, tensor)`` to fill the (already prepared,
         i.e. post gptq_shuffle / qzero-synth) per-expert weights into pinned RAM.
      3. ``warm(initial_expert_ids)`` to preload the cache.
      4. Per forward: ``slotted = ensure(topk_ids)``; feed ``slotted`` + the
         ``cache`` tensors (via :meth:`gpu`) to the expert GEMM.
    """

    def __init__(self, config: ExpertCacheConfig):
        self.cfg = config
        self.device = config.device or torch.device("cuda")
        self.E = config.num_experts
        self.C = config.cache_size
        self._planes: OrderedDict[str, _Plane] = OrderedDict()

        # expert_id -> slot in [0, C); -1 = not resident. Persistent GPU tensor
        # (never reallocated) so a captured `slot_of_expert[topk_ids]` gather
        # stays valid across replays (KV-block-table pattern, #37190).
        self.slot_of_expert = torch.full(
            (self.E,), -1, dtype=torch.int32, device=self.device
        )
        # slot -> expert_id resident there; -1 = free.
        self.expert_of_slot = torch.full(
            (self.C,), -1, dtype=torch.int32, device=self.device
        )
        self._free_slots: list[int] = list(range(self.C))
        # eviction bookkeeping (host-side, eager reference policy)
        self._lru: OrderedDict[int, None] = OrderedDict()  # expert_id -> None
        self._freq: dict[int, int] = {}
        self._last: dict[int, int] = {}
        self._clock = 0
        self.stats = ExpertCacheStats()
        self._allocated = False

    # ---- setup -----------------------------------------------------------
    def add_plane(self, spec: ExpertPlaneSpec) -> None:
        assert not self._allocated, "add_plane after allocate()"
        self._planes[spec.name] = _Plane(spec=spec, master=None, cache=None)  # type: ignore[arg-type]

    def allocate(self) -> None:
        for name, pl in self._planes.items():
            s = pl.spec
            pl.master = torch.empty(
                (self.E, *s.shape), dtype=s.dtype, pin_memory=True
            )
            pl.cache = torch.empty(
                (self.C, *s.shape), dtype=s.dtype, device=self.device
            )
        self._allocated = True

    def bytes_per_expert(self) -> int:
        return sum(pl.spec.nbytes_per_expert() for pl in self._planes.values())

    def gpu_bytes(self) -> int:
        return self.C * self.bytes_per_expert()

    def set_master(self, name: str, expert_id: int, tensor: torch.Tensor) -> None:
        """Copy a prepared per-expert plane into the pinned master."""
        self._planes[name].master[expert_id].copy_(tensor)

    def gpu(self, name: str) -> torch.Tensor:
        """The GPU cache tensor ``[C, *shape]`` for a plane (fed to the kernel)."""
        return self._planes[name].cache

    # ---- residency management (eager reference: correctness gate) --------
    def _pick_victim(self, exclude: set[int]) -> int:
        """Lowest-priority resident expert not in ``exclude``.

        ``exclude`` holds the current step's experts, which must stay resident
        until the GEMM consumes them (else we'd evict a weight we just loaded).
        """
        if self.cfg.policy is EvictionPolicy.LRU:
            for expert_id in self._lru:  # oldest first
                if expert_id not in exclude:
                    del self._lru[expert_id]
                    return expert_id
            raise RuntimeError("no evictable slot (cache_size < step working set)")
        # LFRU: lowest freq / age among non-excluded
        cands = [e for e in self._freq if e not in exclude]
        if not cands:
            raise RuntimeError("no evictable slot (cache_size < step working set)")
        return min(
            cands,
            key=lambda e: self._freq[e] / (self._clock - self._last[e] + 1),
        )

    def _evict(self, expert_id: int) -> int:
        slot = int(self.slot_of_expert[expert_id].item())
        self.slot_of_expert[expert_id] = -1
        self.expert_of_slot[slot] = -1
        self._freq.pop(expert_id, None)
        self._last.pop(expert_id, None)
        self._lru.pop(expert_id, None)
        return slot

    def _load(self, expert_id: int, slot: int) -> None:
        """Synchronous H2D of one expert's planes into ``slot``."""
        for pl in self._planes.values():
            pl.cache[slot].copy_(pl.master[expert_id], non_blocking=True)
        self.expert_of_slot[slot] = expert_id
        self.slot_of_expert[expert_id] = slot

    def _touch(self, expert_id: int, hit: bool) -> None:
        self._clock += 1
        if self.cfg.policy is EvictionPolicy.LRU:
            self._lru[expert_id] = None
            self._lru.move_to_end(expert_id)
        else:
            self._freq[expert_id] = self._freq.get(expert_id, 0) + 1
            self._last[expert_id] = self._clock

    def warm(self, expert_ids: list[int]) -> None:
        for e in expert_ids[: self.C]:
            if self._free_slots:
                slot = self._free_slots.pop()
                self._load(e, slot)
                self._touch(e, hit=False)

    def ensure(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Make every expert in ``topk_ids`` resident; return slot-space ids.

        Eager reference implementation (host-driven, correct). ``topk_ids`` is
        ``[num_tokens, top_k]``; the return has the same shape with values in
        ``[0, C)`` indexing the GPU cache tensors.

        NOTE: this path syncs to host to inspect the needed experts and is the
        correctness baseline, not the cudagraph path. The graph-capturable copy
        (in-graph gather kernel, or out-of-capture async DMA + prefetch) will
        replace the body while keeping the same slot/mapping tensors.
        """
        needed = [int(e) for e in torch.unique(topk_ids).tolist()]
        if len(needed) > self.C:
            raise ValueError(
                f"step needs {len(needed)} experts > cache_size {self.C}"
            )
        protected = set(needed)  # pin current-step experts against eviction
        step_misses = 0
        for e in needed:
            resident = int(self.slot_of_expert[e].item()) >= 0
            if not resident:
                step_misses += 1
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    slot = self._evict(self._pick_victim(protected))
                self._load(e, slot)
                self.stats.misses += 1
            else:
                self.stats.hits += 1
            self._touch(e, hit=resident)
        self.stats.record_step(step_misses)
        # remap to slot space via the persistent mapping tensor
        return self.slot_of_expert[topk_ids.long()].to(topk_ids.dtype)
