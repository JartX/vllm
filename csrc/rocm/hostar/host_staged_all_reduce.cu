// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-staged AllReduce shared library (no P2P, RCCL-free) for RX 7900 XTX.
// C ABI loadable from Python via ctypes — shared across vLLM TP worker procs.
// Supports world_size 2 and 4, fp16 and bf16.
//
// Shared host region (POSIX shm) layout:
//   [slot_0 .. slot_{N-1}][Fw_0..Fw_{N-1}][Fr_0..Fr_{N-1}]
// Each rank copies its partial into its own slot (k_put), publishes a
// data-ready flag (Fw), busy-waits until every peer is ready, then reads +
// accumulates every peer slot, publishes a read-done flag (Fr), and waits
// until peers have consumed its slot before returning (so the next round can't
// overwrite a slot still being read). All ops are kernels on the caller's
// stream → CUDA-graph capturable.
//
// STATUS (2026-05-27): the reduction is correct and graph-capturable — proven
// in a faithful standalone repro (torch CUDAGraph, both GPUs, 100% correct).
// It is NOT usable in vLLM yet: the hipHostRegister() of the cross-process shm
// corrupts vLLM's CUDA-graph memory pool/replay on this ROCm stack (garbled
// model even when the reduction never fires; NOT a VA overlap). Off by default
// (VLLM_HOSTAR=0). See project memory for the full investigation.
#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define HOSTAR_MAX_RANKS 4

// Per-rank round counters in DEVICE memory. Each host flag F[rank] has a single
// writer (this rank), so we never need an atomic RMW on host memory — which
// matters because on this box GPU1 sits behind the PCH chipset and a
// system-scope atomic_add to host memory from GPU1 is NOT visible to the
// host/peer (measured), while a release store IS. The counter is a device-scope
// atomic so it accumulates across graph replays (a plain ++ does not persist
// across captured-graph launches).
__device__ int g_round[HOSTAR_MAX_RANKS];
__device__ int g_cur[HOSTAR_MAX_RANKS];

// Advance this rank's round counter; does not touch host flags yet.
__global__ void k_round(int rank) {
  if (!threadIdx.x) g_cur[rank] = atomicAdd(&g_round[rank], 1) + 1;
}
// Publish a host flag = current round, after a system fence so all prior writes
// to host-pinned memory by this GPU (the slot copy) are visible system-wide
// before the flag (only an explicit __threadfence_system orders bulk data vs
// the flag across the PCH chipset).
__global__ void k_pub(int* F, int rank) {
  if (!threadIdx.x) {
    __threadfence_system();
    __hip_atomic_store(&F[rank], g_cur[rank], __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
// Wait until every peer's flag in `F` has reached this rank's current round.
__global__ void k_spin(const int* F, int n, int rank) {
  if (!threadIdx.x) {
    int me = g_cur[rank];
    for (int p = 0; p < n; p++) {
      if (p == rank) continue;
      while (__hip_atomic_load(&F[p], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM) < me) { /* spin */ }
    }
    __threadfence_system();  // order the spin-acquire before subsequent reads
  }
}
// Copy this rank's data into its host-pinned slot (16-bit elems, dtype-agnostic).
__global__ void k_put(const uint16_t* src, uint16_t* slot, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) slot[i] = src[i];
}
// Accumulate a peer's host-pinned slot into the output. The peer slot is read
// with SYSTEM-scope acquire loads (32-bit, two 16-bit elems per load) so each
// read goes to memory rather than this GPU's stale read cache — defeating the
// cross-chipset "consumer reads stale slot data after the flag" race without a
// blind post-flag delay.
__global__ void k_add(__half* d, const __half* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = n >> 1;
  if (i < pairs) {
    const unsigned* p32 = reinterpret_cast<const unsigned*>(p);
    unsigned w = __hip_atomic_load(&p32[i], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned short lo = w & 0xFFFFu, hi = w >> 16;
    d[2 * i] = __hadd(d[2 * i], *reinterpret_cast<__half*>(&lo));
    d[2 * i + 1] = __hadd(d[2 * i + 1], *reinterpret_cast<__half*>(&hi));
  } else if ((n & 1) && i == pairs) {  // odd tail
    unsigned short v = __hip_atomic_load(
        reinterpret_cast<const unsigned short*>(p) + (n - 1), __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    d[n - 1] = __hadd(d[n - 1], *reinterpret_cast<__half*>(&v));
  }
}
__global__ void k_add_bf16(__hip_bfloat16* d, const __hip_bfloat16* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = n >> 1;
  if (i < pairs) {
    const unsigned* p32 = reinterpret_cast<const unsigned*>(p);
    unsigned w = __hip_atomic_load(&p32[i], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned short lo = w & 0xFFFFu, hi = w >> 16;
    d[2 * i] = d[2 * i] + *reinterpret_cast<__hip_bfloat16*>(&lo);
    d[2 * i + 1] = d[2 * i + 1] + *reinterpret_cast<__hip_bfloat16*>(&hi);
  } else if ((n & 1) && i == pairs) {
    unsigned short v = __hip_atomic_load(
        reinterpret_cast<const unsigned short*>(p) + (n - 1), __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    d[n - 1] = d[n - 1] + *reinterpret_cast<__hip_bfloat16*>(&v);
  }
}

struct Ctx {
  int rank, n_ranks, maxn;
  __half* slots[HOSTAR_MAX_RANKS];  // shared host slot per rank
  int* F;                           // shared flag array base (Fw then Fr)
  hipStream_t s;
};
static Ctx C;

extern "C" int hostar_init(const char* shm, int rank, int max_elems,
                           int n_ranks) {
  if (n_ranks < 2 || n_ranks > HOSTAR_MAX_RANKS) return -3;
  // Pin init-time HIP ops to THIS rank's device (the ctypes-loaded lib may
  // otherwise default to device 0). Save/restore so torch's current device for
  // this thread is untouched.
  int prev_dev = 0;
  hipGetDevice(&prev_dev);
  hipSetDevice(rank);
  // Flag region holds TWO arrays of n_ranks ints: Fw[] (data-ready), Fr[]
  // (read-done), for the two-phase handshake.
  size_t bytes =
      (size_t)n_ranks * max_elems * 2 + 2 * n_ranks * sizeof(int) + 128;
  int fd = shm_open(shm, O_CREAT | O_RDWR, 0666);
  if (fd < 0) return -1;
  ftruncate(fd, bytes);
  void* p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) return -1;
  // NOTE: this hipHostRegister is what corrupts vLLM's CUDA-graph pool (see
  // file header). The reduction below is correct; this call is the blocker.
  if (hipHostRegister(p, bytes, hipHostRegisterPortable) != hipSuccess)
    return -2;
  __half* base = (__half*)p;
  C.F = (int*)(base + (size_t)n_ranks * max_elems);
  C.rank = rank;
  C.n_ranks = n_ranks;
  C.maxn = max_elems;
  for (int r = 0; r < n_ranks; r++) C.slots[r] = base + (size_t)r * max_elems;
  if (rank == 0)
    for (int r = 0; r < 2 * n_ranks; r++) C.F[r] = 0;
  usleep(500000);  // let rank 0's flag-zeroing land before any peer proceeds
  hipStreamCreate(&C.s);
  hipSetDevice(prev_dev);
  return 0;
}

// In-place all-reduce: g holds this rank's partial, receives the sum. n is the
// element count (<= max_elems). dtype: 0 = fp16, 1 = bf16 (staging is byte
// identical; only the accumulate kernel differs). Stream-capturable.
extern "C" void hostar_allreduce(void* g, int n, int dtype, void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  int* Fw = C.F;
  int* Fr = C.F + C.n_ranks;
  int blocks = (n + 255) / 256;

  // Phase 1 — stage my data, publish "ready", wait for every peer.
  k_round<<<1, 1, 0, s>>>(C.rank);
  k_put<<<blocks, 256, 0, s>>>((const uint16_t*)g, (uint16_t*)C.slots[C.rank],
                               n);
  k_pub<<<1, 1, 0, s>>>(Fw, C.rank);
  k_spin<<<1, 1, 0, s>>>(Fw, C.n_ranks, C.rank);

  // Read + accumulate every peer's slot directly from host-pinned memory.
  for (int p = 0; p < C.n_ranks; p++) {
    if (p == C.rank) continue;
    if (dtype == 1)
      k_add_bf16<<<blocks, 256, 0, s>>>((__hip_bfloat16*)g,
                                        (const __hip_bfloat16*)C.slots[p], n);
    else
      k_add<<<blocks, 256, 0, s>>>((__half*)g, (const __half*)C.slots[p], n);
  }

  // Phase 2 — read-acknowledge: publish "I've consumed peers' slots", then wait
  // until every peer has consumed MINE before returning, so the next round's
  // k_put can't overwrite my slot while a peer is still reading it.
  k_pub<<<1, 1, 0, s>>>(Fr, C.rank);
  k_spin<<<1, 1, 0, s>>>(Fr, C.n_ranks, C.rank);
}
#endif  // USE_ROCM
