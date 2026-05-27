// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-staged AllReduce shared library (no P2P, RCCL-free) for RX 7900 XTX.
// C ABI loadable from Python via ctypes — shared across vLLM TP worker procs.
// Supports world_size 2 and 4.
//
// Shared host region (POSIX shm) layout: [slot_0..slot_{N-1}][flag_0..flag_{N-1}].
// Each rank D2H-copies its partial into its own slot, bumps its flag, busy-spins
// until every peer flag has reached the same round, then H2D-pulls each peer
// slot and accumulates into its output. Mechanism validated ~43us/10KB.
//
// Compiled into the _C extension (ROCm only); symbols loaded via ctypes.
#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

#define HOSTAR_MAX_RANKS 4

// graph-safe: round counter lives in the shared flag. atomicAdd advances on
// each replay (no host-supplied round), so a captured graph re-syncs every
// launch. F is the flag array base; F[rank] is this rank's round counter.
__global__ void k_sig(int* F, int rank) {
  if (!threadIdx.x)
    __hip_atomic_fetch_add(&F[rank], 1, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
// Wait until every peer flag has reached this rank's round.
__global__ void k_spin(const int* F, int n, int rank) {
  if (!threadIdx.x) {
    int me = __hip_atomic_load(&F[rank], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
    for (int p = 0; p < n; p++) {
      if (p == rank) continue;
      long g = 0;
      while (__hip_atomic_load(&F[p], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM) < me)
        if (++g > 200000000L) break;
    }
  }
}
__global__ void k_add(__half* d, const __half* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __hadd(d[i], p[i]);
}

struct Ctx {
  int rank, n_ranks, maxn;
  __half* slots[HOSTAR_MAX_RANKS];  // shared host slot per rank
  int* F;                           // shared flag array base
  __half* gp;                       // device scratch for peer pulls
  hipStream_t s;
};
static Ctx C;

extern "C" int hostar_init(const char* shm, int rank, int max_elems,
                           int n_ranks) {
  if (n_ranks < 2 || n_ranks > HOSTAR_MAX_RANKS) return -3;
  size_t bytes = (size_t)n_ranks * max_elems * 2 + n_ranks * sizeof(int) + 128;
  int fd = shm_open(shm, O_CREAT | O_RDWR, 0666);
  if (fd < 0) return -1;
  ftruncate(fd, bytes);
  void* p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hipHostRegister(p, bytes, hipHostRegisterPortable) != hipSuccess)
    return -2;
  __half* base = (__half*)p;
  C.F = (int*)(base + (size_t)n_ranks * max_elems);
  C.rank = rank;
  C.n_ranks = n_ranks;
  C.maxn = max_elems;
  for (int r = 0; r < n_ranks; r++) C.slots[r] = base + (size_t)r * max_elems;
  if (rank == 0)
    for (int r = 0; r < n_ranks; r++) C.F[r] = 0;
  usleep(500000);
  hipMalloc(&C.gp, max_elems * 2);
  hipStreamCreate(&C.s);
  return 0;
}
// In-place: g holds my data, gets reduced sum. n<=max_elems. Stream-capturable.
extern "C" void hostar_allreduce(__half* g, int n, void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  hipMemcpyAsync(C.slots[C.rank], g, n * 2, hipMemcpyDeviceToHost, s);
  k_sig<<<1, 1, 0, s>>>(C.F, C.rank);
  k_spin<<<1, 1, 0, s>>>(C.F, C.n_ranks, C.rank);
  for (int p = 0; p < C.n_ranks; p++) {
    if (p == C.rank) continue;
    hipMemcpyAsync(C.gp, C.slots[p], n * 2, hipMemcpyHostToDevice, s);
    k_add<<<(n + 255) / 256, 256, 0, s>>>(g, C.gp, n);
  }
}
#endif  // USE_ROCM
