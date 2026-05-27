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
// STATUS (2026-05-27): correct and working in vLLM (coherent output, lockstep
// verified) but OFF BY DEFAULT (VLLM_HOSTAR=0) because it does not beat RCCL on
// this box — AllReduce is not on the decode critical path, so it measures ~equal
// at conc1 and a few % slower at higher concurrency. Two fixes were required to
// get here: (1) back the cross-process buffer with an anonymous memfd, NOT a
// file-backed /dev/shm mapping (registering the latter corrupts vLLM's CUDA-
// graph pool on this ROCm stack); (2) reduce OUT-OF-PLACE — vLLM's all_reduce
// custom op is functional, so the input must be left intact and the sum written
// to a fresh output. See project memory for the full investigation.
#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

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
// Each rank owns a DOUBLE buffer (two maxn-sized halves); the round parity picks
// the half, so consecutive rounds never reuse the same buffer. That lets the
// consumer read round k's buffer with no read-acknowledge phase: by the time a
// rank rewrites a half (round k+2) the peer has provably consumed it (round k),
// enforced by the data-ready (Fw) wait alone.
__global__ void k_put(const uint16_t* src, uint16_t* slotbase, int maxn,
                      int rank, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t* slot = slotbase + (size_t)(g_cur[rank] & 1) * maxn;
  if (i < n) slot[i] = src[i];
}
// out = src + peer, in a single pass. The reduction is OUT-OF-PLACE (vLLM's
// all_reduce op is functional, so the rank's input must stay intact): the FIRST
// peer is added with src = the input (seeds the output without a separate copy
// kernel), and any further peers accumulate with src = out. The peer slot is
// read with SYSTEM-scope acquire loads (32-bit, two 16-bit elems per load) so
// each read goes to memory rather than this GPU's stale read cache — defeating
// the cross-chipset "consumer reads stale slot data after the flag" race.
__global__ void k_add(__half* out, const __half* src, const __half* pbase,
                      int maxn, int rank, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = n >> 1;
  const __half* p = pbase + (size_t)(g_cur[rank] & 1) * maxn;
  if (i < pairs) {
    const unsigned* p32 = reinterpret_cast<const unsigned*>(p);
    unsigned w = __hip_atomic_load(&p32[i], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned short lo = w & 0xFFFFu, hi = w >> 16;
    out[2 * i] = __hadd(src[2 * i], *reinterpret_cast<__half*>(&lo));
    out[2 * i + 1] = __hadd(src[2 * i + 1], *reinterpret_cast<__half*>(&hi));
  } else if ((n & 1) && i == pairs) {  // odd tail
    unsigned short v = __hip_atomic_load(
        reinterpret_cast<const unsigned short*>(p) + (n - 1), __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    out[n - 1] = __hadd(src[n - 1], *reinterpret_cast<__half*>(&v));
  }
}
__global__ void k_add_bf16(__hip_bfloat16* out, const __hip_bfloat16* src,
                           const __hip_bfloat16* pbase, int maxn, int rank,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = n >> 1;
  const __hip_bfloat16* p = pbase + (size_t)(g_cur[rank] & 1) * maxn;
  if (i < pairs) {
    const unsigned* p32 = reinterpret_cast<const unsigned*>(p);
    unsigned w = __hip_atomic_load(&p32[i], __ATOMIC_ACQUIRE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned short lo = w & 0xFFFFu, hi = w >> 16;
    out[2 * i] = src[2 * i] + *reinterpret_cast<__hip_bfloat16*>(&lo);
    out[2 * i + 1] = src[2 * i + 1] + *reinterpret_cast<__hip_bfloat16*>(&hi);
  } else if ((n & 1) && i == pairs) {
    unsigned short v = __hip_atomic_load(
        reinterpret_cast<const unsigned short*>(p) + (n - 1), __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    out[n - 1] = src[n - 1] + *reinterpret_cast<__hip_bfloat16*>(&v);
  }
}

struct Ctx {
  int rank, n_ranks, maxn;
  __half* slots[HOSTAR_MAX_RANKS];  // per rank: a DOUBLE buffer (2 * maxn elems)
  int* F;                           // shared data-ready flags Fw[n_ranks]
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
  // Layout: per rank a double buffer (2 * max_elems 16-bit elems), then the
  // data-ready flags Fw[n_ranks]. Double buffering removes the read-ack phase,
  // so only one flag array is needed.
  size_t bytes =
      (size_t)n_ranks * 2 * max_elems * 2 + n_ranks * sizeof(int) + 128;

  // Cross-process shared buffer via memfd (NOT shm_open). Registering a
  // file-backed /dev/shm mapping with hipHostRegister corrupts vLLM's CUDA-
  // graph memory pool on this ROCm stack (measured); an anonymous-backed memfd
  // does NOT. Rank 0 creates the memfd and publishes <pid fd> to a tiny plain
  // file (NOT registered, so harmless); peers open it through /proc/<pid>/fd to
  // share the same anonymous object, then everyone mmaps + registers it.
  char rdv[128];
  snprintf(rdv, sizeof(rdv), "/tmp%s.rdv", shm);  // e.g. /tmp/vllm_hostar.rdv
  int fd = -1;
  if (rank == 0) {
    unlink(rdv);
    fd = (int)syscall(SYS_memfd_create, "hostar", 0u);
    if (fd < 0 || ftruncate(fd, bytes) != 0) return -1;
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%d %d", (int)getpid(), fd);
    int rf = open(rdv, O_CREAT | O_WRONLY | O_TRUNC, 0666);
    if (rf < 0 || write(rf, buf, len) != len) return -1;
    close(rf);
  } else {
    int pid = -1, pfd = -1;
    for (int t = 0; t < 500 && pid < 0; t++) {  // wait for rank 0 (~5s max)
      int rf = open(rdv, O_RDONLY);
      if (rf >= 0) {
        char buf[64] = {0};
        if (read(rf, buf, sizeof(buf) - 1) > 0) sscanf(buf, "%d %d", &pid, &pfd);
        close(rf);
      }
      if (pid < 0) usleep(10000);
    }
    if (pid < 0) return -1;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/fd/%d", pid, pfd);
    fd = open(path, O_RDWR);  // dup rank 0's memfd into this process
    if (fd < 0) return -1;
  }
  void* p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) return -1;
  if (hipHostRegister(p, bytes, hipHostRegisterPortable) != hipSuccess)
    return -2;
  __half* base = (__half*)p;
  C.F = (int*)(base + (size_t)n_ranks * 2 * max_elems);
  C.rank = rank;
  C.n_ranks = n_ranks;
  C.maxn = max_elems;
  for (int r = 0; r < n_ranks; r++)
    C.slots[r] = base + (size_t)r * 2 * max_elems;  // 2 halves per rank
  if (rank == 0)
    for (int r = 0; r < n_ranks; r++) C.F[r] = 0;
  usleep(500000);  // let rank 0's flag-zeroing land before any peer proceeds
  hipStreamCreate(&C.s);
  hipSetDevice(prev_dev);
  return 0;
}

// Out-of-place all-reduce: `in` holds this rank's partial (read-only), `out`
// receives the sum (out = in + sum of peers). n is the element count
// (<= max_elems). dtype: 0 = fp16, 1 = bf16 (staging is byte identical; only the
// accumulate kernel differs). Stream-capturable. Out-of-place because vLLM's
// all_reduce custom op is functional — `in` must be left untouched.
extern "C" void hostar_allreduce(void* in, void* out, int n, int dtype,
                                 void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  int* Fw = C.F;
  int blocks = (n + 255) / 256;
  int mn = C.maxn;

  // Stage my data into this round's buffer half, publish "ready", wait for every
  // peer. No read-ack phase: double buffering guarantees the half I'm about to
  // write was already consumed by every peer (see k_put).
  k_round<<<1, 1, 0, s>>>(C.rank);
  k_put<<<blocks, 256, 0, s>>>((const uint16_t*)in, (uint16_t*)C.slots[C.rank],
                               mn, C.rank, n);
  k_pub<<<1, 1, 0, s>>>(Fw, C.rank);
  k_spin<<<1, 1, 0, s>>>(Fw, C.n_ranks, C.rank);

  // Read + accumulate every peer's slot into `out` directly from host memory.
  // The first peer reads src = `in` (seeds out without a separate copy, leaving
  // `in` intact); further peers accumulate with src = `out`.
  const void* src = in;
  for (int p = 0; p < C.n_ranks; p++) {
    if (p == C.rank) continue;
    if (dtype == 1)
      k_add_bf16<<<blocks, 256, 0, s>>>((__hip_bfloat16*)out,
                                        (const __hip_bfloat16*)src,
                                        (const __hip_bfloat16*)C.slots[p], mn,
                                        C.rank, n);
    else
      k_add<<<blocks, 256, 0, s>>>((__half*)out, (const __half*)src,
                                   (const __half*)C.slots[p], mn, C.rank, n);
    src = out;
  }
}
#endif  // USE_ROCM
