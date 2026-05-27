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

// Per-rank COMMITTED round counter in DEVICE memory, advanced once per
// all-reduce (in k_pubspin). It is bumped with a device-scope atomicAdd so the
// new value persists across captured-graph replays (a plain store does not).
// During an all-reduce the live round is g_R+1 before k_pubspin commits and g_R
// after; k_put (pre-commit) and k_add (post-commit) therefore both derive the
// same round parity. Each host flag F[rank] has a single writer (this rank) so
// no host-memory RMW is needed — which matters because on this box GPU1 sits
// behind the PCH chipset and a system-scope atomic_add to host memory from GPU1
// is NOT visible to the host/peer (measured), while a release store IS.
__device__ int g_R[HOSTAR_MAX_RANKS];

// Publish "my data is ready", wait until every peer is ready, and commit this
// round — all in one single-thread kernel (folding the old k_round + k_pub +
// k_spin saves two launches per all-reduce). The leading __threadfence_system
// makes this GPU's slot writes (k_put) visible system-wide before the flag
// store; the trailing one orders the peers' slot data (made visible by the
// acquire-loads) before the k_add that follows on the same stream. Both fences
// are required to cross the PCH chipset.
__global__ void k_pubspin(int* F, int n, int rank) {
  if (!threadIdx.x) {
    int me = g_R[rank] + 1;  // live round for this all-reduce
    __threadfence_system();
    __hip_atomic_store(&F[rank], me, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    for (int p = 0; p < n; p++) {
      if (p == rank) continue;
      while (__hip_atomic_load(&F[p], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM) < me) { /* spin */ }
    }
    __threadfence_system();      // order peers' slot data before subsequent k_add
    atomicAdd(&g_R[rank], 1);    // commit: g_R now == this round (persists across replays)
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
  // Pre-commit: live round is g_R+1, so its parity is (g_R+1)&1.
  uint16_t* slot = slotbase + (size_t)((g_R[rank] + 1) & 1) * maxn;
  int vec = n >> 3;  // 128-bit (8x16b) coalesced stores for the bulk PCIe write
  if (i < vec) {
    reinterpret_cast<uint4*>(slot)[i] = reinterpret_cast<const uint4*>(src)[i];
  } else if (i == vec) {  // scalar tail (<8 elems)
    for (int j = 8 * vec; j < n; j++) slot[j] = src[j];
  }
}
// out = src + peer, in a single pass. The reduction is OUT-OF-PLACE (vLLM's
// all_reduce op is functional, so the rank's input must stay intact): the FIRST
// peer is added with src = the input (seeds the output without a separate copy
// kernel), and any further peers accumulate with src = out. The peer slot is
// read with plain coalesced 128-bit loads (8 bf16 / thread): the Fw handshake
// (k_pub release-store + fence, k_spin acquire-load + __threadfence_system)
// already establishes that the peer's k_put writes are visible before this
// kernel runs, so per-element acquire loads are unnecessary and would only
// serialize the bulk read. A scalar 16-bit tail handles the non-vectorized end.
__global__ void k_add(__half* out, const __half* src, const __half* pbase,
                      int maxn, int rank, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Post-commit: g_R == this round, parity g_R&1 (same parity k_put used).
  const __half* p = pbase + (size_t)(g_R[rank] & 1) * maxn;
  int vec = n >> 3;  // number of 8-elem (uint4) groups
  if (i < vec) {
    uint4 w = reinterpret_cast<const uint4*>(p)[i];
    const __half* wh = reinterpret_cast<const __half*>(&w);
    __half* o = out + 8 * i;
    const __half* sh = src + 8 * i;
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __hadd(sh[k], wh[k]);
  } else if (i == vec) {  // scalar tail (<8 elems)
    for (int j = 8 * vec; j < n; j++) out[j] = __hadd(src[j], p[j]);
  }
}
__global__ void k_add_bf16(__hip_bfloat16* out, const __hip_bfloat16* src,
                           const __hip_bfloat16* pbase, int maxn, int rank,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const __hip_bfloat16* p = pbase + (size_t)(g_R[rank] & 1) * maxn;
  int vec = n >> 3;
  if (i < vec) {
    uint4 w = reinterpret_cast<const uint4*>(p)[i];
    const __hip_bfloat16* wh = reinterpret_cast<const __hip_bfloat16*>(&w);
    __hip_bfloat16* o = out + 8 * i;
    const __hip_bfloat16* sh = src + 8 * i;
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = sh[k] + wh[k];
  } else if (i == vec) {
    for (int j = 8 * vec; j < n; j++) out[j] = src[j] + p[j];
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
  k_put<<<blocks, 256, 0, s>>>((const uint16_t*)in, (uint16_t*)C.slots[C.rank],
                               mn, C.rank, n);
  k_pubspin<<<1, 1, 0, s>>>(Fw, C.n_ranks, C.rank);

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
