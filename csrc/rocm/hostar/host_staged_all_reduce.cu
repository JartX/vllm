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
// STATUS (2026-05-27): correct and working in vLLM, and BEATS RCCL on this box
// in both regimes (Qwen3.6-27B bf16 TP2, no P2P): decode ~+6-9% tok/s (the
// captured-graph path below), and prefill ~+4% TTFT (the eager pipelined path,
// hostar_allreduce_pipe). Still OFF BY DEFAULT (VLLM_HOSTAR=0); the eager/prefill
// path additionally needs VLLM_HOSTAR_EAGER=1 and VLLM_HOSTAR_MAXELEMS sized for
// the prefill activation (~33M). Key design points, each one a measured fix:
//  (1) Back the cross-process buffer with an anonymous memfd, NOT a file-backed
//      /dev/shm mapping (registering the latter corrupts vLLM's CUDA-graph pool).
//  (2) Reduce OUT-OF-PLACE — vLLM's all_reduce op is functional; leave the input
//      intact and write the sum to a fresh output.
//  (3) Decode: 3 kernels (k_put, k_pubspin, k_add), double-buffered by round
//      parity, 128-bit coalesced loads/stores; wins by ultra-low fixed overhead
//      under graph lockstep (small payloads are latency-bound).
//  (4) Prefill: large eager ARs are bottlenecked by GPU1's host link (it sits
//      behind the PCH chipset, ~4x lower bandwidth — same asymmetry as dead
//      P2P). That link is full-duplex UNDER A KERNEL (simultaneous host loads +
//      stores ~1.8x vs serial) though the copy engine serializes it. So
//      hostar_allreduce_pipe chunks the payload and runs a FUSED kernel that
//      writes my chunk c+1 while it reads+adds the peer's chunk c — driving both
//      PCIe directions at once — beating RCCL's transfer at large sizes.
// See project memory for the full investigation.
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
      (size_t)n_ranks * 2 * max_elems * 2 + (2 + 8) * n_ranks * sizeof(int) + 256;

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
    for (int r = 0; r < (2 + 8) * n_ranks; r++) C.F[r] = 0;  // Fw + Fe + Fp
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

// ===== Eager path (prefill): CP-side flag wait (hipStreamWaitValue), no busy-
// wait kernel. Python drives each call so it passes the round explicitly; buffer
// parity = round&1 (same double-buffer rule). Uses a SEPARATE flag array Fe so
// it never collides with the graph path's Fw. =====
__global__ void k_put_e(const uint16_t* src, uint16_t* slotbase, int maxn,
                        int buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t* slot = slotbase + (size_t)buf * maxn;
  int vec = n >> 3;
  if (i < vec)
    reinterpret_cast<uint4*>(slot)[i] = reinterpret_cast<const uint4*>(src)[i];
  else if (i == vec)
    for (int j = 8 * vec; j < n; j++) slot[j] = src[j];
}
__global__ void k_add_e_bf16(__hip_bfloat16* out, const __hip_bfloat16* src,
                             const __hip_bfloat16* pbase, int maxn, int buf,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const __hip_bfloat16* p = pbase + (size_t)buf * maxn;
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
__global__ void k_add_e(__half* out, const __half* src, const __half* pbase,
                        int maxn, int buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const __half* p = pbase + (size_t)buf * maxn;
  int vec = n >> 3;
  if (i < vec) {
    uint4 w = reinterpret_cast<const uint4*>(p)[i];
    const __half* wh = reinterpret_cast<const __half*>(&w);
    __half* o = out + 8 * i;
    const __half* sh = src + 8 * i;
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __hadd(sh[k], wh[k]);
  } else if (i == vec) {
    for (int j = 8 * vec; j < n; j++) out[j] = __hadd(src[j], p[j]);
  }
}
__global__ void k_fence() { if (!threadIdx.x) __threadfence_system(); }

extern "C" void hostar_allreduce_eager(void* in, void* out, int n, int dtype,
                                       int round, void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  int buf = round & 1;
  int blocks = (n + 255) / 256, mn = C.maxn;
  int* Fe = C.F + C.n_ranks;  // eager flags (after Fw)
  k_put_e<<<blocks, 256, 0, s>>>((const uint16_t*)in,
                                 (uint16_t*)C.slots[C.rank], mn, buf, n);
  k_fence<<<1, 1, 0, s>>>();  // flush my slot to host before publishing
  // Publish via the command processor (no CU busy-wait kernel).
  hipStreamWriteValue32(s, &Fe[C.rank], (unsigned)round, 0);
  for (int p = 0; p < C.n_ranks; p++) {
    if (p == C.rank) continue;
    hipStreamWaitValue32(s, &Fe[p], (unsigned)round, hipStreamWaitValueGte,
                         0xffffffffu);
  }
  k_fence<<<1, 1, 0, s>>>();  // make peers' slot data visible before k_add_e
  const void* src = in;
  for (int p = 0; p < C.n_ranks; p++) {
    if (p == C.rank) continue;
    if (dtype == 1)
      k_add_e_bf16<<<blocks, 256, 0, s>>>((__hip_bfloat16*)out,
                                          (const __hip_bfloat16*)src,
                                          (const __hip_bfloat16*)C.slots[p], mn,
                                          buf, n);
    else
      k_add_e<<<blocks, 256, 0, s>>>((__half*)out, (const __half*)src,
                                     (const __half*)C.slots[p], mn, buf, n);
    src = out;
  }
}


// ===== Pipelined eager all-reduce: chunked, fused write+read kernel that drives
// BOTH PCIe directions at once (GPU1's link is full-duplex under a kernel, ~1.8x
// vs serial). Beats RCCL on large payloads. Double-buffered by round parity;
// per-chunk data-ready flags Fp. =====
#define HOSTAR_PIPE_C 8
__global__ void k_fused_bf16(uint16_t* sw, const uint16_t* iw, int nw,
                             __hip_bfloat16* orr, const __hip_bfloat16* ir,
                             const __hip_bfloat16* pr, int nr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int vw = nw >> 3, vr = nr >> 3;
  if (i < vw)
    reinterpret_cast<uint4*>(sw)[i] = reinterpret_cast<const uint4*>(iw)[i];
  else if ((nw & 7) && i == vw)
    for (int j = 8 * vw; j < nw; j++) sw[j] = iw[j];
  if (i < vr) {
    uint4 w = reinterpret_cast<const uint4*>(pr)[i];
    const __hip_bfloat16* wh = reinterpret_cast<const __hip_bfloat16*>(&w);
    __hip_bfloat16* o = orr + 8 * i; const __hip_bfloat16* sh = ir + 8 * i;
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = sh[k] + wh[k];
  } else if ((nr & 7) && i == vr)
    for (int j = 8 * vr; j < nr; j++) orr[j] = ir[j] + pr[j];
}
__global__ void k_fused_f16(uint16_t* sw, const uint16_t* iw, int nw,
                            __half* orr, const __half* ir, const __half* pr,
                            int nr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int vw = nw >> 3, vr = nr >> 3;
  if (i < vw)
    reinterpret_cast<uint4*>(sw)[i] = reinterpret_cast<const uint4*>(iw)[i];
  else if ((nw & 7) && i == vw)
    for (int j = 8 * vw; j < nw; j++) sw[j] = iw[j];
  if (i < vr) {
    uint4 w = reinterpret_cast<const uint4*>(pr)[i];
    const __half* wh = reinterpret_cast<const __half*>(&w);
    __half* o = orr + 8 * i; const __half* sh = ir + 8 * i;
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __hadd(sh[k], wh[k]);
  } else if ((nr & 7) && i == vr)
    for (int j = 8 * vr; j < nr; j++) orr[j] = __hadd(ir[j], pr[j]);
}
// plain chunk copy / add (prologue write, final read)
__global__ void k_putc(const uint16_t* in, uint16_t* slot, int nw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; int vw = nw >> 3;
  if (i < vw) reinterpret_cast<uint4*>(slot)[i] = reinterpret_cast<const uint4*>(in)[i];
  else if ((nw & 7) && i == vw) for (int j = 8 * vw; j < nw; j++) slot[j] = in[j];
}
__global__ void k_addc_bf16(__hip_bfloat16* o, const __hip_bfloat16* ir,
                            const __hip_bfloat16* pr, int nr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; int vr = nr >> 3;
  if (i < vr) { uint4 w = reinterpret_cast<const uint4*>(pr)[i];
    const __hip_bfloat16* wh = reinterpret_cast<const __hip_bfloat16*>(&w);
    __hip_bfloat16* oo=o+8*i; const __hip_bfloat16* sh=ir+8*i;
#pragma unroll
    for(int k=0;k<8;k++) oo[k]=sh[k]+wh[k]; }
  else if ((nr & 7) && i == vr) for (int j=8*vr;j<nr;j++) o[j]=ir[j]+pr[j];
}
__global__ void k_addc_f16(__half* o, const __half* ir, const __half* pr, int nr){
  int i = blockIdx.x * blockDim.x + threadIdx.x; int vr = nr >> 3;
  if (i < vr) { uint4 w = reinterpret_cast<const uint4*>(pr)[i];
    const __half* wh = reinterpret_cast<const __half*>(&w);
    __half* oo=o+8*i; const __half* sh=ir+8*i;
#pragma unroll
    for(int k=0;k<8;k++) oo[k]=__hadd(sh[k],wh[k]); }
  else if ((nr & 7) && i == vr) for (int j=8*vr;j<nr;j++) o[j]=__hadd(ir[j],pr[j]);
}
__global__ void k_pubc(int* f, int c, int v) {
  if (!threadIdx.x) { __threadfence_system();
    __hip_atomic_store(&f[c], v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
}
__global__ void k_spinc(const int* f, int c, int v) {
  if (!threadIdx.x) {
    while (__hip_atomic_load(&f[c], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v) {}
    __threadfence_system();
  }
}
extern "C" void hostar_allreduce_pipe(void* in, void* out, int n, int dtype,
                                      int round, void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  const int NC = HOSTAR_PIPE_C;
  int buf = round & 1;
  int cs = (n / NC) & ~7;          // chunk size, multiple of 8
  if (cs == 0) cs = (n + 7) & ~7;  // tiny n: single chunk
  int blk = (cs + 255) / 256;
  __half* myslot = C.slots[C.rank] + (size_t)buf * C.maxn;
  int peer = 1 - C.rank;           // 2-rank pipelined path
  __half* peerslot = C.slots[peer] + (size_t)buf * C.maxn;
  int* Fp = C.F + 2 * C.n_ranks;   // pipe flags (after Fw, Fe)
  int* myFp = Fp + C.rank * NC;
  int* peerFp = Fp + peer * NC;
  auto cnt = [&](int c) { int st = c * cs; int e = (c == NC - 1) ? n : st + cs;
                          return e > st ? e - st : 0; };
  // prologue: write chunk 0
  k_putc<<<blk, 256, 0, s>>>((const uint16_t*)in, (uint16_t*)myslot, cnt(0));
  k_pubc<<<1, 1, 0, s>>>(myFp, 0, round);
  for (int c = 0; c < NC; c++) {
    if (cnt(c) == 0) break;
    k_spinc<<<1, 1, 0, s>>>(peerFp, c, round);
    int rc = cnt(c); int off = c * cs;
    if (c + 1 < NC && cnt(c + 1) > 0) {
      int wc = cnt(c + 1); int woff = (c + 1) * cs;
      if (dtype == 1)
        k_fused_bf16<<<blk, 256, 0, s>>>(
            (uint16_t*)(myslot + woff), (const uint16_t*)((__hip_bfloat16*)in + woff), wc,
            (__hip_bfloat16*)out + off, (const __hip_bfloat16*)in + off,
            (const __hip_bfloat16*)(peerslot + off), rc);
      else
        k_fused_f16<<<blk, 256, 0, s>>>(
            (uint16_t*)(myslot + woff), (const uint16_t*)((__half*)in + woff), wc,
            (__half*)out + off, (const __half*)in + off,
            (const __half*)(peerslot + off), rc);
      k_pubc<<<1, 1, 0, s>>>(myFp, c + 1, round);
    } else {
      if (dtype == 1)
        k_addc_bf16<<<blk, 256, 0, s>>>((__hip_bfloat16*)out + off,
            (const __hip_bfloat16*)in + off, (const __hip_bfloat16*)(peerslot + off), rc);
      else
        k_addc_f16<<<blk, 256, 0, s>>>((__half*)out + off,
            (const __half*)in + off, (const __half*)(peerslot + off), rc);
    }
  }
}

#endif  // USE_ROCM
