// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-staged AllReduce shared library (no P2P, RCCL-free) for 2× RX 7900 XTX.
// C ABI loadable from Python via ctypes — shared across vLLM TP worker procs.
//
// Shared host region (POSIX shm) layout: [A:max][B:max][flagA:int][flagB:int].
// rank0 uses A/flagA, rank1 uses B/flagB. AR: D2H mine -> sig -> spin peer ->
// H2D peer -> reduce. Mechanism validated at ~43us/10KB cross-process.
//
// build: hipcc -O3 --offload-arch=gfx1100 -fPIC -shared 30_hostar_lib.cpp -o libhostar.so
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

// graph-safe: round counter lives in device mem. atomicAdd advances on each
// replay (no host-supplied v), so a captured graph re-syncs every launch.
__global__ void k_sig(int* mf) {
  if (!threadIdx.x) __hip_atomic_fetch_add(mf, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void k_spin(const int* pf, const int* mf) {
  if (!threadIdx.x) { long g = 0;
    int me = __hip_atomic_load(mf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(pf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < me)
      if (++g > 200000000L) break; }
}
__global__ void k_add(__half* d, const __half* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = __hadd(d[i], p[i]);
}

struct Ctx { int rank, maxn, iter; __half *myh,*ph,*gp; int *mf,*pf; hipStream_t s; };
static Ctx C;

extern "C" int hostar_init(const char* shm, int rank, int max_elems) {
  size_t bytes = 2*(size_t)max_elems*2 + 128;
  int fd = shm_open(shm, O_CREAT|O_RDWR, 0666); if(fd<0) return -1;
  ftruncate(fd, bytes);
  void* p = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (hipHostRegister(p, bytes, hipHostRegisterPortable)!=hipSuccess) return -2;
  __half* A=(__half*)p; __half* B=A+max_elems; int* fA=(int*)(B+max_elems);
  C.rank=rank; C.maxn=max_elems; C.iter=0;
  C.myh = rank?B:A; C.ph = rank?A:B; C.mf = rank?fA+1:fA; C.pf = rank?fA:fA+1;
  if(rank==0){*fA=0; *(fA+1)=0;} usleep(500000);
  hipMalloc(&C.gp, max_elems*2); hipStreamCreate(&C.s);
  return 0;
}
// In-place: g holds my data, gets reduced sum. n<=max_elems. Stream-capturable.
extern "C" void hostar_allreduce(__half* g, int n, void* stream) {
  hipStream_t s = stream ? (hipStream_t)stream : C.s;
  hipMemcpyAsync(C.myh, g, n*2, hipMemcpyDeviceToHost, s);
  k_sig<<<1,1,0,s>>>(C.mf); k_spin<<<1,1,0,s>>>(C.pf, C.mf);
  hipMemcpyAsync(C.gp, C.ph, n*2, hipMemcpyHostToDevice, s);
  k_add<<<(n+255)/256,256,0,s>>>(g, C.gp, n);
}
