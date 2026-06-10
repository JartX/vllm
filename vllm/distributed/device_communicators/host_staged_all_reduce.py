# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-staged AllReduce for 2x/4x RX 7900 XTX without P2P (RCCL-free).

TP AllReduce that stages through cross-process pinned host memory (anonymous
memfd). Out-of-place to match vLLM's functional all_reduce op. Beats RCCL on
this no-P2P box in both regimes: decode via a captured-graph path (3 kernels,
double-buffered, ultra-low fixed overhead under graph lockstep) and prefill via
an eager pipelined fused-duplex path (hostar_allreduce_pipe: chunks the payload
and overlaps the GPU->host write of chunk c+1 with the host->GPU read+add of
chunk c, exploiting GPU1's full-duplex PCIe link). fp16/bf16, world_size 2 or 4.

OFF BY DEFAULT. VLLM_HOSTAR=1 enables it. The eager/prefill path additionally
needs VLLM_HOSTAR_EAGER=1 and VLLM_HOSTAR_MAXELEMS sized for the prefill
activation (~33M); it only engages after the first CUDA-graph capture (so the
divergent warmup/profiling runs can't desync the cross-process round counter).

Backing symbols (hostar_init / hostar_allreduce{,_eager,_pipe}) are built into
the _C extension from csrc/rocm/hostar/host_staged_all_reduce.cu; VLLM_HOSTAR_LIB
can point at a standalone libhostar.so instead.
"""

import ctypes

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_BYTES = 1 << 20  # >1MB → fall back to RCCL (crossover ~1-2MB)


class HostStagedAllReduce:
    def __init__(self, rank: int, world_size: int, max_elems: int = 1 << 21):
        import os

        self.disabled = True
        self._armed = False
        self._eager = os.environ.get("VLLM_HOSTAR_EAGER", "0") == "1"
        max_elems = int(os.environ.get("VLLM_HOSTAR_MAXELEMS", str(max_elems)))
        if not (envs.VLLM_HOSTAR and world_size in (2, 4)):
            return
        # Symbols live in the _C extension; dlopen it by its installed path.
        # VLLM_HOSTAR_LIB overrides for a standalone libhostar.so build.
        import vllm._C  # noqa: F401

        lib_path = envs.VLLM_HOSTAR_LIB or vllm._C.__file__
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            logger.warning("hostar symbols not loadable (%s); using RCCL", e)
            return
        self._lib.hostar_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.hostar_allreduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.hostar_allreduce_eager.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.hostar_allreduce_pipe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._eround = 0
        if self._lib.hostar_init(b"/vllm_hostar", rank, max_elems, world_size) != 0:
            logger.warning("hostar_init failed; using RCCL")
            return
        self.max_elems = max_elems
        self.disabled = False

    def should_use(self, inp: torch.Tensor) -> bool:
        # The cross-process handshake needs both TP ranks to issue the exact same
        # sequence of calls (the round counters must stay in lockstep). Captured
        # graphs guarantee that (both ranks replay the same graph). Eager does
        # too once steady-state, but NOT during warmup/profiling, where per-rank
        # call counts drift and the spin waits forever. So: always engage while
        # capturing (decode), and engage eager (prefill) only with VLLM_HOSTAR_EAGER
        # and only after the first capture has happened (warmup is past by then).
        if (
            self.disabled
            or inp.dtype not in (torch.float16, torch.bfloat16)
            or inp.numel() > self.max_elems
        ):
            return False
        if torch.cuda.is_current_stream_capturing():
            self._armed = True  # capture => warmup done; safe to also do eager
            return True
        # Eager path (e.g. prefill): only after the first capture, so the
        # divergent warmup/profiling dummy runs don't desync the round counter.
        return self._eager and self._armed

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # Out-of-place to match vLLM's functional all_reduce custom op: reduce
        # into a fresh tensor (like pynccl's empty_like) and leave inp intact, so
        # graph nodes that still read the pre-reduce input aren't corrupted.
        out = torch.empty_like(inp)
        stream = torch.cuda.current_stream().cuda_stream
        dtype = 1 if inp.dtype == torch.bfloat16 else 0
        if torch.cuda.is_current_stream_capturing():
            self._lib.hostar_allreduce(
                inp.data_ptr(), out.data_ptr(), inp.numel(), dtype, stream
            )
        else:
            # Eager (prefill). Large payloads use the pipelined fused-duplex
            # path (chunked write||read on GPU1's full-duplex link) which beats
            # RCCL; small eager uses the simple CP-wait path.
            self._eround += 1
            if inp.numel() >= 262144:
                self._lib.hostar_allreduce_pipe(
                    inp.data_ptr(),
                    out.data_ptr(),
                    inp.numel(),
                    dtype,
                    self._eround,
                    stream,
                )
            else:
                self._lib.hostar_allreduce_eager(
                    inp.data_ptr(),
                    out.data_ptr(),
                    inp.numel(),
                    dtype,
                    self._eround,
                    stream,
                )
        return out
