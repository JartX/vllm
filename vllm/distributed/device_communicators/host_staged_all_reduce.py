# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-staged AllReduce for 2x/4x RX 7900 XTX without P2P (RCCL-free).

Small-payload TP AllReduce that stages through cross-process pinned host memory
(anonymous memfd) and busy-polls a flag; graph-capturable. Correct and coherent
in vLLM, but OFF BY DEFAULT (VLLM_HOSTAR=1 to enable) because it does not beat
RCCL here — AllReduce is not on the decode critical path on this box, so it
measures roughly equal at low concurrency and a few % slower at higher
concurrency. Out-of-place (returns a fresh tensor) to match vLLM's functional
all_reduce op. fp16/bf16, world_size 2 or 4.

Backing symbols (hostar_init/hostar_allreduce) are built into the _C extension
from csrc/rocm/hostar/host_staged_all_reduce.cu; VLLM_HOSTAR_LIB can point at a
standalone libhostar.so instead.
"""

import ctypes

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_BYTES = 1 << 20  # >1MB → fall back to RCCL (crossover ~1-2MB)


class HostStagedAllReduce:
    def __init__(self, rank: int, world_size: int, max_elems: int = 1 << 21):
        self.disabled = True
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
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self._lib.hostar_allreduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p
        ]
        if self._lib.hostar_init(b"/vllm_hostar", rank, max_elems,
                                 world_size) != 0:
            logger.warning("hostar_init failed; using RCCL")
            return
        self.max_elems = max_elems
        self.disabled = False

    def should_use(self, inp: torch.Tensor) -> bool:
        # Engage ONLY while a CUDA graph is being captured. The cross-process
        # k_spin handshake needs both TP ranks to issue the exact same sequence
        # of calls (the shared round counter must stay in lockstep). That holds
        # for captured graphs — both ranks replay the same graph each step — but
        # NOT during eager profiling/warmup, where per-rank call counts drift
        # and k_spin then waits forever. So eager all-reduces fall back to RCCL;
        # capture records the hostar kernels and they execute, in lockstep, only
        # on replay. Large batches / prefill aren't captured → RCCL too.
        return (not self.disabled
                and inp.dtype in (torch.float16, torch.bfloat16)
                and inp.numel() <= self.max_elems
                and inp.numel() * 2 <= _MAX_BYTES
                and torch.cuda.is_current_stream_capturing())

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # Out-of-place to match vLLM's functional all_reduce custom op: reduce
        # into a fresh tensor (like pynccl's empty_like) and leave inp intact, so
        # graph nodes that still read the pre-reduce input aren't corrupted.
        out = torch.empty_like(inp)
        stream = torch.cuda.current_stream().cuda_stream
        dtype = 1 if inp.dtype == torch.bfloat16 else 0
        self._lib.hostar_allreduce(inp.data_ptr(), out.data_ptr(), inp.numel(),
                                   dtype, stream)
        return out
