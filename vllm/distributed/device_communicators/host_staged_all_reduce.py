# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-staged AllReduce for 2× RX 7900 XTX without P2P (RCCL-free).

Small-payload TP AllReduce path that stages through pinned host memory and
busy-polls a cross-process flag; graph-capturable. Beats RCCL ~9-13× on the
10-512KB regime, but end-to-end ties RCCL because the 587us call overlaps
compute. Off by default — flip on with VLLM_HOSTAR=1 for a workload where
AllReduce is on the critical path. Only ranks 0/1, fp16, world_size 2.

Backing lib: csrc/rocm/hostar/host_staged_all_reduce.cpp →
  hipcc -O3 --offload-arch=gfx1100 -fPIC -shared <src> -o libhostar.so
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
        if not (envs.VLLM_HOSTAR and world_size == 2):
            return
        lib_path = envs.VLLM_HOSTAR_LIB or "libhostar.so"
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            logger.warning("hostar lib not loadable (%s); using RCCL", e)
            return
        self._lib.hostar_allreduce.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        if self._lib.hostar_init(b"/vllm_hostar", rank, max_elems) != 0:
            logger.warning("hostar_init failed; using RCCL")
            return
        self.max_elems = max_elems
        self.disabled = False

    def should_use(self, inp: torch.Tensor) -> bool:
        return (not self.disabled and inp.dtype == torch.float16
                and inp.numel() <= self.max_elems
                and inp.numel() * 2 <= _MAX_BYTES)

    def all_reduce(self, inp: torch.Tensor) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.hostar_allreduce(inp.data_ptr(), inp.numel(), stream)
