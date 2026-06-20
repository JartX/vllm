# RDNA3 INT8 + hybrid-GDN GPU page fault — investigation & status

Status: **crash fixed (contained); upstream root cause of the metadata
inconsistency still open.** This file is the hand-off so the work can continue
in a new session.

## Symptom

Production + reproduced:

```
Memory access fault by GPU node-N ... on address 0x...  Reason: Page not present
... Worker proc VllmWorker-* died unexpectedly, shutting down executor.
... step_with_batch_queue -> future.result() -> mq.dequeue -> RuntimeError: cancelled
... EngineDeadError
```

- Model: Qwen3.6-27B GPTQ-W4A16-G128 (hybrid GDN linear-attn + full-attn int8 layers).
- Config that crashes: `--kv-cache-dtype int8_per_token_head`, `--enable-prefix-caching`,
  TRITON_ATTN, TP2 **and** TP4, RDNA3 (gfx1100). `mamba_cache_mode=align`
  (auto-selected when prefix caching is on). async scheduling on
  (`step_with_batch_queue`).
- Pure decode (`num_scheduled_tokens=1`), low KV usage (4–8 %, so **not** OOM).
- `fp16` KV never crashes — int8 only enables the long-context regime that triggers it.

## How to reproduce (reliable, ~20–25 requests)

`tools/rdna3/repro_int8_gdn_pagefault.py`. The trigger is a decode batch of
**mixed / different context lengths**, NOT a uniform one and NOT a single huge
context. A uniform identical prefix resumed 50+ times does **not** crash — the
block-table layout is identical every step. Sweeping many prefix lengths makes
the scheduler co-batch heterogeneous decodes, which is what faults.

```bash
# inside the serving container, against the vLLM port (not a proxy):
python tools/rdna3/repro_int8_gdn_pagefault.py 800 4   # duration_s, concurrency
# watch /health from inside the container + the engine log for the fault.
```

Crash dump from a repro run: 4 reqs co-decoding at
`num_computed_tokens=[7862, 15515, 23459, 31547]`, `num_common_prefix_blocks=[0,0,0,5]`.

## How the faulting kernel was named (the decisive step)

`VLLM_TRACE_FUNCTION=1` does NOT help (it stops tracing after init; steady-state
inference runs via cudagraph / C++ dispatch). A hard GPU page fault kills the
process without a catchable Python traceback. What works:

```bash
export AMD_LOG_LEVEL=3 HIP_LAUNCH_BLOCKING=1   # then launch + run the repro
# after the fault, in the engine log, find "Memory access fault" / "Memory Fault Error"
# and read the dispatches just before it. The last GPU dispatch == the culprit.
```

Result — the last dispatch before the fault was a torch
`at::native::direct_copy_kernel_cuda` (grid {1,1,1} block {128,1,1}), preceded by:
`launch_clamp_scalar(int)` -> `arange_cuda_out` -> `CUDAFunctor_add<int>` ->
`_cuda_scatter_gather_internal_kernel` (a gather) -> two 16-byte H2D copies
(= the 4-request batch). **Not a custom GDN/int8 compute kernel — torch metadata
ops.** That op sequence is exactly `mamba_get_block_table_tensor` align mode.

## Root cause (of the crash) + fix

`vllm/v1/attention/backends/utils.py: mamba_get_block_table_tensor`, align branch:

```python
start_indices = clamp((seq_lens - 1) // block_size, min=0)   # lower bound only!
offsets = arange(1 + num_speculative_blocks)
indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(int64)
return torch.gather(block_table, 1, indices_to_gather)        # OOB if index >= size(1)
```

Fix (commit on branch `perf/rdna3_full_stack`): clamp the gather index to the
valid column range before the gather:

```python
indices_to_gather = indices_to_gather.clamp_(max=block_table.size(1) - 1)
```

Validated: unpatched crashes at ok≈20 every run; patched survives a full pass
(48/48 requests, 0 errors, health 200 throughout).

## OPEN: the real upstream root cause (next session starts here)

The clamp **contains** the symptom but is almost certainly defensive, not causal:

- For the observed crash batch, the longest req was 31547 tokens, `mamba_block_size=16`
  -> `start_index = 31547//16 = 1971`, and `block_table.size(1) = cdiv(31547,16) = 1972`.
  So `start_index (1971) < size(1) (1972)` **by construction**, with `offset=0`
  (no spec decode). Mathematically it should NOT overflow.
- Therefore the overflow means `seq_lens` and `block_table` were **inconsistent**
  for that step — a `seq_len` larger than the blocks actually allocated to that
  request. Suspected origin: the `align` + prefix-cache + heterogeneous-batch +
  async-scheduling path (a resumed prefix-cache request whose full `seq_len`
  doesn't match the step's `block_table` width).

### Next step to pin it

Add a one-shot log when the clamp actually fires, then run the repro once:

```python
# in mamba_get_block_table_tensor, before the clamp:
_w = block_table.size(1)
_bad = (indices_to_gather >= _w)
if bool(_bad.any()):   # NOTE: host sync -> only for an eager / diagnostic run, NOT cudagraph
    logger.warning("mamba align gather OOB: max_idx=%d width=%d seq_lens=%s",
                   int(indices_to_gather.max()), _w, seq_lens.tolist())
```

`bool(.any())` forces a device->host sync that is **illegal during cudagraph
capture** (we hit exactly this earlier: `HIP error: operation not permitted when
stream is capturing`). So run the diagnostic with `--enforce-eager` (or gate the
log so it never runs under capture). The logged `seq_lens` vs `width` mismatch
points at the scheduler/metadata builder producing the inconsistency
(candidates: `gpu_model_runner` seq_lens for resumed prefix-cache reqs;
`gdn_attn.py` / `utils.py` block-table construction in align mode).

## Dead ends (do NOT redo these — all verified wrong)

The fault is in torch metadata ops, NOT in any custom compute kernel. These
were tried and all FAILED (crashed at the same point) before AMD_LOG named the
real kernel:

- Upper-bound guards in the FLA recurrent kernels
  (`fused_recurrent.py`: `fused_recurrent_gated_delta_rule_packed_decode_kernel`
  and `_fwd_kernel`, `state_idx < num_state_slots`). Wrong place.
- Source remap of `non_spec_state_indices_tensor` in
  `qwen_gdn_linear_attn.py::_forward_core_decode_non_spec` (confirmed on the
  active decode path — `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` defaults True —
  yet still crashed). Wrong tensor.
- Gating the GDN cudagraph metadata H2D copies to blocking on ROCm
  (`gdn_attn.py`, commit 9b3e844b2) and the residual sampling/num_computed_tokens
  copies (c29059f57): harmless, but NOT this fix.

## Useful facts / environment

- Test box container: `vllm-vllm1-1`. vLLM loads from
  `/usr/local/lib/python3.12/dist-packages/vllm/...` — `docker cp` a patched
  `.py` there and relaunch (Triton kernels JIT-recompile; no C build needed).
- Launch script in container: `/root/launch_exp27.py` (prod-exact argv).
- From the host you can only reach the server via nginx `:80`; the container
  port `:8000` is NOT published — health-check from *inside* the container.
- Cold start is fast when the compile cache is warm (~30 s), ~20 min cold.
- `--enforce-eager` is required for any diagnostic that does a host sync
  (`.item()`/`.any()`) inside the model forward, because the GDN/attention path
  runs eager-but-cudagraph-adjacent.
