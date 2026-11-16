"""External non-invasive Step 1 + Step 2 Prefill & MoE optimization hooks for TF500 TP8 SM120.

Zero modifications to /data/src/sglang-align. Installed lazily via MetaPathFinder in sitecustomize.py:
- sglang.srt.distributed.device_communicators.pynccl -> install_onto_pynccl (skip 0-size roots on self.comm)
- sglang.srt.layers.dp_attention -> install_onto_dp_attention (CUDA-Graph-safe persistent FP8 gatherv buffers)
- humming.layer -> install_onto_humming_layer (target/draft MoE tight padding)
- sglang.srt.layers.quantization.humming_utils -> install_onto_humming_utils (target MoE MXMMA float8e3m4)
- sglang.srt.managers.scheduler -> install_onto_scheduler (idle single-request chunk boost)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_PAD_OVERRIDE: tuple[int | None, int | None] | None = None
_LAST_GLOBAL_ACTIVE_RANKS: int = 8
_IN_REPLAY_TAIL_DEPTH: int = 0


def install_onto_replay_tail(rt_mod) -> None:
    cls = getattr(rt_mod, "DecoderReplayTail", None)
    if cls is None or getattr(cls, "_xtier_tail_depth_patched", False):
        return
    import functools

    def _wrap_tail(fn):
        @functools.wraps(fn)
        def _inner(*args, **kwargs):
            global _IN_REPLAY_TAIL_DEPTH
            _IN_REPLAY_TAIL_DEPTH += 1
            try:
                return fn(*args, **kwargs)
            finally:
                _IN_REPLAY_TAIL_DEPTH -= 1
        return _inner

    for mname in ("run", "run_idle", "_run_layers_eagerly"):
        if hasattr(cls, mname):
            setattr(cls, mname, _wrap_tail(getattr(cls, mname)))
    cls._xtier_tail_depth_patched = True


def install_onto_pynccl(pynccl_mod) -> None:
    if os.environ.get("XTIER_PREFILL_NCCL_SKIP_ZERO", "1") != "1":
        return
    from torch.distributed import ReduceOp
    from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
        buffer_type,
        cudaStream_t,
        ncclDataTypeEnum,
        ncclRedOpTypeEnum,
    )

    PyNcclCommunicator = pynccl_mod.PyNcclCommunicator
    if getattr(PyNcclCommunicator, "_xtier_skip_zero_patched", False):
        return

    orig_all_gather = PyNcclCommunicator.all_gather
    orig_reduce_scatter = PyNcclCommunicator.reduce_scatter

    def _patched_all_gather(self, output_tensor, input_tensor, sizes=None):
        if sizes is None:
            return orig_all_gather(self, output_tensor, input_tensor, sizes=None)
        if self.disabled:
            return
        assert input_tensor.device == self.device
        stream = self._resolve_stream()
        split_offset = 0
        self.nccl.ncclGroupStart()
        try:
            for root, split_size in enumerate(sizes):
                if split_size == 0:
                    continue
                dst_slice = output_tensor[split_offset : split_offset + split_size]
                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
        finally:
            self.nccl.ncclGroupEnd()

    def _patched_reduce_scatter(self, output_tensor, input_tensor, op=ReduceOp.SUM, sizes=None):
        if sizes is None:
            return orig_reduce_scatter(self, output_tensor, input_tensor, op=op, sizes=None)
        if self.disabled:
            return
        assert input_tensor.device == self.device
        stream = self._resolve_stream()
        split_offset = 0
        self.nccl.ncclGroupStart()
        try:
            for root, split_size in enumerate(sizes):
                if split_size == 0:
                    continue
                chunk = input_tensor[split_offset : split_offset + split_size, ...]
                self.nccl.ncclReduce(
                    buffer_type(chunk.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    chunk.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    ncclRedOpTypeEnum.from_torch(op),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
        finally:
            self.nccl.ncclGroupEnd()

    PyNcclCommunicator.all_gather = _patched_all_gather
    PyNcclCommunicator.reduce_scatter = _patched_reduce_scatter
    PyNcclCommunicator._xtier_skip_zero_patched = True


def install_onto_dp_attention(dp_mod) -> None:
    if getattr(dp_mod, "_xtier_dp_fp8_safe_patched", False):
        return
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    prealloc_rows = int(os.environ.get("XTIER_DP_FP8_PREALLOC_ROWS", "65536") or "65536")
    _persistent_pool: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    _retired_keepalive: list[tuple[torch.Tensor, torch.Tensor]] = []
    _zero_q_s: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _safe_get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        key = (dev_idx, hidden)
        ngroups = hidden // dp_mod._DP_GATHER_FP8_GROUP
        bufs = _persistent_pool.get(key)
        if bufs is not None and bufs[0].shape[0] >= rows:
            return bufs[0][:rows], bufs[1][:rows]
        if torch.cuda.is_current_stream_capturing():
            return (
                torch.empty((rows, hidden), dtype=torch.uint8, device=device),
                torch.empty((rows, ngroups), dtype=torch.float32, device=device),
            )
        alloc_rows = max(rows, prealloc_rows)
        if bufs is not None:
            _retired_keepalive.append(bufs)
        new_bufs = (
            torch.empty((alloc_rows, hidden), dtype=torch.uint8, device=device),
            torch.empty((alloc_rows, ngroups), dtype=torch.float32, device=device),
        )
        _persistent_pool[key] = new_bufs
        return new_bufs[0][:rows], new_bufs[1][:rows]

    orig_dp_gather = dp_mod._dp_gather
    def _tracked_dp_gather(global_tokens, local_tokens, forward_batch, is_partial, cp_shard_counts=None):
        global _LAST_GLOBAL_ACTIVE_RANKS
        if _IN_REPLAY_TAIL_DEPTH == 0 and not torch.cuda.is_current_stream_capturing():
            sz = getattr(forward_batch, 'global_num_tokens_cpu', None)
            if sz is not None:
                _LAST_GLOBAL_ACTIVE_RANKS = sum(1 for x in sz if x > 0)
        return orig_dp_gather(global_tokens, local_tokens, forward_batch, is_partial, cp_shard_counts=cp_shard_counts)
    dp_mod._dp_gather = _tracked_dp_gather

    def _safe_dp_gather_via_all_gatherv_fp8(global_tokens: torch.Tensor, local_real: torch.Tensor, sizes):
        global _LAST_GLOBAL_ACTIVE_RANKS
        if _IN_REPLAY_TAIL_DEPTH == 0 and not torch.cuda.is_current_stream_capturing() and sizes is not None:
            _LAST_GLOBAL_ACTIVE_RANKS = sum(1 for x in sizes if x > 0)
        rows = global_tokens.shape[0]
        hidden = global_tokens.shape[-1]
        dev = global_tokens.device
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        ngroups = hidden // dp_mod._DP_GATHER_FP8_GROUP
        if local_real.shape[0] == 0 and not torch.cuda.is_current_stream_capturing():
            zkey = (dev_idx, hidden)
            z = _zero_q_s.get(zkey)
            if z is None:
                z = (
                    torch.empty((1, hidden), dtype=torch.float8_e4m3fn, device=dev),
                    torch.empty((1, ngroups), dtype=torch.float32, device=dev),
                )
                _zero_q_s[zkey] = z
            q, s = z[0][:0], z[1][:0]
        else:
            q, s = sglang_per_token_group_quant_fp8(local_real.contiguous(), dp_mod._DP_GATHER_FP8_GROUP)
        gq, gs = _safe_get_dp_gather_fp8_bufs(rows, hidden, dev)
        tp_group = dp_mod.get_parallel().tp_group
        tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
        tp_group.all_gatherv(s, sizes=sizes, output=gs)
        dp_mod._dequant_per_token_group_fp8_kernel[(rows,)](
            gq.view(torch.float8_e4m3fn),
            gs,
            global_tokens,
            HIDDEN=hidden,
            NGROUPS=ngroups,
            GROUP=dp_mod._DP_GATHER_FP8_GROUP,
            BLOCK=2048,
        )

    dp_mod._get_dp_gather_fp8_bufs = _safe_get_dp_gather_fp8_bufs
    dp_mod._dp_gather_via_all_gatherv_fp8 = _safe_dp_gather_via_all_gatherv_fp8
    dp_mod._xtier_dp_fp8_safe_patched = True


def install_onto_humming_layer(humming_layer_mod) -> None:
    HummingMethod = humming_layer_mod.HummingLayerMethod
    if getattr(HummingMethod, "_xtier_pad_patched", False):
        return
    orig_prepare = HummingMethod.prepare_layer_meta

    @classmethod
    def _patched_prepare(cls, *args, **kwargs):
        if _PAD_OVERRIDE is not None:
            pn, pk = _PAD_OVERRIDE
            if pn is not None:
                kwargs["pad_n_to_multiple"] = pn
            if pk is not None:
                kwargs["pad_k_to_multiple"] = pk
        return orig_prepare(*args, **kwargs)

    HummingMethod.prepare_layer_meta = _patched_prepare
    HummingMethod._xtier_pad_patched = True


def install_onto_humming_utils(hu_mod) -> None:
    if getattr(hu_mod, "_xtier_hu_patched", False):
        return

    mxmma_mode = os.environ.get("XTIER_PREFILL_MOE_MXMMA", "").strip().lower()
    draft_mxmma_mode = os.environ.get("XTIER_DRAFT_MOE_MXMMA", mxmma_mode).strip().lower()
    pad_n_s = os.environ.get("XTIER_MOE_PAD_N", "").strip()
    pad_k_s = os.environ.get("XTIER_MOE_PAD_K", "").strip()
    pad_n = int(pad_n_s) if pad_n_s else None
    pad_k = int(pad_k_s) if pad_k_s else None

    draft_pad_n_s = os.environ.get("XTIER_DRAFT_MOE_PAD_N", "").strip()
    draft_pad_k_s = os.environ.get("XTIER_DRAFT_MOE_PAD_K", "").strip()
    draft_pad_n = int(draft_pad_n_s) if draft_pad_n_s else None
    draft_pad_k = int(draft_pad_k_s) if draft_pad_k_s else None

    valid_mxmma = ("e3m4", "float8e3m4", "e4m3", "float8e4m3")
    if (
        mxmma_mode not in valid_mxmma
        and draft_mxmma_mode not in valid_mxmma
        and pad_n is None
        and pad_k is None
        and draft_pad_n is None
        and draft_pad_k is None
    ):
        hu_mod._xtier_hu_patched = True
        return

    orig_sublayer = hu_mod._prepare_humming_moe_sublayer
    HummingInputSchema = hu_mod.HummingInputSchema

    def _patched_sublayer(
        layer,
        sublayer_name,
        shape_n,
        shape_k,
        weight_schema,
        input_schema,
        use_deepep_fp8_dispatch,
    ):
        global _PAD_OVERRIDE
        num_experts = getattr(layer, "num_local_experts", 0)
        is_target_moe = num_experts == 384
        is_draft_moe = num_experts == 128
        active_mxmma = mxmma_mode if is_target_moe else (draft_mxmma_mode if is_draft_moe else "")
        if active_mxmma in valid_mxmma:
            a_dt = "float8e3m4" if active_mxmma in ("e3m4", "float8e3m4") else "float8e4m3"
            input_schema = HummingInputSchema.from_config({
                "a_dtype": a_dt,
                "input_scale_group_size": 32,
                "input_scale_dtype": "float8e8m0",
                "input_quant_mode": "dynamic_group",
            })
        prev_pad = _PAD_OVERRIDE
        if is_target_moe and (pad_n is not None or pad_k is not None):
            _PAD_OVERRIDE = (pad_n, pad_k)
        elif is_draft_moe and (draft_pad_n is not None or draft_pad_k is not None):
            _PAD_OVERRIDE = (draft_pad_n, draft_pad_k)
        else:
            _PAD_OVERRIDE = None
        try:
            return orig_sublayer(
                layer,
                sublayer_name,
                shape_n,
                shape_k,
                weight_schema,
                input_schema,
                use_deepep_fp8_dispatch,
            )
        finally:
            _PAD_OVERRIDE = prev_pad

    hu_mod._prepare_humming_moe_sublayer = _patched_sublayer
    hu_mod._xtier_hu_patched = True


def install_onto_scheduler(scheduler_mod) -> None:
    idle_single_chunk = int(os.environ.get("XTIER_IDLE_SINGLE_CHUNK", "0") or "0")
    if idle_single_chunk <= 0:
        return
    Scheduler = scheduler_mod.Scheduler
    if getattr(Scheduler, "_xtier_idle_chunk_patched", False):
        return

    orig_get_adder = Scheduler.get_prefill_adder_class
    subcls_cache: dict[tuple[type, int], type] = {}

    def _patched_get_adder(self):
        adder_cls = orig_get_adder(self)
        has_chunked = 1 if getattr(self, "chunked_req", None) is not None else 0
        key = (adder_cls, has_chunked)
        sub = subcls_cache.get(key)
        if sub is None:
            class _IdleBoostPrefillAdder(adder_cls):
                def __init__(self_adder, *args, **kwargs):
                    args_l = list(args)
                    if len(args_l) >= 7:
                        running_batch = args_l[3]
                        max_prefill = args_l[5]
                        chunk_sz = args_l[6]
                        wq_len = kwargs.get("waiting_queue_len", 0)
                        if (
                            chunk_sz is not None
                            and (running_batch is None or running_batch.is_empty())
                            and has_chunked == 1
                            and wq_len == 0
                            and _LAST_GLOBAL_ACTIVE_RANKS == 1
                        ):
                            args_l[6] = min(max_prefill, max(chunk_sz, idle_single_chunk))
                    super().__init__(*args_l, **kwargs)

            sub = _IdleBoostPrefillAdder
            subcls_cache[key] = sub
        return sub

    Scheduler.get_prefill_adder_class = _patched_get_adder
    Scheduler._xtier_idle_chunk_patched = True
