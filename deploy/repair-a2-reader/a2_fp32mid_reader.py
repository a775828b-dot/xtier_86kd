"""Isolated A2-enabled SM120 DSV4.1 mixed-KV reader with FP32 split-K partials.

Loaded externally via PYTHONPATH sitecustomize without modifying any frozen
source tree (/data/src/sglang-align, /data/src/sglang-dev, /data/xtier-env).
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import torch

_CU_PATH = Path(__file__).resolve().parent / "dsv41_bf16_fp32mid.cu"
_MODULE_NAME = "sparse_mla_sm120_dsv41_bf16_fp32mid_di_a21_v2"
_LN2 = math.log(2)
_DEFAULT_SM_SCALE = 512**-0.5

_plan_cache: dict[tuple, tuple] = {}
_arenas: dict[int, list[torch.Tensor | None]] = {}


@lru_cache(maxsize=1)
def _get_fp32mid_module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    spec = gen_jit_spec(
        _MODULE_NAME,
        [_CU_PATH],
        extra_cuda_cflags=list(nvcc_flags) + [f"-I{jit_env.FLASHINFER_CSRC_DIR}"],
    )
    return spec.build_and_load()


def _arena_slice(
    dev_idx: int, slot: int, shape: tuple[int, ...], alloc_fresh: bool, device: torch.device
) -> torch.Tensor:
    if alloc_fresh:
        return torch.empty(shape, dtype=torch.float32, device=device)
    pair = _arenas.get(dev_idx)
    if pair is None:
        pair = [None, None]
        _arenas[dev_idx] = pair
    numel = math.prod(shape)
    buf = pair[slot]
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.float32, device=device)
        pair[slot] = buf
    return buf[:numel].view(shape)


def install_onto(dsv41_mod) -> None:
    orig_forward = dsv41_mod.forward
    check_capabilities = dsv41_mod.check_capabilities
    _cache_view = dsv41_mod._cache_view
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers.attention.dsv4 import di_align

    def forward(
        *,
        q,
        k_cache,
        indices,
        head_dim_v,
        kv_layout,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_kv_layout=None,
        topk_length=None,
        extra_topk_length=None,
        attn_sink=None,
        softmax_scale=None,
        **kwargs,
    ):
        if not di_align.enabled("A2") or extra_k_cache is None:
            return orig_forward(
                q=q,
                k_cache=k_cache,
                indices=indices,
                head_dim_v=head_dim_v,
                kv_layout=kv_layout,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_kv_layout=extra_kv_layout,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        extra_layout_parsed = KVLayout.parse(extra_kv_layout) if extra_kv_layout is not None else None
        if extra_layout_parsed is not KVLayout.V41_FP4_NEW:
            return orig_forward(
                q=q,
                k_cache=k_cache,
                indices=indices,
                head_dim_v=head_dim_v,
                kv_layout=kv_layout,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_kv_layout=extra_kv_layout,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        check_capabilities()
        from flashinfer.mla._sparse_mla_sm120 import (
            _DECODE_MAX_TOKENS,
            _calibration as calibration,
            _prepared as prepared,
        )
        from flashinfer.mla._sparse_mla_sm120._policy import _MODEL_TYPE_DSV4_1_INLINE

        if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512 or head_dim_v != 512:
            return orig_forward(
                q=q,
                k_cache=k_cache,
                indices=indices,
                head_dim_v=head_dim_v,
                kv_layout=kv_layout,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_kv_layout=extra_kv_layout,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        rows, _, heads, _ = q.shape
        if rows == 0 or rows > _DECODE_MAX_TOKENS:
            return orig_forward(
                q=q,
                k_cache=k_cache,
                indices=indices,
                head_dim_v=head_dim_v,
                kv_layout=kv_layout,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_kv_layout=extra_kv_layout,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        parsed_main = KVLayout.parse(kv_layout)
        if parsed_main is not KVLayout.V41_NEW or q.dtype is not torch.bfloat16 or extra_indices_in_kvcache is None:
            return orig_forward(
                q=q,
                k_cache=k_cache,
                indices=indices,
                head_dim_v=head_dim_v,
                kv_layout=kv_layout,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_kv_layout=extra_kv_layout,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                **kwargs,
            )

        kv = _cache_view(k_cache, parsed_main)
        extra_kv = _cache_view(extra_k_cache, extra_layout_parsed)
        idx = indices.reshape(rows, -1)
        extra_idx = extra_indices_in_kvcache.reshape(rows, -1)
        q_3d = q.squeeze(1)
        if not q_3d.is_contiguous():
            q_3d = q_3d.contiguous()

        out = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
        dev_idx = int(q.device.index)

        key = (
            dev_idx,
            rows,
            heads,
            tuple(kv.shape),
            tuple(kv.stride()),
            tuple(idx.shape),
            tuple(idx.stride()),
            tuple(extra_kv.shape),
            tuple(extra_kv.stride()),
            tuple(extra_idx.shape),
            tuple(extra_idx.stride()),
            topk_length is not None,
            extra_topk_length is not None,
            attn_sink is not None,
        )
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            calibration.refresh_store()
        entry = _plan_cache.get(key)
        if capturing:
            if entry is None:
                raise ValueError("warm up this attention shape before CUDA graph capture")
        elif (
            entry is None
            or entry[0] != calibration._constants_version
            or prepared.tuning_enabled(False)
        ):
            tensors = (
                q_3d,
                kv,
                idx,
                out,
                topk_length,
                attn_sink,
                extra_kv,
                extra_idx,
                extra_topk_length,
                lse,
                None,
                None,
            )
            m = prepared.validate_metadata(
                *tensors,
                model=_MODEL_TYPE_DSV4_1_INLINE,
                is_dsv4_nvfp4=False,
                extra_fp4=True,
                value_dim=512,
            )
            ep = prepared.resolve_execution(m, q.device, "bf16", False, 0)
            facts = dict(ep.inspect())
            spec_heads = heads if heads in (8, 16, 32, 64, 128) else 0
            variant = int(facts["variant"])
            if variant == 0:
                mod = _get_fp32mid_module()
                mod.init_dsv41_bf16_fp32mid_smem(
                    dev_idx,
                    int(m.model),
                    int(spec_heads),
                    True,
                    int(facts["shared_bytes"]),
                )
                fn = mod.execute_dsv41_bf16_fp32mid
            else:
                fn = prepared._execute_fn(False)
            main_stride = int(m.row_stride_bytes) if int(m.model) == _MODEL_TYPE_DSV4_1_INLINE else int(m.page_stride_bytes)
            extra_stride = int(m.extra_row_stride_bytes) if int(m.model) == _MODEL_TYPE_DSV4_1_INLINE else int(m.extra_page_stride_bytes)
            entry = (
                calibration._constants_version,
                variant,
                ep,
                fn,
                int(m.model),
                int(spec_heads),
                int(facts["active_splits"]),
                int(facts["scratch_heads"]),
                int(facts["cpb"]),
                int(facts["shared_bytes"]),
                int(m.page_size),
                int(m.extra_page_size),
                main_stride,
                extra_stride,
            )
            _plan_cache[key] = entry

        (
            _,
            variant,
            ep,
            fn,
            model_id,
            spec_heads,
            active_splits,
            scratch_heads,
            cpb,
            smem_bytes,
            page_size,
            extra_page_size,
            main_stride,
            extra_stride,
        ) = entry

        sm_scale_val = float(softmax_scale if softmax_scale is not None else _DEFAULT_SM_SCALE)
        if variant != 0:
            # Direct single-CTA PREFILL_SG execution (zero scratch, single lookup).
            fn(
                ep,
                q_3d,
                kv,
                idx,
                out,
                out.view(torch.float32),
                out,
                lse,
                sm_scale_val,
                topk_length,
                attn_sink,
                extra_kv,
                extra_idx,
                extra_topk_length,
                _LN2,
            )
            return out.unsqueeze(1), lse.unsqueeze(-1)

        on_default_stream = (
            not capturing
            and torch.cuda.current_stream(q.device).cuda_stream
            == torch.cuda.default_stream(q.device).cuda_stream
        )
        mid_out_f32 = _arena_slice(
            dev_idx, 0, (rows, scratch_heads, active_splits, 512), not on_default_stream, q.device
        )
        mid_lse = _arena_slice(
            dev_idx, 1, (rows, scratch_heads, active_splits), not on_default_stream, q.device
        )

        fn(
            q_3d,
            kv,
            idx,
            mid_out_f32,
            mid_lse,
            out,
            lse,
            topk_length,
            attn_sink,
            extra_kv,
            extra_idx,
            extra_topk_length,
            model_id,
            spec_heads,
            active_splits,
            cpb,
            smem_bytes,
            page_size,
            extra_page_size,
            main_stride,
            extra_stride,
            True,
            sm_scale_val,
            _LN2,
        )
        return out.unsqueeze(1), lse.unsqueeze(-1)

    dsv41_mod.forward = forward
    dsv41_mod._a2_fp32mid_installed = True
