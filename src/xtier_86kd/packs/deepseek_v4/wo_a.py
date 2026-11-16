"""DeepSeek V4.1 attention ``wo_a`` (the grouped low-rank output projection) on SM120 TP8:
- Decode / DSpark verify (0 < T <= XTIER_DS_WO_A_MAX_T, default 56 covering DP8 local_bs=8 x 6 verify = 48
  and draft 8 x 5 = 40): Triton grouped FP8 einsum reading checkpoint FP8 + E8M0 scales directly without
  HBM weight dequantization (eliminates 200 _dequant_kernel calls per 5 steps in 64-way Decode).
- Medium Prefill (MAX_T < T < 1024, e.g. DecoderReplayTail T=384): strided token-major torch.bmm
  (0.0 error vs einsum, 1.15x faster, avoids temporary transpose allocation).
- Large Prefill (T >= 1024 and T % 128 == 0 when XTIER_DS_WO_A_PREFILL_MXFP8=1): zero-dequant SM120
  native MXFP8 grouped GEMM using block_scale_interleave on the exact checkpoint E8M0 weight scales
  (zero duplicate weight VRAM) + single-pass swizzled activation quantization (1.45x+ faster).
"""

import logging
import os
import re

import torch
import triton
import triton.language as tl

from xtier_86kd.arch import e4m3_f32

logger = logging.getLogger(__name__)
MODELS = 'sglang.srt.models.deepseek_v4'
APPLY = f'{MODELS}._apply_wo_a_bf16_matmul'
STREAMING = f'{MODELS}._dequant_fp8_wo_a_streaming'
LOAD = f'{MODELS}.DeepseekV4ForCausalLM.load_weights'
SCALE_NAME = re.compile(r'(?:^|\.)layers\.(\d+)\.attn\.wo_a\.scale$')
LAYER_NAME = re.compile(r'(?:^|\.)layers\.(\d+)\.')
BLOCK = 32                        # checkpoint FP8 block (32 rows x 32 cols)
FAST_SHAPE = (2, 1024, 4096)      # SGLang TP4 BF16 fast-path shape
MAX_T = int(os.environ.get('XTIER_DS_WO_A_MAX_T', '56'))
PREFILL_MXFP8 = os.environ.get('XTIER_DS_WO_A_PREFILL_MXFP8', '0') == '1'
PREFILL_MXFP8_MIN_T = int(os.environ.get('XTIER_DS_WO_A_PREFILL_MXFP8_MIN_T', '1024'))
BLOCK_T, BLOCK_R, BLOCK_H = 16, 32, 128
_scales = {}         # FP8 weight data_ptr -> FP32 block scales [rows / 32, cols / 32]
_mxfp8_wscales = {}  # FP8 weight data_ptr -> list of G swizzled UE8M0 weight scale tensors
_scratch = {}        # (device, numel) -> BF16 buffer for dequantized fallback calls
_capture = [None]    # {layer: raw E8M0 scale} while target model loads


def enabled():
    return os.environ.get('XTIER_DS_WO_A', '1') != '0'


@triton.jit
def _einsum_kernel(o_ptr, w_ptr, s_ptr, out_ptr, T, o_st, o_sg, o_sh, w_sg, w_sr, w_sh, s_stride, out_st, out_sg,
                   out_sr, R: tl.constexpr, D: tl.constexpr, FP8: tl.constexpr, SCALE_BN: tl.constexpr,
                   SCALE_BK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[t, g, r] = sum_h o[t, g, h] * w[g, r, h], w = FP8 x E8M0 scale or BF16; FP32 accumulator."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    g = tl.program_id(2)
    t_mask, r_mask = t < T, r < R
    acc = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
    for h0 in range(0, D, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        h_mask = h < D
        a = tl.load(o_ptr + t[:, None] * o_st + g * o_sg + h[None, :] * o_sh,
                    mask=t_mask[:, None] & h_mask[None, :], other=0.0)
        w = tl.load(w_ptr + g * w_sg + r[:, None] * w_sr + h[None, :] * w_sh,
                    mask=r_mask[:, None] & h_mask[None, :], other=0.0)
        if FP8:
            k = h0 // SCALE_BK + tl.arange(0, BLOCK_H // SCALE_BK)
            s = tl.load(s_ptr + ((g * R + r) // SCALE_BN)[:, None] * s_stride + k[None, :],
                        mask=r_mask[:, None] & (k * SCALE_BK < D)[None, :], other=0.0)
            w = tl.reshape(tl.reshape(e4m3_f32(w), (BLOCK_R, BLOCK_H // SCALE_BK, SCALE_BK)) * s[:, :, None],
                           (BLOCK_R, BLOCK_H)).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc=acc)
    tl.store(out_ptr + t[:, None] * out_st + g * out_sg + r[None, :] * out_sr, acc.to(out_ptr.dtype.element_ty),
             mask=t_mask[:, None] & r_mask[None, :])


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, rows, cols, s_stride, SCALE_BN: tl.constexpr, SCALE_BK: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (r < rows)[:, None] & (c < cols)[None, :]
    w = tl.load(w_ptr + r[:, None] * cols + c[None, :], mask=mask, other=0.0)
    s = tl.load(s_ptr + (r // SCALE_BN)[:, None] * s_stride + (c // SCALE_BK)[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + r[:, None] * cols + c[None, :], (e4m3_f32(w) * s).to(tl.bfloat16), mask=mask)


def einsum(o, w, scale=None):
    """o [T, G, D] BF16 x w [G, R, D] (FP8 with FP32 block scales, or BF16) -> [T, G, R] BF16."""
    T, G, D = o.shape
    R = w.shape[1]
    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    fp8 = scale is not None
    s = scale if fp8 else w
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(R, BLOCK_R), G)
    _einsum_kernel[grid](o, w.view(torch.uint8) if fp8 else w, s, out, T, *o.stride(), *w.stride(), s.stride(0),
                         *out.stride(), R=R, D=D, FP8=fp8,
                         SCALE_BN=G * R // scale.shape[0] if fp8 else 1, SCALE_BK=D // scale.shape[1] if fp8 else 1,
                         BLOCK_T=BLOCK_T, BLOCK_R=BLOCK_R, BLOCK_H=BLOCK_H, num_warps=4, num_stages=3)
    return out


def dequantize(w, scale, out):
    """FP8 w [rows, cols] x FP32 block scales -> BF16 out [rows, cols]."""
    rows, cols = w.shape
    _dequant_kernel[(triton.cdiv(rows, 64), triton.cdiv(cols, 128))](
        w.view(torch.uint8), scale, out, rows, cols, scale.stride(0), SCALE_BN=rows // scale.shape[0], SCALE_BK=cols // scale.shape[1],
        BLOCK_M=64, BLOCK_N=128, num_warps=4)
    return out


def _prefill_mxfp8_grouped(o: torch.Tensor, wo_a_fp8: torch.Tensor, wscales_swizzled: list[torch.Tensor]) -> torch.Tensor:
    from sglang.srt.layers.attention.dsv4 import di_align
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mm_mxfp8, flashinfer_mxfp8_quantize

    T, G, D = o.shape
    R = wo_a_fp8.shape[1]
    # Pack [T, G, D] -> [G, T, D] contiguous and quantize all G*T rows in ONE kernel launch!
    o_gtd = o.transpose(0, 1).contiguous().view(G * T, D)
    if di_align.enabled("A7"):
        from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import mxfp8_quantize_swizzled
        q_all, s_all = mxfp8_quantize_swizzled(o_gtd, align_a7=True)
    else:
        q_all, s_all = flashinfer_mxfp8_quantize(o_gtd, is_sf_swizzled_layout=True, alignment=32)

    # Because T % 128 == 0 and D % 128 == 0, each 128-row swizzled SF block is self-contained per group g!
    s_per_group = s_all.numel() // G
    s_flat = s_all.view(-1)
    out = torch.empty((T, G, R), dtype=torch.bfloat16, device=o.device)
    for g in range(G):
        qg = q_all[g * T : (g + 1) * T]
        sg = s_flat[g * s_per_group : (g + 1) * s_per_group]
        out[:, g, :] = flashinfer_mm_mxfp8(
            qg,
            wo_a_fp8[g].t(),
            sg,
            wscales_swizzled[g],
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cutlass",
        )
    return out


def _bmm_token_major(o: torch.Tensor, wo_a_bf16: torch.Tensor) -> torch.Tensor:
    out = torch.empty((o.shape[0], wo_a_bf16.shape[0], wo_a_bf16.shape[1]), dtype=o.dtype, device=o.device)
    with torch.no_grad():
        torch.bmm(o.transpose(0, 1), wo_a_bf16.transpose(1, 2), out=out.transpose(0, 1))
    return out


def _apply(original, o, wo_a, *args, **kwargs):
    """Around SGLang's ``_apply_wo_a_bf16_matmul(o [T, G, D], wo_a [G, R, D], ...)``."""
    ptr = wo_a.data_ptr()
    scale = _scales.get(ptr) if wo_a.dtype == torch.float8_e4m3fn else None
    if o.dim() == 3 and o.dtype == torch.bfloat16 and o.is_cuda and o.shape[0] > 0:
        T = o.shape[0]
        if T <= MAX_T and (scale is not None or (wo_a.dtype == torch.bfloat16 and tuple(wo_a.shape) != FAST_SHAPE)):
            return einsum(o, wo_a, scale)
        if PREFILL_MXFP8 and scale is not None and T >= PREFILL_MXFP8_MIN_T and (T % 128 == 0):
            ws_list = _mxfp8_wscales.get(ptr)
            if ws_list is not None:
                return _prefill_mxfp8_grouped(o, wo_a, ws_list)
    if scale is not None:
        key = (wo_a.device, wo_a.numel())
        if key not in _scratch:
            _scratch[key] = torch.empty(wo_a.shape, dtype=torch.bfloat16, device=wo_a.device)
        flat = wo_a.reshape(-1, wo_a.shape[-1])
        wo_a = dequantize(flat, scale, _scratch[key].view(flat.shape)).view(wo_a.shape)
        if o.dim() == 3 and o.dtype == torch.bfloat16 and o.is_cuda and o.shape[0] > 0 and tuple(wo_a.shape) != FAST_SHAPE:
            return _bmm_token_major(o, wo_a)
    return original(o, wo_a, *args, **kwargs)


def _tap(weights):
    for name, tensor in weights:
        m = SCALE_NAME.search(name) if _capture[0] is not None else None
        if m:
            _capture[0][int(m.group(1))] = tensor.detach().clone()
        yield name, tensor


def _streaming(original, weights, *args, **kwargs):
    return original(_tap(weights) if _capture[0] is not None else weights, *args, **kwargs)


def _load_weights(original, self, weights, *args, **kwargs):
    if kwargs.get('is_nextn', args[0] if args else False):
        return original(self, weights, *args, **kwargs)
    _capture[0] = {}
    try:
        result = original(self, weights, *args, **kwargs)
        captured = _capture[0]
    finally:
        _capture[0] = None
    if captured:
        _convert(self, captured)
    return result


def _relative(a, b):
    return ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30)).item()


def _build_exact_swizzled_scales(raw_scale: torch.Tensor, G: int, R: int, D: int, w_bf16_3d: torch.Tensor, fp8_3d: torch.Tensor):
    """Expand checkpoint [G*R/32, D/32] E8M0 scales to [G, R, D/32] UE8M0 uint8 and swizzle via
    FlashInfer block_scale_interleave — zero weight re-quantization and zero duplicate weight storage!"""
    try:
        from flashinfer import block_scale_interleave
        s_dev = raw_scale.to(device=fp8_3d.device)
        if s_dev.dtype == torch.uint8:
            s_u8_block = s_dev
        elif hasattr(torch, 'float8_e8m0fnu') and s_dev.dtype == torch.float8_e8m0fnu:
            s_u8_block = s_dev.view(torch.uint8)
        else:
            # Convert exact power-of-2 float32 scale 2^(e-127) -> uint8 biased exponent e
            s_u8_block = ((s_dev.float().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        # Repeat each 32-row scale across its 32 rows -> [G*R, D//32]
        s_rows = s_u8_block.repeat_interleave(BLOCK, dim=0).contiguous().view(G, R, D // BLOCK)
        ws_list = [block_scale_interleave(s_rows[g].contiguous()).contiguous() for g in range(G)]
        # Self-verify numerical accuracy on a 128-row random input before enabling!
        o_chk = torch.randn((128, G, D), device=fp8_3d.device, dtype=torch.bfloat16)
        ref = torch.einsum('tgd,grd->tgr', o_chk, w_bf16_3d)
        got = _prefill_mxfp8_grouped(o_chk, fp8_3d, ws_list)
        rel = _relative(ref, got)
        if rel > 0.03:
            logger.warning('xtier_86kd: wo_a MXFP8 self-check rel_rms=%.5f > 0.03, keeping BF16 bmm fallback', rel)
            return None
        return ws_list
    except Exception as exc:
        logger.warning('xtier_86kd: wo_a MXFP8 scale swizzle skipped: %s', exc)
        return None


def _convert(model, captured):
    """Each captured layer's BF16 wo_a back to its exact FP8 form, checked; layers that do not check keep BF16."""
    from sglang.srt.runtime_context import get_parallel
    done, mxfp8_done, kept, freed = 0, 0, [], 0
    for name, mod in model.named_modules():
        m = LAYER_NAME.search(name + '.')
        if not (m and int(m.group(1)) in captured and all(hasattr(mod, a) for a in ('wo_a', 'o_lora_rank',
                                                                                   'n_local_groups'))):
            continue
        layer, w = int(m.group(1)), getattr(mod.wo_a, 'weight', None)
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda or w.dim() != 2:
            continue
        G, R = mod.n_local_groups, mod.o_lora_rank
        raw_s = captured[layer]
        if raw_s.shape[0] * BLOCK != w.shape[0]:
            parallel = get_parallel()
            local_blocks = w.shape[0] // BLOCK
            if (w.shape[0] % BLOCK == 0 and raw_s.shape[0] == local_blocks * parallel.attn_tp_size
                    and raw_s.shape[1] * BLOCK == w.shape[1]):
                start = parallel.attn_tp_rank * local_blocks
                raw_s = raw_s[start:start + local_blocks]
        if layer == 0:
            logger.info('xtier_86kd: wo_a TP%d rank%d weight=%s scale=%s MAX_T=%d PREFILL_MXFP8=%s',
                        get_parallel().attn_tp_size, get_parallel().attn_tp_rank,
                        tuple(w.shape), tuple(raw_s.shape), MAX_T, PREFILL_MXFP8)
        s = raw_s.to(device=w.device).float().contiguous()
        sn, sk = s.shape
        ok = (G, R, w.shape[1]) != FAST_SHAPE and w.shape == (sn * BLOCK, sk * BLOCK) and G * R == w.shape[0]
        if ok:
            blocks = w.float().view(sn, BLOCK, sk, BLOCK)
            fp8 = (blocks / s[:, None, :, None]).view_as(w).to(torch.float8_e4m3fn)
            ok = torch.equal((fp8.float().view(sn, BLOCK, sk, BLOCK) * s[:, None, :, None]).view_as(w).to(w.dtype), w)
            del blocks
        if ok:
            o = torch.randn(6, G, w.shape[1], device=w.device, dtype=torch.bfloat16)
            ok = _relative(torch.einsum('tgd,grd->tgr', o, w.view(G, R, -1)), einsum(o, fp8.view(G, R, -1), s)) <= 1e-3
        if not ok:
            kept.append(layer)
            continue
        ws_list = None
        if PREFILL_MXFP8 and R % 128 == 0 and w.shape[1] % 128 == 0:
            ws_list = _build_exact_swizzled_scales(raw_s, G, R, w.shape[1], w.view(G, R, -1), fp8.view(G, R, -1))
        mod.wo_a.weight = torch.nn.Parameter(fp8, requires_grad=False)
        _scales[fp8.data_ptr()] = s
        if ws_list is not None:
            _mxfp8_wscales[fp8.data_ptr()] = ws_list
            mxfp8_done += 1
        freed += w.numel()
        done += 1
    torch.cuda.empty_cache()
    logger.info('xtier: DeepSeek wo_a: %d layers in FP8 (%d prefill-MXFP8 ready, %.2f GiB of BF16 freed)%s',
                done, mxfp8_done, freed / 2**30, f', layers {kept} kept BF16' if kept else '')


def install(registry, hook_type):
    if not enabled():
        return
    registry.register(APPLY, _apply, hook_type.AROUND)
    if os.environ.get('XTIER_DS_WO_A_FP8', '1') != '0':
        registry.register(STREAMING, _streaming, hook_type.AROUND)
        registry.register(LOAD, _load_weights, hook_type.AROUND)
