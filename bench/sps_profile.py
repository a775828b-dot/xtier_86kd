"""The SPS cost table DSpark's compact (variable-length) verification needs, measured on this machine by SGLang's
dspark_sps_profiler. Without a measured table, or with a flat one, SGLANG_RAGGED_VERIFY_MODE=compact verifies full width
every step, the same as static.

Includes Attention-DP (--attn-dp-size > 1) support:
- Detects true DP rank count from len(/server_info.internal_states) when SGLang reports dp_size=1 under --attn-dp-size 8.
- Dispatches balanced per-DP-rank load batches via ``x-override-routed-dp-rank`` so all DP ranks align immediately.
"""
from __future__ import annotations

import dataclasses
from functools import partial
import json
import sys
import threading
import urllib.request


RUN_TP = {
    '--base-url': ['http://127.0.0.1:30000'],
    '--fracs': ['0.2', '0.4', '0.6', '0.8', '1.0'],
    '--batch-size': ['1', '2', '4', '8', '16', '20', '32'],
}
FIT = {'--no-plot': []}


def _replace_ctx(ctx, **changes):
    if hasattr(ctx, "__struct_fields__"):
        import msgspec
        return msgspec.structs.replace(ctx, **changes)
    return dataclasses.replace(ctx, **changes)


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _install_attn_dp_patches(profiler) -> None:
    orig_fetch_ctx = profiler.fetch_server_context

    def _patched_fetch_ctx(*, base_url: str, local_tokenizer_path=None, allowed_modes=("static",)):
        ctx = orig_fetch_ctx(
            base_url=base_url,
            local_tokenizer_path=local_tokenizer_path,
            allowed_modes=allowed_modes,
        )
        info = _get_json(base_url + "/server_info", timeout=profiler.DEFAULT_TIMEOUT)
        states = info.get("internal_states") or []
        real_dp = max(int(ctx.dp_size or 1), len(states))
        if real_dp != ctx.dp_size:
            max_per_dp = states[0].get("effective_max_running_requests_per_dp", -1) if states else -1
            skip_max = float(max_per_dp * real_dp) if max_per_dp and max_per_dp > 0 else ctx.skip_max_running_requests_threshold
            ctx = _replace_ctx(
                ctx,
                dp_size=real_dp,
                skip_max_running_requests_threshold=skip_max,
            )
        profiler._xtier_detected_dp_size = real_dp
        return ctx

    orig_start_load = profiler.start_load

    def _patched_start_load(*, base_url: str, num_requests: int, input_len: int, max_new_tokens: int,
                            temperature: float, vocab_size: int, rng) -> threading.Thread:
        dp_sz = getattr(profiler, "_xtier_detected_dp_size", 1)
        if dp_sz <= 1 or num_requests % dp_sz != 0:
            return orig_start_load(
                base_url=base_url,
                num_requests=num_requests,
                input_len=input_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                vocab_size=vocab_size,
                rng=rng,
            )
        token_high = vocab_size - profiler.RANDOM_TOKEN_HIGH_MARGIN
        per_rank = num_requests // dp_sz
        rank_payloads = []
        for r in range(dp_sz):
            ids = [
                [rng.randrange(profiler.RANDOM_TOKEN_LOW, token_high) for _ in range(input_len)]
                for _ in range(per_rank)
            ]
            rank_payloads.append((
                r,
                {
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "stream": False,
                },
            ))

        def _post_all() -> None:
            subthreads = []
            for r, payload in rank_payloads:
                def _one(rk=r, pl=payload):
                    try:
                        _post_json(
                            base_url + "/generate",
                            pl,
                            {"x-override-routed-dp-rank": str(rk)},
                            profiler.DEFAULT_TIMEOUT,
                        )
                    except Exception:
                        pass
                t = threading.Thread(target=_one, daemon=True)
                t.start()
                subthreads.append(t)
            for t in subthreads:
                t.join()

        master = threading.Thread(target=_post_all, daemon=True)
        master.start()
        return master

    profiler.fetch_server_context = _patched_fetch_ctx
    profiler.start_load = _patched_start_load


def main():
    from sglang.benchmark import dspark_sps_profiler as profiler

    profiler.ols_resid_backfit = partial(profiler.ols_resid_backfit, mbin_w=1)
    _install_attn_dp_patches(profiler)

    args = sys.argv[1:]
    if not args or args[0] not in ('all', 'run', 'fit'):
        args = ['all'] + args
    added = {**(RUN_TP if args[0] in ('all', 'run') else {}), **(FIT if args[0] in ('all', 'fit') else {})}
    for flag, values in added.items():
        if flag not in args:
            args += [flag, *values]
    sys.argv = [sys.argv[0]] + args
    profiler.cli_main()


if __name__ == '__main__':
    main()
