# xtier_86kd

DeepSeek V4.1 Flash 1017 acceleration for SGLang tensor parallelism 8 on eight RTX PRO 6000D GPUs. This project ports the DeepSeek pack from [XTier](https://github.com/a775828b-dot/xtier) without enabling XTier's hybrid MoE engine. SGLang's Humming MoE runner remains in use.

The complete DeepSeek pack is in `src/xtier_86kd/packs/deepseek_v4`. Its applicable hooks cover DSpark verification, paged decode indexing, FP8 GEMM tuning, FlashInfer MXFP8 small batch, attention `wo_a`, and draft FP8 LM head. Current SGLang already selects dense-block candidates on SM120. The legacy candidate override is retained behind `XTIER_86KD_LEGACY_INDEXER=1` for older commits and must stay off on the tested commit. Legacy GPU hooks and the optional Engram NVMe row store are retained. The row store's C++ source is in `csrc/tables`; it is inactive with the default host Engram table.

## Install on the server

```bash
/data/xtier-env/bin/pip install -e /data/src/xtier_86kd --no-deps
```

Set `XTIER=0` and `XTIER_86KD=1` in the existing SGLang startup environment. Existing `XTIER_DS_*`, `XTIER_MXFP8_B12X_ROWS`, `XTIER_DRAFT_LMHEAD_FP8`, and related pack switches remain supported. Use the same SGLang commit and serving parameters as the baseline for an A/B benchmark. `XTIER_86KD=0` removes all hooks without reinstalling.

The FP8 tuned config JSONs were measured for RTX PRO 5000; they remain in the pack but do not match the 6000D device name. New 6000D tuning must be measured before adding configs.

Source license: Apache-2.0; see `LICENSE`.

