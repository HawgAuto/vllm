# Qwen3.8 Flash Next — HaloLoom runtime

This fork packages the current gfx1151 runtime for the two losslessly packaged
`Bf16Core-int4-int8` and `int8Core-int4-int8` variants. The method and build tools
are [HaloLoom](https://github.com/HawgAuto/HaloLoom). This is the required serving
fork, not support in unmodified upstream vLLM.

## Runtime contract

- ROCm 10 / Torch 2.13 / Python 3.14; AMD gfx1151 only for these native kernels.
- Native vLLM extension ancestry: `21ee743f4`; public repair base:
  `ab3c2dc13266e6f85fbdda8af95eaec11fb30bd8`.
- BF16 or INT8 dense core, with the accepted mixed INT4/INT8 expert recipe.
- SSD-backed PLE and an explicit BF16 KV pool sized per profile. Model files remain immutable.
- Ordinary compiled concurrency uses the repaired PLE request axis and the
  request-aware Triton QSA scorer for multiple requests. The singleton HIP
  specialization is unchanged. No new batch-HIP speedup is claimed or enabled.
- MTP3 retains a one-sequence guard. Do not extrapolate ordinary concurrency to MTP.
- Profiler stack/lifecycle fixes, speculative scalar normalization and excluded
  Quark embedding preservation are retained from the public reconciliation branch.

## Container launch

Image: `ghcr.io/hawgauto/haloloom-vllm-full-gfx1151:v0.1.5-qwen38-quickstart`. It is built
by `Dockerfile.quickstart`: the qualified `v0.1.5-qwen38` runtime plus this launcher and the
frozen compile/autotune seeds (`data/seeds.json` binds every seed file by SHA-256; the
~2.2 GB seed tree is a build input, not a Git object).

Mount one complete variant at `/model` read-only and a writable cache directory at
`/workspace`. Expose `/dev/kfd` and the AMD render node, not NVIDIA devices:

```sh
docker run -d --device /dev/kfd --device /dev/dri/renderD128 --group-add $(stat -c %g /dev/kfd) \
  --user 1000:1000 --shm-size 16g -p 8000:8000 \
  -v /path/to/int8Core-int4-int8:/model:ro -v /path/to/cache:/workspace \
  ghcr.io/hawgauto/haloloom-vllm-full-gfx1151:v0.1.5-qwen38-quickstart \
  qwen38 --variant int8 --mtp 3                 # default profile: 262,144 native context
#  qwen38 --variant bf16 --mtp 0 --profile max  # max profile: YaRN x4, per-variant limit
```

Profiles:

| Profile | Context | Rope | KV pool |
| --- | --- | --- | --- |
| `default` | 262,144 | native | 7.0 GiB ordinary / 8.0 GiB MTP3 |
| `max` int8 ordinary | 1,048,576 | YaRN x4 | 25.5 GiB |
| `max` int8 MTP3 | 917,504 | YaRN x4 | 24.75 GiB |
| `max` bf16 ordinary | 851,968 | YaRN x4 | 21.0 GiB |
| `max` bf16 MTP3 | 720,896 | YaRN x4 | 19.75 GiB |

Seeds: `seeds/int8`, `seeds/bf16` (default profile) and `seeds/int8-yarn4`,
`seeds/bf16-yarn4` (max profile). vLLM's compile-cache key includes `max_model_len` but
omits the dense-core variant and `--hf-overrides`, so the launcher keeps one cache root per
variant and rope setting and never shares compiled graphs between them. Any other
`--context` value reuses only the tuned kernels and compiles its graph once.

Sampling defaults to Qwen's thinking-mode values from the checkpoint's
`generation_config.json` (`--sampling thinking`); `--sampling instruct` and
`--sampling vllm` are available. OpenAI tool calling is on by default (`--enable-auto-tool-choice
--tool-call-parser qwen3_coder`, matching the chat template's XML tool-call format;
`--tool-call-parser none` disables it). The chat template defaults to `reasoning_effort`
`xhigh` (accepts `xhigh`, `medium`, `low`) and keeps prior-turn reasoning
(`preserve_thinking`). `--print-command` shows the exact server command;
`--cpu-check` validates installed payload hashes and public model indices without
claiming GPU inference. Ordinary multi-sequence launches disable prefix caching, matching
the qualified sweep; `--max-num-batched-tokens` defaults to 12288. MTP3 is limited to one
sequence. The native kernel source and accepted binaries are under `csrc/` and `kernels/`.
Dependency wheels are build inputs, not committed Git objects; the matching HaloLoom
release packet carries their hashes and source bindings.

## Verification and limits

The isolated antecedent ordinary sweep delivered all 126 requests / 64,512 output
tokens at 1/2/4/8/16/32 submitted streams. At 32 submissions only 24 decodes
actually overlapped; near-full KV occupancy and high first-token latency prevent
a sustained 32-way or low-latency claim. Concurrency eight had a throughput cliff.
Generated token identity varied under repetition. These are scoped delivery and
operator checks, **not independent model-quality qualification**.

The follow-up native batch-HIP ABBA experiment failed its full-model wall-speed
admission gate and is intentionally not part of this default runtime.
Public-layout model execution and registry availability are recorded separately
in the matching release receipt; a source commit or static check is not that proof.

Additive source-bound CPU regressions are under `tests/`; existing reconciliation
regressions remain under `tests/haloloom/` at the repository root. GPU/model tests
must run in an isolated owned transaction, not while another iGPU job is active.
