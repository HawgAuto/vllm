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
- SSD-backed PLE and an explicit 5.5 GiB BF16 KV pool. Model files remain immutable.
- Ordinary compiled concurrency uses the repaired PLE request axis and the
  request-aware Triton QSA scorer for multiple requests. The singleton HIP
  specialization is unchanged. No new batch-HIP speedup is claimed or enabled.
- MTP3 retains a one-sequence guard. Do not extrapolate ordinary concurrency to MTP.
- Profiler stack/lifecycle fixes, speculative scalar normalization and excluded
  Quark embedding preservation are retained from the public reconciliation branch.

## Container launch

Use the image and immutable digest published in HaloLoom's matching runtime
release manifest. Mount one complete variant at `/model` read-only and a writable
cache directory at `/workspace`. Expose `/dev/kfd` and the appropriate AMD render
node, not NVIDIA devices. Select the model variant explicitly:

```sh
# Inside the matching image:
/opt/venv/bin/python3 /opt/haloloom/qwen38/launch.py --variant int8 --mtp 0 --max-num-seqs 4 --context 4096
/opt/venv/bin/python3 /opt/haloloom/qwen38/launch.py --variant bf16 --mtp 3 --max-num-seqs 1
```

`--print-command` shows the exact server command. `--cpu-check` validates installed
payload hashes and public model indices without claiming GPU inference. Ordinary
multi-sequence launches disable prefix caching, matching the qualified sweep;
`--max-num-batched-tokens` defaults to 12288. An explicit 2048 setting selects the
separately measured smaller-chunk geometry, not an inferred universal improvement.
The native kernel source and accepted binaries are under `csrc/` and `kernels/`.
Dependency wheels are build inputs, not committed Git objects; the matching
HaloLoom release packet carries their hashes and source bindings.

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
