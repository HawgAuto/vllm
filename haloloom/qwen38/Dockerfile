# syntax=docker/dockerfile:1
FROM ghcr.io/hawgauto/haloloom-vllm-full-gfx1151@sha256:3f80b2e086010a8d5c3c89ed236318f38a2ad3549170799765e29844e3800139
USER root

# Verified native wheels and dependency payloads; no network resolver or CUDA swap.
RUN --mount=type=bind,source=haloloom/qwen38/wheelhouse,target=/wheels \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-index --no-deps \
    --force-reinstall /wheels/*.whl

# Source comes from this fork. The native ELF extensions in the pinned wheel are
# unchanged from the accepted build; the native source ancestry is recorded.
COPY vllm/ /opt/venv/lib/python3.14/site-packages/vllm/
COPY haloloom/qwen38/python/ /opt/venv/lib/python3.14/site-packages/
COPY haloloom/qwen38/kernels/ /opt/venv/lib/python3.14/site-packages/
COPY haloloom/qwen38/python/ /opt/haloloom/qwen38/python/
COPY haloloom/qwen38/kernels/ /opt/haloloom/qwen38/kernels/
COPY haloloom/qwen38/data/ /opt/haloloom/qwen38/data/
COPY haloloom/qwen38/launch.py haloloom/qwen38/validate.py /opt/haloloom/qwen38/

RUN /opt/venv/bin/python3 -B /opt/haloloom/qwen38/validate.py
ENV HSA_ENABLE_SDMA=0 \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HYPERLOOM_GFX1151_LOWBIT_BRIDGE=0 \
    VLLM_EXPERIMENTAL_GFX1151_MOE=1 \
    VLLM_QWEN4_PLE_SSD=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1
LABEL org.opencontainers.image.source="https://github.com/HawgAuto/vllm" \
      haloloom.vllm_source_ref="source-bound-by-installed-files.json" \
      haloloom.runtime_profile="qwen38-bf16core-int8core-gfx1151"
WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/python3", "-B", "/opt/haloloom/qwen38/launch.py"]
