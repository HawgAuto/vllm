# SPDX-License-Identifier: Apache-2.0
"""Default-off framework-only Qwen4Exp MTP K1 profile.

API: VllmConfig.additional_config["qwen4exp_mtp_k1"] = True.
This does not admit AMD QSA attention: its independent guard remains authoritative.
"""
import torch
from vllm import envs


def qwen4_mtp_k1_enabled(vllm_config) -> bool:
    additional = vllm_config.additional_config
    return isinstance(additional, dict) and additional.get("qwen4exp_mtp_k1") is True


def validate_qwen4_mtp_k1(vllm_config) -> bool:
    if not qwen4_mtp_k1_enabled(vllm_config):
        return False

    def require(condition, message):
        if not condition:
            raise ValueError("Qwen4Exp MTP K1: " + message)

    spec = vllm_config.speculative_config
    require(spec is not None, "real speculative_config required")
    require(spec.method == "mtp", "method must be mtp")
    require(spec.num_speculative_tokens == 1, "exactly K1 required")
    require(spec.disable_padded_drafter_batch is True, "unpadded drafter required")
    require(not spec.parallel_drafting, "parallel/tree drafting unsupported")
    require(spec.num_speculative_tokens_per_batch_size is None, "dynamic K unsupported")
    require(not spec.enable_adaptive_verification, "adaptive verification unsupported")
    require(not spec.use_heterogeneous_vocab, "heterogeneous vocabulary unsupported")
    draft = spec.draft_model_config
    require(draft is not None, "resolved native draft config required")
    hf = draft.hf_config
    text = getattr(hf, "text_config", hf)
    require(hf.model_type == "qwen4_exp_mtp" and hf.architectures == ["Qwen4ExpMTP"],
            "exact native Qwen4ExpMTP architecture required")
    require(hf.n_predict == 1, "one MTP layer required")
    require(hf.hc_mult == text.hc_count == 4 and text.hidden_size == 2560,
            "native 2560 x 4 HC geometry required")
    require(text.indexer_compress_ratio == 4 and text.indexer_head_dim == 128
            and text.indexer_kv_heads == 1 and text.head_dim == 256
            and text.num_key_value_heads == 2, "native QSA geometry required")
    for parallel in (vllm_config.parallel_config, spec.draft_parallel_config):
        require(parallel is not None, "resolved parallel configs required")
        require(all(getattr(parallel, name) == 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "decode_context_parallel_size", "prefill_context_parallel_size",
            "data_parallel_size")), "TP/PP/DCP/PCP/DP must all equal 1")
        require(not parallel.use_ubatching and not parallel.enable_eplb,
                "ubatching/EPLB unsupported")
    model = vllm_config.model_config
    require(model.enforce_eager and draft.enforce_eager, "target and draft must be eager")
    require(model.dtype == draft.dtype == torch.bfloat16, "BF16 activation/KV profile required")
    require(not model.is_mm_prefix_lm and model.rswa_window is None,
            "PrefixLM/R-SWA unsupported")
    require(vllm_config.scheduler_config.max_num_seqs == 1, "max_num_seqs must equal 1")
    require(vllm_config.scheduler_config.async_scheduling is False, "async must be explicitly off")
    cache = vllm_config.cache_config
    require(cache.enable_prefix_caching is False and not cache.kv_sharing_fast_prefill,
            "prefix reuse/fast-prefill unsupported")
    require(cache.cache_dtype in ("auto", "bfloat16")
            and spec.kv_cache_dtype in (None, "auto", "bfloat16"), "BF16 KV required")
    require(vllm_config.kv_transfer_config is None, "KV connector unsupported")
    require(not envs.VLLM_BATCH_INVARIANT, "batch invariance unsupported")
    return True


def validate_qwen4_mtp_streams(result, rows, sample_width, feedback_width):
    if (type(result) is not tuple or len(result) != 2
            or not all(isinstance(t, torch.Tensor) for t in result)
            or result[0].shape != (rows, sample_width)
            or result[1].shape != (rows, feedback_width)):
        raise ValueError("Qwen MTP stream contract requires (sample_hidden, HC_feedback)")
