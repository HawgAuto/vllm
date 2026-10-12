# SPDX-License-Identifier: Apache-2.0
"""Default-off AMD attention gate for the single unpadded Qwen K1 protocol.

Cache ownership, compaction and group handoff live ONLY in Qwen4ExpK1Proposer.
No kernel changes; neither opt-in alone admits attention. CPU qualification is
not GPU or real-model serving admission.
"""
import os
from vllm.v1.spec_decode.qwen4_mtp_profile import validate_qwen4_mtp_k1

FLAG = 'VLLM_QWEN4_AMD_QSA_MTP1'

def validate_mtp1(v):
    if os.environ.get(FLAG) != '1':
        raise NotImplementedError('AMD QSA does not support speculative decoding/MTP')
    if not validate_qwen4_mtp_k1(v):
        raise NotImplementedError('AMD QSA MTP1 requires explicit unpadded framework opt-in')
    # Preserve attention-specific checks not subsumed by framework validation.
    for c in (v.model_config.hf_text_config,
              v.speculative_config.draft_model_config.hf_text_config):
        if (getattr(c, 'indexer_compress_ratio', None) != 4
                or getattr(c, 'indexer_head_dim', None) != 128
                or getattr(c, 'indexer_kv_heads', None) != 1
                or getattr(c, 'indexer_n_heads', None) != 4
                or getattr(c, 'head_dim', None) != 256):
            raise NotImplementedError('AMD QSA MTP1 requires current native Qwen QSA geometry')
    block = v.cache_config.block_size
    if block is not None and block % 8:
        raise NotImplementedError('AMD QSA MTP1 requires block size divisible by ring8')
