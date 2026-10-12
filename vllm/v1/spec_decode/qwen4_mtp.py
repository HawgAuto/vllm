# SPDX-License-Identifier: Apache-2.0
"""Unpadded, eager Qwen4Exp K1 proposer with explicit KV-group ownership.

Uses the native Eagle input shifting, forward, sampling and index-sharing hooks.
No attention implementation, guard, cache geometry or GPU kernel is changed.
"""
from copy import copy
from dataclasses import fields
import torch
from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec, FullAttentionSpec, MLAAttentionSpec,
    UniformTypeKVCacheSpecs, KVQuantMode,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.qwen4_mtp_profile import validate_qwen4_mtp_k1
from vllm.v1.worker.utils import AttentionGroup


class Qwen4ExpK1Proposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        if not validate_qwen4_mtp_k1(vllm_config):
            raise ValueError("Qwen4ExpK1Proposer requires explicit qwen4exp_mtp_k1 opt-in")
        super().__init__(vllm_config, device, runner)
        self._qwen_groups = {}
        self._qwen_active = {}
        self._qwen_compacted = False

    def _resolve_qwen_owners(self, kv_cache_config):
        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        resolved = {}
        owner_ids = set()
        names = self._draft_attn_layer_names
        if not names:
            raise ValueError("Qwen MTP draft attention owners are missing")
        for name in sorted(names):
            owner = layers.get(name)
            if owner is None or id(owner) in owner_ids:
                raise ValueError(f"Missing/aliased Qwen attention owner: {name}")
            owner_ids.add(id(owner))
            matches = [(gid, group) for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                       for layer_name in group.layer_names if layer_name == name]
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous KV group identity: {name}")
            gid, group = matches[0]
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                if name not in spec.kv_cache_specs:
                    raise ValueError(f"Missing per-layer uniform spec: {name}")
                spec = spec.kv_cache_specs[name]
            if getattr(owner, "kv_sharing_target_layer_name", None) is not None:
                raise ValueError("Qwen MTP K1 does not support cross-owner KV aliases")
            self._validate_qwen_spec(spec)
            owner_spec = owner.get_kv_cache_spec(self.vllm_config)
            # Allocator page padding is physical layout, not an owner substitution.
            if type(owner_spec) is not type(spec) or any(
                    getattr(owner_spec, f.name) != getattr(spec, f.name)
                    for f in fields(spec) if f.name != "page_size_padded"):
                raise ValueError(f"Qwen owner/cache spec mismatch: {name}")
            resolved[name] = (gid, spec, owner)
        families = {type(spec) for _, spec, _ in resolved.values()}
        if families != {FullAttentionSpec, MLAAttentionSpec, CircularBufferSpec}:
            raise ValueError("Qwen MTP requires main, compressed and raw cache owners")
        main_gids = {gid for gid, spec, _ in resolved.values() if type(spec) is FullAttentionSpec}
        if len(main_gids) != 1:
            raise ValueError("Ambiguous primary main KV group")
        return resolved, main_gids.pop()

    @staticmethod
    def _validate_qwen_spec(spec):
        common = spec.dtype == torch.bfloat16 and spec.kv_quant_mode == KVQuantMode.NONE
        if type(spec) is FullAttentionSpec:
            valid = (spec.num_kv_heads == 2 and spec.head_size == spec.head_size_v == 256
                     and spec.tokens_per_state == 1 and spec.uses_slot_mapping
                     and spec.sliding_window is None and spec.attention_chunk_size is None
                     and not spec.non_causal)
        elif type(spec) is MLAAttentionSpec:
            valid = (spec.num_kv_heads == 1 and spec.head_size == 128
                     and spec.tokens_per_state == 4 and spec.uses_slot_mapping
                     and spec.block_size % 4 == 0 and spec.storage_block_size is None)
        elif type(spec) is CircularBufferSpec:
            # 128 BF16 channels + 3 exact int64 MRoPE axes = 140 BF16 units.
            valid = (spec.num_kv_heads == 1 and spec.head_size == 140
                     and spec.head_size_v == 0 and spec.block_size == 8
                     and spec.num_states == 8 and not spec.uses_slot_mapping)
        else:
            valid = False
        if not common or not valid:
            raise ValueError(f"Unsupported Qwen MTP K1 cache geometry: {spec}")

    def validate_same_kv_cache_group(self, kv_cache_config):
        # Replace the single-group invariant with complete unique-owner validation.
        self._resolve_qwen_owners(kv_cache_config)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        resolved, primary_gid = self._resolve_qwen_owners(kv_cache_config)
        groups = {}
        for name, (gid, spec, owner) in resolved.items():
            if kernel_block_sizes is not None:
                if gid >= len(kernel_block_sizes) or kernel_block_sizes[gid] != spec.block_size:
                    raise ValueError("Qwen MTP K1 does not qualify split kernel/manager blocks")
            backend = owner.get_attn_backend()
            # Same backend is insufficient: QSA side owners have different specs.
            key = (gid, backend.full_cls_name(), spec, getattr(owner, "num_heads", 0))
            if key not in groups:
                groups[key] = AttentionGroup(backend, [], spec, gid)
            groups[key].layer_names.append(name)
        # Construct after collecting names, so builder head inference sees all owners.
        for group in groups.values():
            group.create_metadata_builders(self.vllm_config, self.device)
        self.draft_attn_groups = list(groups.values())
        self.kv_cache_gid = primary_gid
        self.block_size = next(spec.block_size for gid, spec, _ in resolved.values()
                               if type(spec) is FullAttentionSpec)
        self._qwen_required_gids = {gid for gid, _, _ in resolved.values()}
        self._qwen_raw_gids = {gid for gid, spec, _ in resolved.values()
                              if type(spec) is CircularBufferSpec}
        if self._qwen_raw_gids & {gid for gid, spec, _ in resolved.values()
                                  if type(spec) is not CircularBufferSpec}:
            raise ValueError("Raw circular state must have its own KV group")
        self.begin_qwen4_mtp_metadata()

    def begin_qwen4_mtp_metadata(self):
        """Runner starts a fresh capture epoch before its KV-group loop."""
        self._qwen_groups = {}
        self._qwen_active = {}
        self._qwen_compacted = False

    def capture_qwen4_mtp_metadata(self, gid, common):
        if gid not in self._qwen_required_gids:
            return  # target-only caches are not draft owners
        if gid in self._qwen_groups:
            raise ValueError(f"Duplicate Qwen metadata group {gid}")
        self._validate_common(common)
        cm = copy(common)
        cm.block_table_tensor = common.block_table_tensor.clone()
        # Raw has uses_slot_mapping=False: never read an uninitialized generic map.
        cm.slot_mapping = (torch.full((common.num_actual_tokens,), -1,
                                      dtype=torch.int64, device=self.device)
                           if gid in self._qwen_raw_gids else common.slot_mapping.clone())
        cm.query_start_loc = common.query_start_loc.clone()
        cm.query_start_loc_cpu = common.query_start_loc_cpu.clone()
        cm.seq_lens = common.seq_lens.clone()
        self._qwen_groups[gid] = cm

    def _validate_common(self, cm):
        n = cm.num_actual_tokens
        if (cm.num_reqs != 1 or n < 1 or n > self.max_num_tokens
                or cm.query_start_loc_cpu.tolist() != [0, n]
                or not torch.equal(cm.query_start_loc.cpu(), cm.query_start_loc_cpu)
                or cm.causal is not True or cm.mm_req_doc_ranges
                or cm.rswa_prefix_lens is not None):
            raise ValueError("Qwen MTP K1 requires coherent unpadded single-request rows")
        if (cm.seq_lens_cpu_upper_bound is None
                or not torch.equal(cm.seq_lens.cpu(), cm.seq_lens_cpu_upper_bound)
                or not n <= int(cm.seq_lens[0]) <= self.max_model_len):
            raise ValueError("Qwen MTP K1 requires exact CPU/device sequence lengths")

    def _require_capture(self, common):
        self._validate_common(common)
        if set(self._qwen_groups) != self._qwen_required_gids:
            raise ValueError("Incomplete runner-to-Qwen KV-group metadata capture")
        for cm in self._qwen_groups.values():
            if (cm.num_actual_tokens != common.num_actual_tokens
                    or not torch.equal(cm.query_start_loc, common.query_start_loc)
                    or not torch.equal(cm.seq_lens, common.seq_lens)):
                raise ValueError("Stale/misaligned runner-to-Qwen group rows")
        primary = self._qwen_groups[self.kv_cache_gid]
        if not torch.equal(primary.block_table_tensor, common.block_table_tensor):
            raise ValueError("Primary Qwen KV-group identity mismatch")

    def prepare_inputs(self, common_attn_metadata, sampled_token_ids, num_draft_tokens):
        self._require_capture(common_attn_metadata)
        if self._qwen_compacted:
            raise ValueError("Qwen metadata already compacted")
        if (len(sampled_token_ids) != 1 or len(num_draft_tokens) != 1
                or num_draft_tokens[0] not in (0, 1)
                or (num_draft_tokens[0] == 1 and len(sampled_token_ids[0]) not in (1, 2))):
            raise ValueError("Unchecked rejection rows are unsupported by Qwen K1")
        compact, rows = super().prepare_inputs(common_attn_metadata, sampled_token_ids, num_draft_tokens)
        # Native base computes the rejection gather used by runner IDs/HC/positions.
        # Apply that SAME gather to every group's slots, not merely its prefix.
        compact.max_query_len = compact.num_actual_tokens
        if common_attn_metadata.positions is not None:
            compact.positions = common_attn_metadata.positions.index_select(0, rows.long())
        self._validate_common(compact)
        for gid, cm in self._qwen_groups.items():
            group_cm = copy(compact)
            group_cm.block_table_tensor = cm.block_table_tensor
            group_cm.slot_mapping = cm.slot_mapping.index_select(0, rows.long())
            self._qwen_active[gid] = group_cm
        self._qwen_compacted = True
        return compact, rows

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata, draft_index=0):
        if draft_index != 0:
            raise ValueError("Qwen MTP supports only first-pass K1 metadata")
        self._validate_common(common_attn_metadata)
        if not self._qwen_compacted:
            self._require_capture(common_attn_metadata)
            self._qwen_active = dict(self._qwen_groups)
        elif not self._qwen_active or any(
                cm.num_actual_tokens != common_attn_metadata.num_actual_tokens
                or not torch.equal(cm.seq_lens, common_attn_metadata.seq_lens)
                for cm in self._qwen_active.values()):
            raise ValueError("Compacted Qwen metadata identity mismatch")
        per_group, per_layer = [], {}
        for group in self.draft_attn_groups:
            cm = self._qwen_active[group.kv_cache_group_id]
            md = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm, draft_index=0)
            per_group.append(md)
            for name in group.layer_names:
                per_layer[name] = md
        return per_group, per_layer

    def _get_slot_mapping(self, num_tokens, slot_mapping=None):
        if not self._qwen_active:
            raise ValueError("Qwen first-pass metadata not built")
        per_layer = {}
        for group in self.draft_attn_groups:
            cm = self._qwen_active[group.kv_cache_group_id]
            if num_tokens != cm.num_actual_tokens:
                raise ValueError("Qwen K1 does not allow padded forward rows")
            for name in group.layer_names:
                per_layer[name] = cm.slot_mapping[:num_tokens]
        return per_layer

    def set_inputs_first_pass(
            self, target_token_ids, next_token_ids, target_positions,
            target_hidden_states, token_indices_to_sample, cad,
            num_rejected_tokens_gpu):
        self._validate_common(cad)
        n = cad.num_actual_tokens
        expected_positions = ((self.draft_model_config.mrope_num_dims, n)
                              if self.uses_mrope else (n,))
        if (target_token_ids.shape != (n,) or next_token_ids.shape != (1,)
                or target_hidden_states.shape != (n, self.hidden_size)
                or target_positions.shape != expected_positions
                or num_rejected_tokens_gpu is not None
                or (token_indices_to_sample is not None
                    and token_indices_to_sample.tolist() != [n - 1])):
            raise ValueError("Qwen MTP K1 first-pass rows must be exactly unpadded and aligned")
        return super().set_inputs_first_pass(
            target_token_ids, next_token_ids, target_positions, target_hidden_states,
            token_indices_to_sample, cad, num_rejected_tokens_gpu)

    def _determine_batch_execution_and_padding(self, num_tokens, use_cudagraphs=True):
        # This profile is explicitly eager, unpadded and DP1.
        return CUDAGraphMode.NONE, num_tokens, None

    def propose(self, num_speculative_tokens, *args, **kwargs):
        if num_speculative_tokens != 1:
            raise ValueError("Qwen MTP runtime draft horizon must remain K1")
        try:
            return super().propose(num_speculative_tokens, *args, **kwargs)
        finally:
            # Never let a later request reuse captured block IDs or rejection rows.
            self.begin_qwen4_mtp_metadata()
