"""Explicit native Quark file-to-file INT4 -> pre-shuffle GPTQ wire bridge.

No quantization, serving dispatch, or kernel implementation lives here. Callers
must bind the producer image/source where direct_quantize_checkpoint passes
reorder=False to Pack_4_bits; JSON's export.pack_method alone is insufficient.
"""
import torch

WIRE_ID = 'quark-file2file-int4-no-reorder-KN8-to-GPTQ-K8N-v1'


def validate_export_config(config):
    q = config.get('quantization_config', {})
    g = q.get('global_quant_config', {})
    w = g.get('weight', {})
    expected = {'dtype': 'int4', 'qscheme': 'per_group', 'group_size': 128,
                'ch_axis': -1, 'symmetric': True, 'is_dynamic': False}
    if q.get('quant_method') != 'quark' or q.get('version') != '0.12.post1' or any(w.get(k) != v for k,v in expected.items()):
        raise ValueError('only explicit symmetric static Quark INT4 group-128 is supported')
    if g.get('input_tensors') is not None or g.get('output_tensors') is not None:
        raise ValueError('activation quantization is outside this wire contract')
    if q.get('layer_quant_config') or q.get('layer_type_quant_config'):
        raise ValueError('per-layer overrides require separate resolution')
    e=q.get('export', {})
    if e.get('weight_format') != 'real_quantized' or e.get('pack_method') != 'reorder':
        raise ValueError('unknown Quark export wire metadata')
    if not {'mtp.*', 'model.visual.*'}.issubset(q.get('exclude', [])):
        raise ValueError('protected consumer-visible policy is missing')


def decode_quark_int4(packed):
    if packed.ndim != 2 or packed.dtype != torch.int32 or packed.numel() == 0:
        raise ValueError('Quark weight must be nonempty int32 [K,N/8]')
    q=torch.stack([(packed >> (4*i)) & 15 for i in range(8)], dim=-1)
    q=q.reshape(packed.shape[0], packed.shape[1]*8)
    return torch.where(q >= 8, q-16, q).to(torch.int8)


def convert_projection(packed, source_scales):
    codes=decode_quark_int4(packed)
    k,n=codes.shape
    if k % 128 or source_scales.shape != (n,k//128):
        raise ValueError('native file-to-file source requires F32 scales [N,K/128]')
    if source_scales.dtype != torch.float32 or source_scales.device != packed.device:
        raise ValueError('this producer contract requires colocated F32 scales')
    if not bool((torch.isfinite(source_scales) & (source_scales > 0)).all()):
        raise ValueError('source scales must be positive and finite')
    scales=source_scales.to(torch.bfloat16).contiguous()
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError('scale rounding overflow/underflow')
    relative=((scales.float()-source_scales).abs()/source_scales).max().item()
    if relative > 0.004:
        raise ValueError('BF16 scale rounding exceeds the declared bound')
    # Native consumer subtracts offset 8; its shuffle is deliberately separate.
    output=torch.zeros((k//8,n),dtype=torch.int32,device=packed.device)
    for i in range(8):
        output |= (codes[i::8].to(torch.int32)+8) << (4*i)
    return output.contiguous(), scales.T.contiguous(), {
        'wire_id': WIRE_ID, 'integer_codes_preserved': True,
        'scale_input_dtype': 'float32', 'scale_output_dtype': 'bfloat16',
        'scale_max_relative_error': relative,
        'scale_max_absolute_error': (scales.float()-source_scales).abs().max().item(),
    }
