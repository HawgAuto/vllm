"""Private, same-precision MoE warp policy; default disabled."""
QUALIFIED_TOKENS=frozenset((960,1568,3136,10976))

def select_configs(config, *, enabled, tokens, bf16, block_shape, w4, w8):
    # Leave every unsupported/native decode/MTP geometry byte-for-byte alone.
    qualified=(enabled and tokens in QUALIFIED_TOKENS and bf16
        and tuple(block_shape or ())==(0,128) and bool(w4)!=bool(w8)
        and all(config.get(k,default)==value for k,default,value in (
          ('BLOCK_SIZE_M',None,64),('BLOCK_SIZE_N',64,64),
          ('BLOCK_SIZE_K',32,32),('GROUP_SIZE_M',None,1),
          ('SPLIT_K',1,1),('num_warps',4,4),('num_stages',2,2))))
    if not qualified:return config,config
    down=dict(config,num_warps=2)
    return (dict(config,num_warps=2) if w4 else config),down
