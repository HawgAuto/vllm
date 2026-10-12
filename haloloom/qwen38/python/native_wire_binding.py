"""Lossless Quark file2file integer layout conversion; no quantization."""

import torch

def convert_native(w,s,*,n,k,bits,group):
 if w.device.type!='cpu' or s.device.type!='cpu':raise ValueError('CPU source packet only')
 if s.dtype!=torch.float32 or tuple(s.shape)!=(n,k//group) or k%group:raise ValueError('scale shape/dtype/group')
 if not bool((torch.isfinite(s)&(s>0)).all()):raise ValueError('invalid scales')
 if bits==8:
  if w.dtype!=torch.int8 or tuple(w.shape)!=(k,n):raise ValueError('INT8 source signedKN')
  return (w.T.to(torch.int16)+128).to(torch.uint8).contiguous(),s.contiguous()
 if bits!=4 or w.dtype!=torch.int32 or tuple(w.shape)!=(k,(n+7)//8) or k%2:raise ValueError('INT4 nativegeometry')
 q=torch.stack([(w>>(4*j))&15 for j in range(8)],-1).reshape(k,-1)
 if n%8 and bool((q[:,n:]!=0).any()):raise ValueError('nonzero native output padding')
 q=q[:,:n];signed=torch.where(q>=8,q-16,q).T;u=(signed+8).to(torch.uint8)
 return (u[:,0::2]|(u[:,1::2]<<4)).contiguous(),s.contiguous()
