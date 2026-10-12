"""Minimal HIP code-object loader/launcher (ctypes) sharing torch's HIP runtime and current stream."""
import ctypes, glob, os
import torch

_lib = None


def lib():
    global _lib
    if _lib is None:
        cands = sorted(glob.glob('/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib/libamdhip64.so*'))
        if not cands:
            raise RuntimeError('libamdhip64 not found')
        torch.cuda.init()
        _lib = ctypes.CDLL(cands[0])
        _lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [ctypes.c_uint, ctypes.c_void_p,
                                                                                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        _lib.hipFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    return _lib


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f'{what} failed: hipError {rc}')


class Module:
    def __init__(self, path):
        self.blob = open(path, 'rb').read()
        self._buf = ctypes.create_string_buffer(self.blob, len(self.blob))
        self.mod = ctypes.c_void_p()
        check(lib().hipModuleLoadData(ctypes.byref(self.mod), self._buf), 'hipModuleLoadData')
        self.funcs = {}

    def fn(self, name):
        if name not in self.funcs:
            f = ctypes.c_void_p()
            check(lib().hipModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), f'hipModuleGetFunction({name})')
            self.funcs[name] = f
        return self.funcs[name]

    def attrs(self, name):
        # HIP_FUNC_ATTRIBUTE_*: 0 max threads, 1 shared size, 3 local size, 4 num regs
        out = {}
        for k, i in (('max_threads', 0), ('shared_bytes', 1), ('local_bytes', 3), ('num_regs', 4)):
            v = ctypes.c_int()
            if lib().hipFuncGetAttribute(ctypes.byref(v), i, self.fn(name)) == 0:
                out[k] = v.value
        return out

    def launch(self, name, grid, block, args, shared=0, stream=None):
        """args: list of (ctype_value). Tensors must be passed as ctypes.c_void_p(t.data_ptr())."""
        holders = [a if isinstance(a, ctypes._SimpleCData) else ctypes.c_void_p(a) for a in args]
        ptrs = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders])
        s = torch.cuda.current_stream().cuda_stream if stream is None else stream
        g = tuple(grid) + (1,) * (3 - len(grid)); b = tuple(block) + (1,) * (3 - len(block))
        check(lib().hipModuleLaunchKernel(self.fn(name), g[0], g[1], g[2], b[0], b[1], b[2], shared, ctypes.c_void_p(s), ptrs, None),
              f'launch {name}')


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())
