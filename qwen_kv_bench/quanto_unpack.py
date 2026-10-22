"""Quanto's pinned native unpack kernels behind a small, stream-aware C ABI.

Preserves both GPU kernel bodies. Only host dispatch changes: input device and
current Torch CUDA stream are explicit. No Torch C++/Python development headers,
unused AWQ/Marlin GEMMs, installed-package edits, or CPU fallback are needed.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

UPSTREAM_SHA256='f7001704be6504f4f8a0a67f588aba60c2dc13f942ed9d87e1e15fc18777e204'
_LIB=None
_METADATA=None


def patched_source(original):
    if hashlib.sha256(original).hexdigest()!=UPSTREAM_SHA256:
        raise ValueError('Quanto native unpack source differs from pinned 0.2.6 wheel')
    kernels=re.findall(r'__global__ void unpack_\dbit_kernel\([^}]+\}',original.decode())
    if len(kernels)!=2:raise ValueError('Quanto native kernel definitions changed')
    # Keep the original license header and GPU functions byte-for-byte. The
    # upstream Torch allocation/stream-0 host functions are replaced by a C ABI.
    license_header=original.decode().split('#include',1)[0]
    return license_header+'''#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#define BLOCK_SIZE 256
'''+ '\n\n'.join(kernels)+'''
extern "C" int qwen_quanto_unpack(
    uint64_t input, uint64_t output, uint64_t count, int bits,
    uint64_t stream_handle, int device) {
    if ((bits != 2 && bits != 4) || count > INT_MAX / (8 / bits)) return cudaErrorInvalidValue;
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess || count == 0) return status;
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    auto source = reinterpret_cast<unsigned char*>(input);
    auto destination = reinterpret_cast<unsigned char*>(output);
    int n = static_cast<int>(count);
    int blocks = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (bits == 2) {
        unpack_2bit_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(source, destination, n);
    } else {
        unpack_4bit_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(source, destination, n);
    }
    return cudaGetLastError();
}
extern "C" const char* qwen_quanto_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
'''


def build_unpack(original, root, nvcc, arch):
    """Compile/link without requiring GPU execution or Torch development files."""
    source=patched_source(original)
    compiler=subprocess.check_output([str(nvcc),'--version'],text=True)
    identity=hashlib.sha256((source+compiler+str(arch)).encode()).hexdigest()
    root=Path(root)/identity[:16];root.mkdir(parents=True,exist_ok=True)
    cu=root/'unpack_compat.cu';library=root/'libquanto_unpack.so';manifest=root/'build.json'
    source_hash=hashlib.sha256(source.encode()).hexdigest()
    if library.is_file() and manifest.is_file():
        metadata=json.loads(manifest.read_text())
        if (metadata['compiled_source_sha256']==source_hash and metadata['library_sha256']==
                hashlib.sha256(library.read_bytes()).hexdigest()):return library,metadata
    cu.write_text(source);temporary=library.with_suffix('.part.so')
    command=[str(nvcc),'-std=c++17','-O3','--shared','-Xcompiler=-fPIC',
        '-gencode',f'arch=compute_{arch},code=sm_{arch}',str(cu),'-o',str(temporary)]
    print('Building native Quanto unpack only (current CUDA stream; unchanged GPU kernels).',flush=True)
    r=subprocess.run(command,capture_output=True,text=True)
    (root/'compile.json').write_text(json.dumps(dict(command=command,returncode=r.returncode,
        stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    if r.returncode:raise RuntimeError('native Quanto unpack compile failed: '+r.stderr[-3000:])
    temporary.replace(library)
    metadata=dict(backend='native Quanto unpack only',quanto='0.2.6',upstream_source_sha256=UPSTREAM_SHA256,
        compiled_source_sha256=source_hash,gpu_kernel_bodies='unchanged upstream 2/4-bit kernels',
        host_compatibility='C ABI; current Torch CUDA stream and input device are explicit',
        compiler=compiler,architecture=str(arch),command=command,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(metadata,indent=2)+'\n');return library,metadata


def ensure_cuda_unpack():
    global _LIB,_METADATA
    import torch
    if not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET')=='1':
        return dict(backend='upstream CPU; no GPU measurement')
    if _LIB is not None:return _METADATA
    from optimum.quanto.library.extensions import get_extension
    extension=get_extension('quanto_cuda')
    upstream=next(Path(p) for p in extension.sources if Path(p).name=='unpack.cu')
    cuda_home=os.environ.get('CUDA_HOME')
    if not cuda_home:raise RuntimeError('set CUDA_HOME or run the comparison helper to prepare local CUDA tools')
    arch=''.join(map(str,torch.cuda.get_device_capability()))
    path,metadata=build_unpack(upstream.read_bytes(),Path(__file__).resolve().parents[1]/'runs/_matched_baselines/quanto_unpack',
        Path(cuda_home)/'bin/nvcc',arch)
    native=C.CDLL(str(path))
    native.qwen_quanto_unpack.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_int,C.c_uint64,C.c_int]
    native.qwen_quanto_unpack.restype=C.c_int
    native.qwen_quanto_error.argtypes=[C.c_int];native.qwen_quanto_error.restype=C.c_char_p

    def unpack(tensor,bits):
        if tensor.dtype!=torch.uint8 or not tensor.is_cuda or not tensor.is_contiguous() or bits not in (2,4):
            raise ValueError('native Quanto unpack expects contiguous CUDA uint8 and 2/4 bits')
        if tensor.numel()*(8//bits)>2**31-1:raise ValueError('native Quanto unpack output exceeds upstream int32 bounds')
        with torch.cuda.device(tensor.device):
            stream=torch.cuda.current_stream(tensor.device)
            output=torch.empty((tensor.shape[0]*(8//bits),*tensor.shape[1:]),device=tensor.device,dtype=torch.uint8)
            status=native.qwen_quanto_unpack(tensor.data_ptr(),output.data_ptr(),tensor.numel(),bits,
                int(stream.cuda_stream),tensor.device.index)
            if status:raise RuntimeError('native Quanto unpack: '+native.qwen_quanto_error(status).decode())
            tensor.record_stream(stream)
        return output

    # The released CUDA dispatch calls extension.lib.unpack. Keep that
    # dispatch, quantizer, packed tensors and dequantizer; bind only unpack.
    # Group size 64 does not select the omitted AWQ/Marlin weight GEMMs.
    shim=SimpleNamespace(unpack=unpack,native_library=native)
    extension._lib=shim;_LIB=shim;_METADATA=dict(metadata,torch=torch.__version__,cuda_home=cuda_home)
    return _METADATA
