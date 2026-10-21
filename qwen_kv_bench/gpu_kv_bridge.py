"""Bounded-chunk P1/T1 bridges and compressed-byte filesystem controls.

The GPU bridge currently materializes one at-most-32-token chunk of words.
It is a composed decode/pack path, not a fused two-layer-scratch converter.
"""
import hashlib
import os
from pathlib import Path
import time
from . import pool_p1 as p


def _geometry(engine,layout):
    g=engine.base.geometry
    if (g['layers'],2*g['channels'],g['heads'],g['head_dim'])!=(layout.layers,layout.channels,layout.heads,layout.head_dim):
        raise ValueError('transfer/pool geometry mismatch')
    if layout.tokens>32:raise ValueError('bridge accepts one bounded T1 chunk at a time')


def encode_pool(engine,arena,state,start=0,*,raw_fallback=True):
    """P1 -> T1; original KV is not an input."""
    _geometry(engine,arena.layout)
    words,status=unpack_words(arena,state)
    packet=engine.encode(words,start,raw_fallback=raw_fallback)
    packet.encoder_error=packet.encoder_error|status.any().to(engine.t.int32)
    return packet


def unpack_words(arena,state):
    """P1 -> one bounded canonical raw chunk; a transfer baseline control."""
    from .transfer_gpu_predictor import _require_torch
    t=_require_torch();g=arena.layout
    if g.tokens>32:raise ValueError('bounded bridge control accepts at most 32 tokens')
    ids=t.arange(g.sectors,dtype=t.int64,device=arena.payload.device)
    tiles,status=p.unpack_tiles(arena,state,ids)
    words=tiles.reshape(g.layers,g.pages,2*g.heads,g.tile_tokens,g.head_dim).permute(0,1,3,2,4).reshape(g.layers,g.pages*g.tile_tokens,g.channels)[:,:g.tokens].contiguous()
    return words,status


def pack_words(words,state,layout,capacity_bytes):
    """Canonical raw chunk -> P1; matches the bridge's allocation boundary."""
    arena=p.allocate_arena(layout,state,capacity_bytes)
    p.encode_into(words,state,arena)
    return arena


def install_pool(engine,packet,state,layout,capacity_bytes):
    """T1 -> P1; one bounded word chunk is temporary, then released."""
    _geometry(engine,layout)
    if tuple(packet.shape)!=layout.shape:raise ValueError('packet/pool chunk shape mismatch')
    words,error,_=engine.decode(packet)
    arena=p.allocate_arena(layout,state,capacity_bytes)
    p.encode_into(words,state,arena)
    # Never publish a bad decode as a valid resident pool. This is a GPU
    # status dependency, not a CPU scalar branch or a silent CPU repair.
    arena.descriptors[:,2]=engine.t.where(error.any(),-1,arena.descriptors[:,2])
    return arena,error


def filesystem_roundtrip(data,path):
    """Store/read the identical T bytes. Does not drop the OS page cache."""
    data=bytes(data);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    write_seconds=time.perf_counter()-start
    start=time.perf_counter();recovered=path.read_bytes();read_seconds=time.perf_counter()-start
    if recovered!=data:raise AssertionError('external T bytes changed')
    st=path.stat()
    return recovered,dict(file_bytes=st.st_size,filesystem_allocated_bytes=st.st_blocks*512,
        write_and_file_directory_fsync_seconds=write_seconds,read_seconds=read_seconds,
        sha256=hashlib.sha256(data).hexdigest(),byte_identical=True,
        scope='Local filesystem read/write; OS page cache allowed. Not a cold SSD or real network bandwidth measurement.')
