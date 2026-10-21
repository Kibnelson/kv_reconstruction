"""Versioned P1 head-sector pool: independent NumPy oracle and strict CUDA API.

The CUDA path writes into a preallocated bounded arena. Descriptor offsets
come from a GPU scan; no device scalar is read by encode/unpack. Capacity
failures are unpublished (-1) descriptors, never out-of-bounds raw fallbacks.
The arena is an append-batch prototype, not a reusable serving allocator.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

FORMAT = "kv_pool_head_sectors_v1"
ALIGNMENT = 16
DESC_FIELDS = ("offset", "allocated_bytes", "mode_raw0_packed1_invalid_minus1", "stored_escape_count")


@dataclass(frozen=True)
class Layout:
    layers: int
    tokens: int
    heads: int
    head_dim: int
    tile_tokens: int = 16

    def __post_init__(self):
        for k, v in vars(self).items():
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(k + " must be a positive integer")
        if self.head_dim < 8 or self.head_dim > 256 or self.head_dim & (self.head_dim - 1):
            raise ValueError("head_dim must be a power of two from 8 to 256")
        if self.tile_tokens > 32 or self.tile_tokens & (self.tile_tokens - 1):
            raise ValueError("tile_tokens must be a power of two at most 32")
        if self.sectors * 2 * self.sector_words >= 1 << 62:
            raise ValueError("layout exceeds bounded INT64 addressing")

    @property
    def channels(self): return 2 * self.heads * self.head_dim
    @property
    def pages(self): return (self.tokens + self.tile_tokens - 1) // self.tile_tokens
    @property
    def sectors(self): return self.layers * self.pages * 2 * self.heads
    @property
    def sector_words(self): return self.tile_tokens * self.head_dim
    @property
    def shape(self): return (self.layers, self.tokens, self.channels)
    @property
    def raw_bytes(self): return int(np.prod(self.shape)) * 2
    @property
    def padded_raw_bytes(self): return self.sectors * self.sector_words * 2

    def description(self):
        return dict(format=FORMAT, **vars(self), sector_order="layer,page,K_then_V_head",
                    dense_bit_order="lsb-first uint32 little endian", alignment=ALIGNMENT,
                    descriptor_fields=list(DESC_FIELDS), padded_tail_words=0)


@dataclass(frozen=True)
class Calibration:
    references: np.ndarray
    widths: np.ndarray
    sha256: str


def calibration(references, widths):
    refs = np.asarray(references); bits = np.asarray(widths)
    if refs.ndim != 2 or refs.shape != bits.shape or not refs.size:
        raise ValueError("matching nonempty [layer,combined channel] calibration required")
    if refs.dtype.kind not in 'iu' or bits.dtype.kind not in 'iu':
        raise ValueError("integer calibration required")
    if np.any(refs > 255) or np.any(refs < 0) or np.any(bits < 1) or np.any(bits > 8):
        raise ValueError("references must fit UINT8 and widths must be in [1,8]")
    refs = np.ascontiguousarray(refs, dtype=np.uint8).copy(); bits = np.ascontiguousarray(bits, dtype=np.uint8).copy()
    h = hashlib.sha256(FORMAT.encode() + str(refs.shape).encode() + refs.tobytes() + bits.tobytes()).hexdigest()
    refs.flags.writeable = False; bits.flags.writeable = False
    return Calibration(refs, bits, h)


def check_calibration(layout, state):
    if state.references.shape != (layout.layers, layout.channels) or state.widths.shape != state.references.shape:
        raise ValueError("calibration geometry differs from layout")


def sector_coordinates(layout, sid):
    if isinstance(sid, bool) or not 0 <= int(sid) < layout.sectors:
        raise ValueError("sector ID out of range")
    layer, remainder = divmod(int(sid), layout.pages * 2 * layout.heads)
    page, head = divmod(remainder, 2 * layout.heads)
    return layer, page, head


def original_tiles(words, layout, ids):
    words = np.asarray(words)
    if words.dtype != np.uint16 or tuple(words.shape) != layout.shape:
        raise ValueError("original words must be UINT16 in the declared layout")
    out = np.zeros((len(ids), layout.tile_tokens, layout.head_dim), np.uint16)
    for row, sid in enumerate(ids):
        l, p, h = sector_coordinates(layout, sid); begin = p * layout.tile_tokens
        n = min(layout.tile_tokens, layout.tokens - begin)
        out[row, :n] = words[l, begin:begin + n, h * layout.head_dim:(h + 1) * layout.head_dim]
    return out


@dataclass
class ReferencePacket:
    payload: np.ndarray
    descriptors: np.ndarray
    layout: Layout
    calibration_sha256: str


def reference_encode(words, state, layout, capacity_bytes=None):
    """Independent byte-oriented oracle. No GPU timing or GPU claim."""
    check_calibration(layout, state)
    tiles = original_tiles(words, layout, np.arange(layout.sectors))
    descriptors = np.zeros((layout.sectors, 4), np.int64); pieces = []; offset = 0
    for sid, tile in enumerate(tiles):
        l, p, h = sector_coordinates(layout, sid); d = layout.head_dim; n = layout.sector_words
        refs = state.references[l, h*d:(h+1)*d].astype(np.int32)
        widths = state.widths[l, h*d:(h+1)*d].astype(np.int32)
        valid_tokens = min(layout.tile_tokens, layout.tokens - p * layout.tile_tokens)
        original = tile.ravel(); exp = ((tile.astype(np.int32) >> 7) & 255)
        delta = exp - refs; zz = np.where(delta >= 0, 2*delta, -2*delta-1)
        sentinel = (1 << widths) - 1
        valid = np.arange(layout.tile_tokens)[:, None] < valid_tokens
        escaped = valid & (zz >= sentinel)
        code = np.where(valid, np.where(escaped, sentinel, zz), 0)
        dense_bytes = ((layout.tile_tokens * int(widths.sum()) + 31)//32)*4
        esc = exp[escaped].astype(np.uint8).tobytes()
        allocated = ((n + dense_bytes + len(esc) + 15)//16)*16
        if allocated >= 2*n:
            mode=0; esc_count=0; body=original.astype('<u2').tobytes(); allocated=2*n
        else:
            mode=1; esc_count=len(esc); dense=bytearray(dense_bytes); at=0
            for row in code:
                for value, width in zip(row, widths):
                    for b in range(int(width)):
                        if (int(value) >> b) & 1:dense[at//8] |= 1 << (at%8)
                        at+=1
            sm=(((original.astype(np.int32)>>8)&128)|(original&127)).astype(np.uint8).tobytes()
            body=sm+bytes(dense)+esc
            body+=b'\0'*(allocated-len(body))
        descriptors[sid]=[offset,allocated,mode,esc_count]; pieces.append(body); offset+=allocated
    if capacity_bytes is None:capacity_bytes=offset
    if capacity_bytes <= 0 or capacity_bytes % ALIGNMENT:
        raise ValueError("capacity must be a positive multiple of 16")
    payload=np.zeros(capacity_bytes,np.uint8)
    for sid,body in enumerate(pieces):
        off,size,_,_=map(int,descriptors[sid])
        if off+size>capacity_bytes:descriptors[sid,2]=-1
        else:payload[off:off+size]=np.frombuffer(body,np.uint8)
    return ReferencePacket(payload,descriptors,layout,state.sha256)


def reference_unpack(packet, state, ids):
    layout=packet.layout;check_calibration(layout,state)
    if packet.calibration_sha256!=state.sha256:raise ValueError("wrong calibration identity")
    out=np.zeros((len(ids),layout.tile_tokens,layout.head_dim),np.uint16)
    for index,sid in enumerate(ids):
        l,p,h=sector_coordinates(layout,sid);off,size,mode,escapes=map(int,packet.descriptors[int(sid)])
        n=layout.sector_words;valid_tokens=min(layout.tile_tokens,layout.tokens-p*layout.tile_tokens)
        if off<0 or off%16 or size<=0 or size%16 or off+size>len(packet.payload) or mode not in (0,1):
            raise ValueError("unpublished or malformed P1 sector")
        body=packet.payload[off:off+size]
        if mode==0:
            if size!=2*n or escapes:raise ValueError("invalid raw sector")
            values=np.frombuffer(body,dtype='<u2').copy()
        else:
            d=layout.head_dim;refs=state.references[l,h*d:(h+1)*d].astype(np.int32)
            widths=state.widths[l,h*d:(h+1)*d].astype(np.int32)
            dense_bytes=((layout.tile_tokens*int(widths.sum())+31)//32)*4
            if escapes<0 or escapes>n or ((n+dense_bytes+escapes+15)//16)*16!=size:raise ValueError("invalid packed sector length")
            dense=body[n:n+dense_bytes];at=0;rank=0;values=np.zeros(n,np.uint16)
            for i in range(n):
                w=int(widths[i%d]);code=0
                for b in range(w):code|=((int(dense[at//8])>>(at%8))&1)<<b;at+=1
                if i//d>=valid_tokens:continue
                if code==(1<<w)-1:
                    if rank>=escapes:raise ValueError("escape rank exceeds count")
                    exp=int(body[n+dense_bytes+rank]);rank+=1
                else:exp=int(refs[i%d])+(-((code+1)//2) if code&1 else code//2)
                if not 0<=exp<=255:raise ValueError("exponent outside BF16")
                sm=int(body[i]);values[i]=((sm&128)<<8)|(exp<<7)|(sm&127)
            if rank!=escapes:raise ValueError("escape count differs from sentinels")
        out[index]=values.reshape(layout.tile_tokens,layout.head_dim)
        out[index,valid_tokens:]=0
    return out


def _cuda_torch(device):
    try:import torch
    except ImportError as e:raise RuntimeError("CUDA PyTorch and Triton are required for P1 GPU kernels") from e
    selected=torch.device(device)
    if selected.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError("P1 requires an available CUDA device; no CPU fallback. Use the explicit NumPy reference for CPU validation.")
    if selected.index is None:selected=torch.device('cuda',torch.cuda.current_device())
    return torch,selected


@dataclass
class DeviceCalibration:
    references: object
    widths: object
    sha256: str


def to_cuda(state,device='cuda:0'):
    torch,device=_cuda_torch(device)
    return DeviceCalibration(torch.from_numpy(state.references.copy()).to(device),
                             torch.from_numpy(state.widths.copy()).to(device),state.sha256)


@dataclass
class Arena:
    payload: object
    descriptors: object
    scan_workspace: object
    layout: Layout
    calibration_sha256: str


def allocate_arena(layout,state,capacity_bytes):
    torch,device=_cuda_torch(state.references.device)
    if state.references.shape!=(layout.layers,layout.channels):raise ValueError("wrong calibration geometry")
    if isinstance(capacity_bytes,bool) or not isinstance(capacity_bytes,int) or capacity_bytes<=0 or capacity_bytes%16:
        raise ValueError("capacity must be a positive multiple of 16")
    return Arena(torch.empty(capacity_bytes,dtype=torch.uint8,device=device),
                 torch.empty((layout.sectors,4),dtype=torch.int64,device=device),
                 torch.empty(layout.sectors,dtype=torch.int64,device=device),layout,state.sha256)


def _check_device_metadata(torch,state,arena):
    g=arena.layout;dev=arena.payload.device
    if arena.payload.dtype!=torch.uint8 or arena.payload.ndim!=1 or not arena.payload.is_contiguous() or not arena.payload.numel() or arena.payload.numel()%16:
        raise ValueError("invalid P1 arena tensor")
    if arena.descriptors.dtype!=torch.int64 or tuple(arena.descriptors.shape)!=(g.sectors,4) or not arena.descriptors.is_contiguous() or arena.descriptors.device!=dev:
        raise ValueError("invalid P1 directory tensor")
    if arena.scan_workspace.dtype!=torch.int64 or tuple(arena.scan_workspace.shape)!=(g.sectors,) or not arena.scan_workspace.is_contiguous() or arena.scan_workspace.device!=dev:
        raise ValueError("invalid P1 scan workspace")
    for a in [state.references,state.widths]:
        if a.dtype!=torch.uint8 or tuple(a.shape)!=(g.layers,g.channels) or not a.is_contiguous() or a.device!=dev:
            raise ValueError("invalid P1 calibration tensor")


def encode_into(words,state,arena):
    """Enqueue count, scan, and pack into an owned arena. No host scalar reads.

    Descriptors with mode -1 are NOT published; caller must retain the source.
    A production cache controller/attention kernel must consume their status.
    """
    torch,_=_cuda_torch(arena.payload.device);g=arena.layout;_check_device_metadata(torch,state,arena)
    if tuple(words.shape)!=g.shape or words.dtype!=torch.uint16 or not words.is_contiguous():
        raise ValueError("contiguous original UINT16 words in the declared layout required")
    if words.device!=arena.payload.device or state.references.device!=words.device or state.widths.device!=words.device:
        raise ValueError("inputs, calibration and arena must already share the CUDA device")
    if state.sha256!=arena.calibration_sha256:raise ValueError("wrong calibration identity")
    from . import pool_p1_kernels as k
    constants=dict(T=g.tokens,C=g.channels,HEADS2=2*g.heads,D=g.head_dim,
                   TILE=g.tile_tokens,PAGES=g.pages,N=g.sector_words)
    with torch.cuda.device(words.device):
        k.count_sectors[(g.sectors,)](words,state.references,state.widths,arena.descriptors,**constants,num_warps=4)
        torch.cumsum(arena.descriptors[:,1],dim=0,out=arena.scan_workspace)
        k.place_offsets[((g.sectors+255)//256,)](arena.descriptors,arena.scan_workspace,g.sectors,256,num_warps=4)
        k.write_sectors[(g.sectors,)](words,state.references,state.widths,arena.descriptors,
            arena.payload,arena.payload.view(torch.uint16),arena.payload.view(torch.int32),
            CAPACITY=arena.payload.numel(),**constants,num_warps=4)
    return arena


def unpack_tiles(arena,state,sector_ids):
    """GPU head-tile reads; output/status only. No original words are inputs."""
    torch,_=_cuda_torch(arena.payload.device);g=arena.layout;_check_device_metadata(torch,state,arena)
    if sector_ids.device!=arena.payload.device or sector_ids.dtype!=torch.int64 or sector_ids.ndim!=1 or not sector_ids.is_contiguous():
        raise ValueError("contiguous INT64 sector IDs must already be on the CUDA device")
    if state.sha256!=arena.calibration_sha256 or state.references.device!=arena.payload.device or state.widths.device!=arena.payload.device:
        raise ValueError("wrong calibration identity or device")
    if not sector_ids.numel():raise ValueError("at least one sector ID required")
    from . import pool_p1_kernels as k
    out=torch.empty((sector_ids.numel(),g.tile_tokens,g.head_dim),dtype=torch.uint16,device=arena.payload.device)
    status=torch.empty(sector_ids.numel(),dtype=torch.int32,device=arena.payload.device)
    with torch.cuda.device(arena.payload.device):
        k.unpack_sectors[(sector_ids.numel(),)](arena.payload,arena.payload.view(torch.uint16),
            arena.payload.view(torch.int32),arena.descriptors,state.references,state.widths,sector_ids,out,status,
            CAPACITY=arena.payload.numel(),SECTORS=g.sectors,T=g.tokens,C=g.channels,
            HEADS2=2*g.heads,D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,N=g.sector_words,num_warps=4)
    return out,status


def arena_statistics(arena,state):
    """Synchronizing diagnostics OUTSIDE codec timing. Never an allocator input."""
    d=arena.descriptors.detach().cpu().numpy();g=arena.layout
    required=int(np.sum(d[:,1]));valid=d[:,2]>=0
    return dict(required_payload_bytes=required,arena_capacity_bytes=arena.payload.numel(),
        descriptor_bytes=arena.descriptors.numel()*arena.descriptors.element_size(),
        encode_scan_workspace_bytes=arena.scan_workspace.numel()*arena.scan_workspace.element_size(),
        calibration_device_bytes=state.references.numel()+state.widths.numel(),
        published_sectors=int(np.count_nonzero(valid)),unpublished_sectors=int(np.count_nonzero(~valid)),
        packed_sectors=int(np.count_nonzero(d[:,2]==1)),raw_sectors=int(np.count_nonzero(d[:,2]==0)),
        stored_escape_bytes=int(d[valid,3].sum()),
        tail_padding_words=(g.pages*g.tile_tokens-g.tokens)*g.layers*g.channels,
        resident_packet_allocated_bytes=int(arena.payload.numel()+arena.descriptors.numel()*8),
        payload_required_bpv=required*8/int(np.prod(g.shape)),
        resident_packet_allocated_bpv=(arena.payload.numel()+arena.descriptors.numel()*8)*8/int(np.prod(g.shape)),
        scope="owned arena capacity plus directory is resident allocation; scan is encoder workspace; original input, calibration and output tiles are separate; no attention kernel")
