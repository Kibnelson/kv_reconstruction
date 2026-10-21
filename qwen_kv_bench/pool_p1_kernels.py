"""P1 CUDA kernels. No Torch, CPU KV arithmetic, or host-sized allocation.

One program owns a [token tile, KV head] sector. Packed exponent fields are
LSB-first, written into initialized UINT32 words using disjoint atomic ORs.
Escape ranks are cooperative prefix scans inside that sector. This is a
pool pack/tile-unpack prototype, not an attention or T entropy kernel.
"""
import triton
import triton.language as tl


@triton.jit
def count_sectors(W, R, B, DESC, T: tl.constexpr, C: tl.constexpr,
                  HEADS2: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr,
                  PAGES: tl.constexpr, N: tl.constexpr):
    sid = tl.program_id(0)
    layer = sid // (PAGES * HEADS2)
    page = (sid // HEADS2) % PAGES
    head = sid % HEADS2
    i = tl.arange(0, N)
    token = page * TILE + i // D
    channel = head * D + i % D
    valid = token < T
    word = tl.load(W + (layer.to(tl.int64) * T + token) * C + channel,
                   mask=valid, other=0).to(tl.int32)
    ref = tl.load(R + layer * C + channel).to(tl.int32)
    width = tl.load(B + layer * C + channel).to(tl.int32)
    delta = ((word >> 7) & 255) - ref
    zz = tl.where(delta >= 0, 2 * delta, -2 * delta - 1)
    escaped = valid & (zz >= ((1 << width) - 1))
    escapes = tl.sum(escaped.to(tl.int32), 0)
    dense_bytes = ((tl.sum(width, 0) + 31) // 32) * 4
    packed = ((N + dense_bytes + escapes + 15) // 16) * 16
    mode = packed < 2 * N
    size = tl.where(mode, packed, 2 * N)
    tl.store(DESC + sid * 4, 0)
    tl.store(DESC + sid * 4 + 1, size)
    tl.store(DESC + sid * 4 + 2, mode.to(tl.int64))
    tl.store(DESC + sid * 4 + 3, tl.where(mode, escapes, 0))


@triton.jit
def place_offsets(DESC, PREFIX, SECTORS: tl.constexpr, BLOCK: tl.constexpr):
    sid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size = tl.load(DESC + sid * 4 + 1, sid < SECTORS, other=0)
    end = tl.load(PREFIX + sid, sid < SECTORS, other=0)
    tl.store(DESC + sid * 4, end - size, sid < SECTORS)


@triton.jit
def write_sectors(W, R, B, DESC, A8, A16, A32,
                  CAPACITY: tl.constexpr, T: tl.constexpr, C: tl.constexpr,
                  HEADS2: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr,
                  PAGES: tl.constexpr, N: tl.constexpr):
    sid = tl.program_id(0)
    layer = sid // (PAGES * HEADS2)
    page = (sid // HEADS2) % PAGES
    head = sid % HEADS2
    base = tl.load(DESC + sid * 4)
    size = tl.load(DESC + sid * 4 + 1)
    mode = tl.load(DESC + sid * 4 + 2)
    fit = (base >= 0) & (base + size <= CAPACITY)
    i = tl.arange(0, N)
    token = page * TILE + i // D
    channel = head * D + i % D
    valid = token < T
    word = tl.load(W + (layer.to(tl.int64) * T + token) * C + channel,
                   mask=valid, other=0).to(tl.int32)
    if fit:
        if mode == 0:
            tl.store(A16 + base // 2 + i, word.to(tl.uint16))
        else:
            ref = tl.load(R + layer * C + channel).to(tl.int32)
            width = tl.load(B + layer * C + channel).to(tl.int32)
            delta = ((word >> 7) & 255) - ref
            zz = tl.where(delta >= 0, 2 * delta, -2 * delta - 1)
            sentinel = (1 << width) - 1
            escaped = valid & (zz >= sentinel)
            code = tl.where(valid, tl.where(escaped, sentinel, zz), 0).to(tl.uint32)
            bits = tl.cumsum(width, 0) - width
            dense_words = (tl.sum(width, 0) + 31) // 32
            dense_base = (base + N) // 4
            tl.store(A32 + dense_base + i, 0, i < dense_words)
            sm = (((word >> 8) & 128) | (word & 127)).to(tl.uint8)
            tl.store(A8 + base + i, sm)
            escapes = tl.load(DESC + sid * 4 + 3)
            tail = N + dense_words * 4 + escapes
            pad = tl.arange(0, 16)
            tl.store(A8 + base + tail + pad, 0, tail + pad < size)
            # Each program owns its entire sector; all zero stores above must
            # precede atomic ORs by any participating thread in this program.
            tl.debug_barrier()
            shift = bits & 31
            tl.atomic_or(A32 + dense_base + bits // 32, code << shift,
                         sem="relaxed", scope="cta")
            cross = shift + width > 32
            tl.atomic_or(A32 + dense_base + bits // 32 + 1,
                         code >> ((32 - shift) & 31), mask=cross,
                         sem="relaxed", scope="cta")
            rank = tl.cumsum(escaped.to(tl.int32), 0) - 1
            tl.store(A8 + base + N + dense_words * 4 + rank,
                     ((word >> 7) & 255).to(tl.uint8), escaped)
    else:
        # Unpublished sector. All writes are suppressed; the original source
        # remains owned by the caller. A tile reader reports this as invalid.
        tl.store(DESC + sid * 4 + 2, -1)


@triton.jit
def unpack_sectors(A8, A16, A32, DESC, R, B, IDS, OUT, STATUS,
                   CAPACITY: tl.constexpr, SECTORS: tl.constexpr,
                   T: tl.constexpr, C: tl.constexpr, HEADS2: tl.constexpr,
                   D: tl.constexpr, TILE: tl.constexpr, PAGES: tl.constexpr,
                   N: tl.constexpr):
    out_id = tl.program_id(0)
    sid = tl.load(IDS + out_id)
    id_ok = (sid >= 0) & (sid < SECTORS)
    base = tl.load(DESC + sid * 4, id_ok, other=-1)
    size = tl.load(DESC + sid * 4 + 1, id_ok, other=0)
    mode = tl.load(DESC + sid * 4 + 2, id_ok, other=-1)
    escapes = tl.load(DESC + sid * 4 + 3, id_ok, other=-1)
    basic = id_ok & (base >= 0) & (base % 16 == 0) & (size > 0) & (size % 16 == 0)
    basic = basic & (base + size <= CAPACITY) & ((mode == 0) | (mode == 1))
    i = tl.arange(0, N)
    result = tl.full((N,), 0, tl.int32)
    good = tl.full((), False, tl.int1)
    if basic:
        layer = sid // (PAGES * HEADS2)
        page = (sid // HEADS2) % PAGES
        head = sid % HEADS2
        valid = page * TILE + i // D < T
        if mode == 0:
            if (size == 2 * N) & (escapes == 0):
                result = tl.load(A16 + base // 2 + i).to(tl.int32)
                result = tl.where(valid, result, 0)
                good = tl.full((), True, tl.int1)
        else:
            channel = head * D + i % D
            width = tl.load(B + layer * C + channel).to(tl.int32)
            bits = tl.cumsum(width, 0) - width
            dense_words = (tl.sum(width, 0) + 31) // 32
            need = ((N + dense_words * 4 + escapes + 15) // 16) * 16
            layout_ok = (escapes >= 0) & (escapes <= N) & (size == need)
            if layout_ok:
                dense_base = (base + N) // 4
                shift = bits & 31
                lo = tl.load(A32 + dense_base + bits // 32).to(tl.uint32)
                cross = shift + width > 32
                hi = tl.load(A32 + dense_base + bits // 32 + 1,
                             cross, other=0).to(tl.uint32)
                code = ((lo >> shift) | (hi << ((32 - shift) & 31))) & ((1 << width) - 1)
                escaped = valid & (code == ((1 << width) - 1))
                rank = tl.cumsum(escaped.to(tl.int32), 0) - 1
                raw_exp = tl.load(A8 + base + N + dense_words * 4 + rank,
                                  escaped & (rank < escapes), other=0).to(tl.int32)
                ref = tl.load(R + layer * C + channel).to(tl.int32)
                signed_code = code.to(tl.int32)
                delta = tl.where((signed_code & 1) == 1, -((signed_code + 1) >> 1), signed_code >> 1)
                exp = tl.where(escaped, raw_exp, ref + delta)
                sm = tl.load(A8 + base + i).to(tl.int32)
                result = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
                good = (tl.sum(escaped.to(tl.int32), 0) == escapes)
                good = good & (tl.sum((valid & ((exp < 0) | (exp > 255))).to(tl.int32), 0) == 0)
                result = tl.where(valid & good, result, 0)
    tl.store(OUT + out_id * N + i, tl.where(good, result, 0).to(tl.uint16))
    tl.store(STATUS + out_id, tl.where(good, 0, 1).to(tl.int32))
