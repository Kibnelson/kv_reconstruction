"""T1 feasibility kernels: binary logistic probabilities and per-token rANS.

The stream is a new ABI, not constriction/T0 bytes. Binary probabilities use
the saved BF16 boundaries and logistic table, 12-bit conditional frequencies,
and a positive uniform word prior. No exp/libm call constructs probabilities.
The standard byte-rANS equations are described in
https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h .
This implementation favors transparent correctness over throughput tuning.
"""
import triton
import triton.language as tl


@triton.jit
def _mass(edge, mean, inverse, BD, FT):
    boundary = tl.load(BD+edge)
    standardized = (boundary-mean)*inverse
    coordinate = tl.minimum(tl.maximum((standardized+32.)*256.,0.),16384.)
    index = tl.minimum(tl.floor(coordinate).to(tl.int32),16383)
    fraction = tl.floor((coordinate-index)*65536.).to(tl.int64)
    a = tl.load(FT+index); b = tl.load(FT+index+1)
    cdf = a+(((b-a)*fraction)>>16)
    return cdf*65535+edge.to(tl.int64)*65536


@triton.jit
def frequency(prefix, plane, mean, inverse, BD, FT):
    low = _mass(prefix,mean,inverse,BD,FT)
    mid = _mass(prefix+(1<<plane),mean,inverse,BD,FT)
    high = _mass(prefix+(1<<(plane+1)),mean,inverse,BD,FT)
    count = ((mid-low)*4096)//(high-low)
    return tl.minimum(tl.maximum(count,1),4095)


@triton.jit
def probe_frequency(RANK, MU, INV, BD, FT, OUT, COUNT:tl.constexpr, PLANE:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    rank=tl.load(RANK+i,i<COUNT,other=0).to(tl.int32)
    mu=tl.load(MU+i,i<COUNT,other=0);inv=tl.load(INV+i,i<COUNT,other=1)
    prefix=rank & (65535 ^ ((1<<(PLANE+1))-1))
    f=frequency(prefix,PLANE,mu,inv,BD,FT)
    tl.store(OUT+i,f,i<COUNT)


@triton.jit(do_not_specialize=['LAYER','CH'])
def encode_group(W,MU,INV,BD,FT,STATE,COUNT,BUFFER,ERROR,
                 T:tl.constexpr,C:tl.constexpr,LAYER,CH,
                 GROUP:tl.constexpr,CAP:tl.constexpr):
    lane=tl.program_id(0)
    state=tl.load(STATE+lane);used=tl.load(COUNT+lane);error=tl.load(ERROR+lane)
    for j in range(GROUP-1,-1,-1):
        word=tl.load(W+(LAYER*T+lane)*C+CH+j).to(tl.int32)
        rank=tl.where((word&32768)!=0,65535^word,32768^word)
        mu=tl.load(MU+lane*GROUP+j);inv=tl.load(INV+lane*GROUP+j)
        for plane in range(16):
            prefix=rank & (65535 ^ ((1<<(plane+1))-1))
            f0=frequency(prefix,plane,mu,inv,BD,FT)
            bit=(rank>>plane)&1
            f=tl.where(bit==0,f0,4096-f0);start=tl.where(bit==0,0,f0)
            limit=f*524288
            while state>=limit:
                tl.store(BUFFER+lane*CAP+used,(state&255).to(tl.uint8),used<CAP)
                error=error | tl.where(used<CAP,0,1)
                used+=1;state=state>>8
            state=(state//f)*4096+(state%f)+start
    tl.store(STATE+lane,state);tl.store(COUNT+lane,used);tl.store(ERROR+lane,error)


@triton.jit
def compact_lanes(BUFFER,STATE,COUNT,ENDS,DIRECTORY,DATA,
                  CAP:tl.constexpr,BLOCK:tl.constexpr):
    lane=tl.program_id(0)
    count=tl.load(COUNT+lane);length=count+4
    end=tl.load(ENDS+lane);offset=end-length
    tl.store(DIRECTORY+lane*2,offset);tl.store(DIRECTORY+lane*2+1,length)
    i=tl.arange(0,4);state=tl.load(STATE+lane)
    tl.store(DATA+offset+i,((state>>(i*8))&255).to(tl.uint8))
    for begin in range(0,count,BLOCK):
        j=begin+tl.arange(0,BLOCK)
        source=count-1-j
        value=tl.load(BUFFER+lane*CAP+source,(j<count)&(source>=0)&(source<CAP),other=0)
        tl.store(DATA+offset+4+j,value,j<count)


@triton.jit(do_not_specialize=['SIZE'])
def initialize_decoder(DATA,DIRECTORY,STATE,POSITION,ERROR,SIZE):
    lane=tl.program_id(0)
    offset=tl.load(DIRECTORY+lane*2);length=tl.load(DIRECTORY+lane*2+1)
    valid=(offset>=0)&(length>=4)&(offset<=SIZE)&(length<=SIZE-offset)
    state=tl.full((),0,tl.int64)
    for j in range(4):
        byte=tl.load(DATA+offset+j,valid,other=0).to(tl.int64)
        state=state|(byte<<(8*j))
    valid=valid&(state>=8388608)&(state<2147483648)
    tl.store(STATE+lane,state);tl.store(POSITION+lane,4)
    tl.store(ERROR+lane,tl.where(valid,0,1))


@triton.jit(do_not_specialize=['LAYER','CH','SIZE'])
def decode_group(DATA,DIRECTORY,STATE,POSITION,ERROR,MU,INV,BD,FT,OUT,
                 T:tl.constexpr,C:tl.constexpr,LAYER,CH,
                 GROUP:tl.constexpr,SIZE):
    lane=tl.program_id(0)
    offset=tl.load(DIRECTORY+lane*2);length=tl.load(DIRECTORY+lane*2+1)
    state=tl.load(STATE+lane);pos=tl.load(POSITION+lane);error=tl.load(ERROR+lane)
    valid=(offset>=0)&(length>=4)&(offset<=SIZE)&(length<=SIZE-offset)&(error==0)
    # Invalid streams execute no decoding loop. Their outputs are zero and the
    # error status remains nonzero; no raw/original tensor is consulted.
    if valid:
        for j in range(GROUP):
            rank=tl.full((),0,tl.int32)
            mu=tl.load(MU+lane*GROUP+j);inv=tl.load(INV+lane*GROUP+j)
            for step in range(16):
                plane=15-step
                f0=frequency(rank,plane,mu,inv,BD,FT)
                slot=state&4095;bit=(slot>=f0).to(tl.int32)
                f=tl.where(bit==0,f0,4096-f0);start=tl.where(bit==0,0,f0)
                state=f*(state>>12)+slot-start
                rank=rank|(bit<<plane)
                # At most two bytes are needed for a valid 12-bit symbol.
                # Bounded iteration prevents a malformed stream from hanging.
                for renorm in range(2):
                    if state<8388608:
                        available=pos<length
                        byte=tl.load(DATA+offset+pos,available,other=0).to(tl.int64)
                        error=error|tl.where(available,0,1)
                        pos+=1;state=(state<<8)|byte
                error=error|tl.where(state>=8388608,0,1)
            word=tl.where((rank&32768)!=0,rank^32768,rank^65535)
            tl.store(OUT+(LAYER*T+lane)*C+CH+j,word.to(tl.uint16))
    else:
        for j in range(GROUP):tl.store(OUT+(LAYER*T+lane)*C+CH+j,0)
    tl.store(STATE+lane,state);tl.store(POSITION+lane,pos);tl.store(ERROR+lane,error)


@triton.jit
def finish_decoder(STATE,POSITION,DIRECTORY,ERROR):
    lane=tl.program_id(0)
    state=tl.load(STATE+lane);pos=tl.load(POSITION+lane);length=tl.load(DIRECTORY+lane*2+1)
    old=tl.load(ERROR+lane)
    tl.store(ERROR+lane,old|tl.where((state==8388608)&(pos==length),0,1))
