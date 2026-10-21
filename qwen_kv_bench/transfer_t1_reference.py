"""Independent NumPy/Python oracle for the T1 binary CDF and rANS ABI."""
import numpy as np
from .transfer_coset import BD,F_TAB,cdf_int,ordered,word_of

LOW=1<<23
PRECISION=12
TOTAL=1<<PRECISION


def frequency(prefix,plane,mu,inverse,table=F_TAB):
    prefix=np.asarray(prefix,dtype=np.int64)
    mu=np.asarray(mu,dtype=np.float64);inverse=np.asarray(inverse,dtype=np.float64)
    def mass(edge):
        c=cdf_int((BD[edge]-mu)*inverse,table)
        return c*65535+edge*65536
    a=mass(prefix);b=mass(prefix+(1<<plane));c=mass(prefix+(1<<(plane+1)))
    return np.clip(((b-a)*TOTAL)//(c-a),1,TOTAL-1)


def encode_lane(words,mu,inverse,table=F_TAB):
    words=np.asarray(words,np.uint16).ravel();mu=np.asarray(mu,np.float64).ravel();inverse=np.asarray(inverse,np.float64).ravel()
    if words.shape!=mu.shape or mu.shape!=inverse.shape:raise ValueError('parameter/word shape mismatch')
    ranks=ordered(words);frequencies=[]
    for plane in range(16):
        prefix=ranks&(65535^((1<<(plane+1))-1))
        frequencies.append(frequency(prefix,plane,mu,inverse,table))
    state=LOW;emitted=bytearray()
    for i in range(len(words)-1,-1,-1):
        for plane in range(16):
            bit=(int(ranks[i])>>plane)&1;f0=int(frequencies[plane][i])
            f=f0 if not bit else TOTAL-f0;start=0 if not bit else f0
            limit=((LOW>>PRECISION)<<8)*f
            while state>=limit:emitted.append(state&255);state>>=8
            state=(state//f)*TOTAL+state%f+start
    return state.to_bytes(4,'little')+bytes(reversed(emitted))


def decode_lane(data,mu,inverse,table=F_TAB):
    data=bytes(data);mu=np.asarray(mu,np.float64).ravel();inverse=np.asarray(inverse,np.float64).ravel()
    if len(data)<4 or mu.shape!=inverse.shape:raise ValueError('truncated stream or wrong parameters')
    state=int.from_bytes(data[:4],'little');pos=4
    if not LOW<=state<LOW*256:raise ValueError('invalid initial rANS state')
    ranks=[]
    for mean,inv in zip(mu,inverse):
        rank=0
        for plane in range(15,-1,-1):
            f0=int(frequency(rank,plane,mean,inv,table));slot=state&(TOTAL-1)
            bit=int(slot>=f0);f=f0 if not bit else TOTAL-f0;start=0 if not bit else f0
            state=f*(state>>PRECISION)+slot-start;rank|=bit<<plane
            while state<LOW:
                if pos>=len(data):raise ValueError('truncated rANS stream')
                state=(state<<8)|data[pos];pos+=1
        ranks.append(rank)
    if state!=LOW or pos!=len(data):raise ValueError('rANS stream termination mismatch')
    return word_of(np.asarray(ranks))
