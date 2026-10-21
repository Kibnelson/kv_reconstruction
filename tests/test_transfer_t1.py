import importlib.util
import os
import unittest
import numpy as np
from qwen_kv_bench import transfer_t1_reference as oracle
from qwen_kv_bench import transfer_reference as ref,transfer_probability as prob
from qwen_kv_bench import transfer_t1 as t1
from tests.test_pool_attention import bits
try:
    import torch
except ImportError:
    torch=None


def model_and_words():
    rng=np.random.default_rng(901)
    def capture(n):
        return dict(K_bits=bits(rng.normal(scale=.25,size=(2,n,8))),V_bits=bits(rng.normal(scale=.25,size=(2,n,8))),Hk=np.array(1),hd=np.array(8),base=np.array(10000.))
    base=ref.fit([capture(40)],capture(32),block_size=8,max_positions=64)
    state=prob.empty_state(base,'B')
    state.coefficients[:]=rng.integers(-2048,2049,state.coefficients.shape)
    state.digest();test=capture(1)
    return base,state,ref.words(test),test


def check_input_chunk_equivalence(test,device,interpreter=False):
    base,state,words,_=model_and_words();e=t1.Engine(base,state,device,_interpreter=interpreter)
    identity=e.state_sha256;rng=np.random.default_rng(101)
    # Exercise column counts beyond both candidate boundaries independently
    # of the small causal fixture. Compare integer products with NumPy.
    features=rng.integers(-(1<<24),1<<24,(3,769),dtype=np.int64)
    source=torch.from_numpy(features).to(device)
    for g in [0,1]:
        old=e.p.predictors[0][g]
        coefficients=rng.integers(-65536,65537,(8,769),dtype=np.int32)
        bias=rng.integers(-4096,4097,8,dtype=np.int64)
        e.p.predictors[0][g]=dict(old,A=torch.from_numpy(coefficients).to(device),
            columns=torch.arange(769,dtype=torch.int64,device=device),b=torch.from_numpy(bias).to(device))
        expected=np.clip(((features@coefficients.astype(np.int64).T+(1<<15))>>16)+bias,-e.feature_limit,e.feature_limit)
        if g==0:
            co,si=base.cosine[2:5],base.sine[2:5];a,b=expected[:,::2].copy(),expected[:,1::2].copy()
            expected[:,::2]=(a*co-b*si+(1<<14))>>15
            expected[:,1::2]=(a*si+b*co+(1<<14))>>15
        for chunk in [64,512,0]:
            e.input_chunk=chunk
            np.testing.assert_array_equal(e._mean(source,0,g,2).cpu().numpy(),expected)
        e.p.predictors[0][g]=old
    # Complete stream and cross-implementation packet-only decoding use the
    # unmodified, validated causal fixture and original source words.
    source=torch.from_numpy(words).to(device);e.input_chunk=64
    original=e.encode(source,raw_fallback=False)
    for chunk in [512,0]:
        e.input_chunk=chunk;packet=e.encode(source,raw_fallback=False)
        np.testing.assert_array_equal(packet.payload.cpu().numpy(),original.payload.cpu().numpy())
        np.testing.assert_array_equal(packet.directory.cpu().numpy(),original.directory.cpu().numpy())
        decoded,error,_=e.decode(original)
        np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(decoded.cpu().numpy(),words)
        test.assertEqual(e.state_sha256,identity)


def check_entropy_cache_equivalence(test,device,interpreter=False):
    from qwen_kv_bench import transfer_t1_kernels as k
    base,state,words,_=model_and_words();e=t1.Engine(base,state,device,_interpreter=interpreter)
    rng=np.random.default_rng(102);words=rng.integers(0,65536,(2,3,16),dtype=np.uint16)
    words[0,0,:8]=[0,32768,0x7f80,0xff80,0x7fc1,0xffff,1,32769]
    source=torch.from_numpy(words).to(device);records=e._walk(source,0)
    ranks=rng.integers(0,65536,257,dtype=np.int64);probe_words=oracle.word_of(ranks)
    probe_mu=rng.normal(size=257);probe_inverse=2.**rng.uniform(-16,16,257)
    probe_args=[torch.from_numpy(x.copy()).to(device) for x in [probe_words,probe_mu,probe_inverse]]
    probe=torch.empty((1,257,16),dtype=torch.uint16,device=device)
    k.prepare_frequencies[(17,)](*probe_args,e.boundaries,e.table,probe,T=1,C=257,LAYER=0,CH=0,GROUP=257,BLOCK=256,num_warps=4,enable_fp_fusion=False)
    for plane in range(16):
        prefix=ranks&(65535^((1<<(plane+1))-1))
        np.testing.assert_array_equal(probe.cpu().numpy()[0,:,plane],oracle.frequency(prefix,plane,probe_mu,probe_inverse))
    l,g,mu,inverse,_=records[-1];cache=torch.empty((3,8,16),dtype=torch.uint16,device=device)
    k.prepare_frequencies[(2,)](source,mu,inverse,e.boundaries,e.table,cache,T=3,C=16,LAYER=l,CH=g*8,GROUP=8,BLOCK=256,num_warps=4,enable_fp_fusion=False)
    rank=oracle.ordered(words[l,:,g*8:g*8+8]);actual=cache.cpu().numpy()
    for plane in range(16):
        prefix=rank&(65535^((1<<(plane+1))-1))
        np.testing.assert_array_equal(actual[:,:,plane],oracle.frequency(prefix,plane,mu.cpu().numpy(),inverse.cpu().numpy()))
    original=e.encode(source,raw_fallback=False);identity=e.state_sha256
    wire=t1.export_packet(original,{'model':'synthetic'})
    e.entropy_backend='cached';cached=e.encode(source,raw_fallback=False)
    test.assertEqual(cached.encoder_probability_cache_bytes,3*8*16*2)
    test.assertEqual(t1.export_packet(cached,{'model':'synthetic'}),wire)
    for packet in [original,cached]:
        out,error,_=e.decode(packet)
        np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),words)
    e.entropy_backend='scalar';out,error,_=e.decode(cached)
    np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),words)
    test.assertEqual(e.state_sha256,identity)
    cached.directory[1,1]=3;e.entropy_backend='cached'
    _,error,_=e.decode(cached);test.assertNotEqual(error[1].item(),0)


class ReferenceTests(unittest.TestCase):
    def test_entropy_backend_is_explicit(self):
        base,state,words,capture=model_and_words()
        with self.assertRaises(ValueError):t1.Engine(base,state,'cpu',entropy_backend='unknown')
    def test_input_chunk_requires_nonnegative_integer(self):
        base,state,words,capture=model_and_words()
        for invalid in [-1,True,1.5,'512']:
            with self.assertRaises(ValueError):t1.Engine(base,state,'cpu',input_chunk=invalid)
    def test_identity_digest_binds_complete_canonical_manifest(self):
        identity={'model':'example','config':{'heads':8,'dimension':128}}
        reordered={'config':{'dimension':128,'heads':8},'model':'example'}
        self.assertEqual(t1.model_identity_digest(identity),t1.model_identity_digest(reordered))
        changed={'model':'example','config':{'heads':4,'dimension':128}}
        self.assertNotEqual(t1.model_identity_digest(identity),t1.model_identity_digest(changed))
        with self.assertRaises(ValueError):t1.model_identity_digest({})

    def test_special_words_and_adversarial_probabilities_roundtrip(self):
        words=np.array([0,32768,1,32769,0x7f80,0xff80,0x7fc1,0xffff,0x3f80,0xbf80],np.uint16)
        mu=np.linspace(-100,100,len(words));inv=2.**np.linspace(-8,16,len(words))
        data=oracle.encode_lane(words,mu,inv)
        np.testing.assert_array_equal(oracle.decode_lane(data,mu,inv),words)
        with self.assertRaises(ValueError):oracle.decode_lane(data[:-1],mu,inv)
        with self.assertRaises(ValueError):oracle.decode_lane(data+b'\0',mu,inv)

    def test_every_word_has_positive_binary_frequencies(self):
        ranks=np.arange(65536,dtype=np.int64)
        for plane in range(16):
            prefix=ranks&(65535^((1<<(plane+1))-1))
            f=oracle.frequency(prefix,plane,0.,4.)
            self.assertTrue(np.all((f>=1)&(f<=4095)))

    def test_public_api_refuses_cpu_entropy_fallback(self):
        base,state,words,capture=model_and_words()
        with self.assertRaises(RuntimeError):t1.Engine(base,state,'cpu')


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1' and torch is not None and importlib.util.find_spec('triton') is not None
CUDA=torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None


@unittest.skipUnless(INTERPRETER,'explicit CPU kernel interpreter only')
class InterpreterTests(unittest.TestCase):
    def test_cached_probabilities_bytes_and_cross_backend_decode_are_exact(self):
        check_entropy_cache_equivalence(self,'cpu',True)
    def test_input_chunk_reductions_and_complete_packets_are_exact(self):
        check_input_chunk_equivalence(self,'cpu',True)
    def engine(self):
        base,state,words,capture=model_and_words()
        return t1.Engine(base,state,'cpu',_interpreter=True),words,capture

    def test_gpu_frequency_source_matches_original_integer_table(self):
        from qwen_kv_bench.transfer_t1_kernels import probe_frequency
        rng=np.random.default_rng(76);n=257
        rank=rng.integers(0,65536,n,dtype=np.int32);mu=rng.normal(size=n);inv=2.**rng.uniform(-16,16,n)
        args=[torch.from_numpy(x.copy()) for x in [rank,mu,inv,oracle.BD,oracle.F_TAB]]
        out=torch.empty(n,dtype=torch.int64)
        for plane in [0,3,7,15]:
            probe_frequency[((n+255)//256,)](*args,out,COUNT=n,PLANE=plane,BLOCK=256)
            prefix=rank&(65535^((1<<(plane+1))-1))
            np.testing.assert_array_equal(out.numpy(),oracle.frequency(prefix,plane,mu,inv))

    def test_causal_parameters_packet_bytes_and_packet_only_decode(self):
        e,w,capture=self.engine();source=torch.from_numpy(w)
        records=e._walk(source,0);rows=prob.trace(e.base,capture,e.adapter)
        for (_,_,mu,inv,off),r in zip(records,rows):
            expected_off=prob.offsets(e.adapter,r['context'],r['layer'],r['group'])
            np.testing.assert_array_equal(mu.numpy(),r['mu'])
            np.testing.assert_array_equal(off.numpy(),expected_off)
            np.testing.assert_array_equal(inv.numpy(),r['inverse']*e.adapter.scale_table[expected_off+prob.MAX_OFFSET,None])
        packet=e.encode(source,raw_fallback=False)
        prepared=e._encode_prepared(source,0,records,raw_fallback=False)
        np.testing.assert_array_equal(prepared.payload.numpy(),packet.payload.numpy())
        np.testing.assert_array_equal(prepared.directory.numpy(),packet.directory.numpy())
        np.testing.assert_array_equal(packet.encoder_error.numpy(),0)
        means=np.concatenate([r[2].numpy()[0] for r in records]);inverses=np.concatenate([r[3].numpy()[0] for r in records])
        expected=oracle.encode_lane(w[:,0].ravel(),means,inverses)
        self.assertEqual(packet.payload.numpy().tobytes(),expected)
        del source
        recovered,error,decoded_records=e.decode(packet,trace=True)
        np.testing.assert_array_equal(error.numpy(),0);np.testing.assert_array_equal(recovered.numpy(),w)
        for enc,dec in zip(records,decoded_records):
            for a,b in zip(enc[2:],dec[2:]):np.testing.assert_array_equal(a.numpy(),b.numpy())
        wire=t1.export_packet(packet,{'model':'synthetic software check'})
        imported=t1.import_packet(wire,e,{'model':'synthetic software check'})
        out,err,_=e.decode(imported);np.testing.assert_array_equal(err.numpy(),0);np.testing.assert_array_equal(out.numpy(),w)
        with self.assertRaises(ValueError):t1.import_packet(wire[:-1]+bytes([wire[-1]^1]),e,{'model':'synthetic software check'})
        with self.assertRaises(ValueError):t1.import_packet(wire,e,{'model':'different model'})
        self.check_identity_envelopes(e,packet,w)

    def check_identity_envelopes(self,e,packet,w):
        identity={'model':'synthetic','complete_config':'x'*6000}
        compact=t1.export_packet(packet,identity)
        legacy=t1.export_packet(packet,identity,identity_mode='full')
        counts=t1.packet_accounting(compact);old_counts=t1.packet_accounting(legacy)
        self.assertEqual(counts['identity_mode'],'sha256');self.assertEqual(old_counts['identity_mode'],'full')
        self.assertLess(len(compact),len(legacy)-5800)
        self.assertEqual(counts['complete_bytes'],counts['payload_bytes']+counts['directory_bytes']+counts['envelope_bytes'])
        for data in [compact,legacy]:
            imported=t1.import_packet(data,e,identity)
            np.testing.assert_array_equal(imported.payload.cpu().numpy(),packet.payload.cpu().numpy())
            np.testing.assert_array_equal(imported.directory.cpu().numpy(),packet.directory.cpu().numpy())
            out,error,_=e.decode(imported)
            np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),w)
            with self.assertRaises(ValueError):t1.import_packet(data,e,{'model':'synthetic','complete_config':'y'*6000})
        # Even with a valid outer checksum, an ambiguous identity is rejected.
        import hashlib,json
        magic,mn,dn,pn,digest=t1.HEADER.unpack(compact[:t1.HEADER.size]);body=compact[t1.HEADER.size:]
        meta=json.loads(body[:mn]);meta['model_identity']=identity;encoded=prob.canonical(meta)
        changed=encoded+body[mn:]
        ambiguous=t1.HEADER.pack(magic,len(encoded),dn,pn,hashlib.sha256(changed).digest())+changed
        with self.assertRaises(ValueError):t1.import_packet(ambiguous,e,identity)

    def test_raw_fallback_and_invalid_lane_bounds_are_explicit(self):
        e,w,capture=self.engine();source=torch.from_numpy(w)
        raw=e.encode(source,raw_fallback=True)
        self.assertTrue(raw.raw)
        out,error,_=e.decode(raw);np.testing.assert_array_equal(out.numpy(),w);np.testing.assert_array_equal(error.numpy(),0)
        coded=e.encode(source,raw_fallback=False);coded.directory[0,1]=3
        _,error,_=e.decode(coded);self.assertNotEqual(error[0].item(),0)

    def test_parallel_lanes_preserve_arbitrary_original_words(self):
        e,w,capture=self.engine();rng=np.random.default_rng(87)
        w=rng.integers(0,65536,size=(2,3,16),dtype=np.uint16)
        w[0,0,:8]=[0,32768,0x7f80,0xff80,0x7fc1,0xffff,1,32769]
        source=torch.from_numpy(w);records=e._walk(source,0);packet=e.encode(source,raw_fallback=False)
        for lane in range(3):
            mu=np.concatenate([r[2].numpy()[lane] for r in records]);inv=np.concatenate([r[3].numpy()[lane] for r in records])
            off,size=packet.directory[lane].tolist()
            self.assertEqual(packet.payload[off:off+size].numpy().tobytes(),oracle.encode_lane(w[:,lane].ravel(),mu,inv))
        out,error,_=e.decode(packet);np.testing.assert_array_equal(error.numpy(),0);np.testing.assert_array_equal(out.numpy(),w)


@unittest.skipUnless(CUDA,'actual CUDA hardware unavailable')
class CudaTests(unittest.TestCase):
    def test_cached_probabilities_bytes_and_cross_backend_decode_are_exact(self):
        check_entropy_cache_equivalence(self,'cuda')
    def test_input_chunk_reductions_and_complete_packets_are_exact(self):
        check_input_chunk_equivalence(self,'cuda')
    def test_gpu_entropy_core_preserves_all_65536_BF16_words(self):
        from qwen_kv_bench import transfer_t1_kernels as k
        tokens=32;channels=2048;group=32;cap=32*channels+16
        w=np.arange(65536,dtype=np.uint16).reshape(1,tokens,channels)
        source=torch.from_numpy(w).cuda();mu=torch.zeros((tokens,group),dtype=torch.float64,device='cuda');inv=torch.ones_like(mu)
        bd=torch.from_numpy(oracle.BD.copy()).cuda();ft=torch.from_numpy(oracle.F_TAB.copy()).cuda()
        reference=None
        for cached in [False,True]:
            xs=torch.full((tokens,),1<<23,dtype=torch.int64,device='cuda');count=torch.zeros_like(xs);err=torch.zeros(tokens,dtype=torch.int32,device='cuda')
            buffer=torch.empty((tokens,cap),dtype=torch.uint8,device='cuda');cache=torch.empty((tokens,group,16),dtype=torch.uint16,device='cuda')
            for ch in reversed(range(0,channels,group)):
                if cached:
                    k.prepare_frequencies[(64,)](source,mu,inv,bd,ft,cache,T=tokens,C=channels,LAYER=0,CH=ch,GROUP=group,BLOCK=256,num_warps=4,enable_fp_fusion=False)
                    k.encode_group_cached[(tokens,)](source,cache,xs,count,buffer,err,T=tokens,C=channels,LAYER=0,CH=ch,GROUP=group,CAP=cap,num_warps=1)
                else:k.encode_group[(tokens,)](source,mu,inv,bd,ft,xs,count,buffer,err,T=tokens,C=channels,LAYER=0,CH=ch,GROUP=group,CAP=cap,num_warps=1,enable_fp_fusion=False)
            ends=torch.cumsum(count+4,0);data=torch.empty(int(ends[-1].item()),dtype=torch.uint8,device='cuda');directory=torch.empty((tokens,2),dtype=torch.int64,device='cuda')
            k.compact_lanes[(tokens,)](buffer,xs,count,ends,directory,data,CAP=cap,BLOCK=256,num_warps=4)
            np.testing.assert_array_equal(err.cpu().numpy(),0)
            if reference is None:reference=(data.cpu().numpy().copy(),directory.cpu().numpy().copy())
            else:
                np.testing.assert_array_equal(data.cpu().numpy(),reference[0]);np.testing.assert_array_equal(directory.cpu().numpy(),reference[1])
            position=torch.empty_like(xs);k.initialize_decoder[(tokens,)](data,directory,xs,position,err,SIZE=data.numel(),num_warps=1)
            out=torch.empty_like(source)
            for ch in range(0,channels,group):
                k.decode_group[(tokens,)](data,directory,xs,position,err,mu,inv,bd,ft,out,T=tokens,C=channels,LAYER=0,CH=ch,GROUP=group,SIZE=data.numel(),REUSE_MASS=cached,num_warps=1,enable_fp_fusion=False)
            k.finish_decoder[(tokens,)](xs,position,directory,err,num_warps=1)
            np.testing.assert_array_equal(err.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),w)

    def test_actual_gpu_binary_counts_match_cpu_table(self):
        from qwen_kv_bench.transfer_t1_kernels import probe_frequency
        rng=np.random.default_rng(76);n=257
        rank=rng.integers(0,65536,n,dtype=np.int32);mu=rng.normal(size=n);inv=2.**rng.uniform(-16,16,n)
        args=[torch.from_numpy(x.copy()).cuda() for x in [rank,mu,inv,oracle.BD,oracle.F_TAB]]
        out=torch.empty(n,dtype=torch.int64,device='cuda')
        for plane in [0,3,7,15]:
            probe_frequency[((n+255)//256,)](*args,out,COUNT=n,PLANE=plane,BLOCK=256,num_warps=4,enable_fp_fusion=False)
            prefix=rank&(65535^((1<<(plane+1))-1))
            np.testing.assert_array_equal(out.cpu().numpy(),oracle.frequency(prefix,plane,mu,inv))

    def test_actual_gpu_entropy_stream_is_exact_and_matches_cpu_byte_oracle(self):
        base,state,w,capture=model_and_words();e=t1.Engine(base,state)
        records=e._walk(torch.from_numpy(w).cuda(),0)
        packet=e.encode(torch.from_numpy(w).cuda(),raw_fallback=False)
        means=np.concatenate([r[2].cpu().numpy()[0] for r in records]);inverses=np.concatenate([r[3].cpu().numpy()[0] for r in records])
        self.assertEqual(packet.payload.cpu().numpy().tobytes(),oracle.encode_lane(w[:,0].ravel(),means,inverses))
        out,error,_=e.decode(packet);np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),w)
        InterpreterTests.check_identity_envelopes(self,e,packet,w)

    def test_parallel_gpu_lanes_preserve_special_words(self):
        base,state,w,capture=model_and_words();e=t1.Engine(base,state)
        w=np.random.default_rng(87).integers(0,65536,size=(2,3,16),dtype=np.uint16)
        w[0,0,:8]=[0,32768,0x7f80,0xff80,0x7fc1,0xffff,1,32769]
        packet=e.encode(torch.from_numpy(w).cuda(),raw_fallback=False)
        out,error,_=e.decode(packet);np.testing.assert_array_equal(error.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),w)


if __name__=='__main__':unittest.main()
