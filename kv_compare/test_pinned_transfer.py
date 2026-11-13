"""Packet identity, receive isolation and benchmark accounting; no GPU speed claims."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import head_spread_wire as wire
from . import native_compare as native
from .pinned_transfer import HostPacket,PinnedParts,PinnedT
from scripts.bench_t_transfer_path import aggregate,decision,run_fresh,run_retrieval


class PartsAdapter:
    device=torch.device("cpu")
    def __init__(self,parts,aux=None):self.fields,self.aux=parts,aux or {}
    def parts(self,encoded):return self.fields,self.aux
    def receive(self,parts,meta):return parts


class PinnedTests(unittest.TestCase):
    def setUp(self):
        self.saved=os.environ.get("TRITON_INTERPRET")
        os.environ["TRITON_INTERPRET"]="1"
    def tearDown(self):
        if self.saved is None:os.environ.pop("TRITON_INTERPRET",None)
        else:os.environ["TRITON_INTERPRET"]=self.saved

    def test_all_bf16_words_preserve_canonical_frame_and_raw_receive(self):
        words=torch.arange(65536,dtype=torch.int64).to(torch.int16).reshape(2,128,256)
        adapter=native.Raw(words.shape,"cpu")
        digest=hashlib.sha256(words.numpy().tobytes()).hexdigest()
        legacy,_=native.pack_parts("raw_bf16",words.shape,"a"*64,digest,[("words",words)])
        path=PinnedParts(adapter,"raw_bf16",words.shape,"a"*64,len(legacy),software=True)
        packet=path.pack(adapter.encode(words),digest)
        self.assertEqual(packet.view(),legacy)
        path.poison_receiver()
        encoded,expected=path.receive(packet)
        self.assertEqual(expected,digest)
        self.assertTrue(torch.equal(adapter.decode(encoded),words))
        self.assertNotEqual(encoded.data_ptr(),words.data_ptr())

    def test_multiple_dtypes_empty_noncontiguous_and_odd_byte_offsets(self):
        fields=[("u8",torch.tensor([1,2,3],dtype=torch.uint8)),
            ("u16",torch.arange(6,dtype=torch.int16).view(torch.uint16)),
            ("i16",torch.arange(12,dtype=torch.int16).reshape(3,4).t()),
            ("i32",torch.tensor([-1,0,2**30],dtype=torch.int32)),
            ("i64",torch.tensor([-2**40,2**40],dtype=torch.int64)),
            ("empty",torch.empty(0,7,dtype=torch.int16))]
        adapter=PartsAdapter(fields,{"tail_values":3})
        old,_=native.pack_parts("exans",[9],"a"*64,"b"*64,fields,adapter.aux)
        path=PinnedParts(adapter,"exans",[9],"a"*64,len(old),software=True)
        packet=path.pack(None,"b"*64)
        self.assertEqual(packet.view(),old)
        result,_=path.receive(packet)
        for name,t in fields:
            self.assertEqual(result[name].dtype,t.dtype)
            self.assertEqual(result[name].shape,t.shape)
            self.assertTrue(torch.equal(result[name],t))

    def test_reused_buffers_restore_a_previous_packet_without_sender_data(self):
        adapter=PartsAdapter([("words",torch.arange(31,dtype=torch.int16))])
        old,_=native.pack_parts("raw_bf16",[31],"a"*64,"b"*64,adapter.fields)
        path=PinnedParts(adapter,"raw_bf16",[31],"a"*64,len(old)+100,software=True)
        first=bytes(path.pack(None,"b"*64).view())
        original=adapter.fields[0][1].clone()
        adapter.fields[0][1].fill_(-999)
        path.pack(None,"c"*64)
        path.poison_receiver()
        received,digest=path.receive(HostPacket.copy_from(first,pin=False))
        self.assertEqual(digest,"b"*64)
        self.assertTrue(torch.equal(received["words"],original))

    def test_corruption_truncation_wrong_state_and_trailing_bytes_rejected(self):
        adapter=PartsAdapter([("words",torch.arange(7,dtype=torch.int16))])
        old,_=native.pack_parts("raw_bf16",[7],"a"*64,"b"*64,adapter.fields)
        path=PinnedParts(adapter,"raw_bf16",[7],"a"*64,len(old)+100,software=True)
        for bad in (old[:-1],old+b"x",old[:-1]+bytes([old[-1]^1])):
            with self.assertRaises(ValueError):path.receive(HostPacket.copy_from(bad,pin=False))
        other=PinnedParts(adapter,"raw_bf16",[7],"c"*64,len(old),software=True)
        with self.assertRaises(ValueError):other.receive(HostPacket.copy_from(old,pin=False))

    def test_capacity_limit_does_not_silently_resize_or_truncate(self):
        adapter=PartsAdapter([("words",torch.arange(64,dtype=torch.int16))])
        path=PinnedParts(adapter,"raw_bf16",[64],"a"*64,100,software=True)
        with self.assertRaisesRegex(ValueError,"preallocated capacity"):path.pack(None,"b"*64)
        with self.assertRaises(ValueError):HostPacket(torch.zeros(3,dtype=torch.uint8),4)

    def test_rechecksummed_malicious_descriptor_cannot_overflow(self):
        adapter=PartsAdapter([("words",torch.zeros(1,dtype=torch.int16))])
        old,_=native.pack_parts("raw_bf16",[1],"a"*64,"b"*64,adapter.fields)
        _,hlen,_=native.PREFIX.unpack_from(old)
        meta=json.loads(old[native.PREFIX.size:native.PREFIX.size+hlen])
        meta["parts"][0]["shape"]=[2**32,2**32]
        header=json.dumps(meta).encode();body=header+old[native.PREFIX.size+hlen:]
        forged=native.PREFIX.pack(native.MAGIC,len(header),hashlib.sha256(body).digest())+body
        path=PinnedParts(adapter,"raw_bf16",[1],"a"*64,len(forged),software=True)
        with self.assertRaises(ValueError):path.receive(HostPacket.copy_from(forged,pin=False))

    def test_rechecksummed_unaligned_T_directory_rejected(self):
        from .head_spread import Policy
        codec=SimpleNamespace(device=torch.device("cpu"),layout=lambda t:(2,152),L=1,W=16,C=8,
            hd=8,TB=8,ng=1,mode="uniform",policy=Policy())
        body=struct.pack("<2I",25<<3,23<<3)+bytes(96)
        header=wire.HEADER.pack(wire.MAGIC,1,1,8,16,8,8,8,1,0,2,48,0,
            bytes.fromhex("a"*64),bytes.fromhex(codec.policy.digest()),bytes.fromhex("b"*64),hashlib.sha256(body).digest())
        path=PinnedT(codec,8,"a"*64,len(header+body),software=True)
        with self.assertRaisesRegex(ValueError,"unaligned"):path.receive(HostPacket.copy_from(header+body,pin=False))

    def test_global_and_learned_T_changed_input_parity_and_original_receiver(self):
        # Run the Triton interpreter in its own process even on GPU test hosts.
        script=r'''
import hashlib,math
import numpy as np
import torch
from kv_compare.head_spread import HeadSpreadCodec,Policy
from kv_compare.fast_dense_encode import FastDenseEncode
from kv_compare import head_spread_wire as w
from kv_compare.pinned_transfer import PinnedT,HostPacket
L,T,hd,G=2,8,8,4;W=hd*G;C=W//2
gen=torch.Generator().manual_seed(730)
k=dict(Aq=torch.randint(-2,3,(L,W,W),generator=gen,dtype=torch.int8),sa=torch.full((L,W),.002),
    b=torch.zeros(L,W),sp=torch.full((L,W),.3),isx=torch.full((L,W),12.),cos=torch.ones(T,hd//2),sin=torch.zeros(T,hd//2))
k['Aq'][0].zero_();k['inv']=(4096/(2*math.log(2)*k['sp'].double())).float()
inputs=[(.1*torch.randn((L,T,W),generator=gen)).bfloat16().view(torch.int16) for _ in range(2)]
inputs[0][0,0,:5]=torch.tensor([0,-32768,32640,32705,-128],dtype=torch.int16)
for learned in (False,True):
    c=dict(k);p=Policy()
    if learned:
        c.update(groups=4,Acq=torch.zeros_like(c['Aq']))
        p=Policy(kind=2,alpha=.75,weights=np.tile(np.eye(G,dtype=np.float32),(L,1,1)),bias=np.zeros((L,G),np.float32))
        codec=HeadSpreadCodec(c,C,hd,p,tokens_per_program=8)
    else:
        codec=FastDenseEncode(c,C,hd,tokens_per_program=8);codec.policy=p
    pred=w.consts_digest(c,C,hd,8,'uniform');S,cap=codec.layout(T)
    arena=torch.zeros(S*cap,dtype=torch.int16)
    path=PinnedT(codec,T,pred,w.HEADER.size+4*S+2*arena.numel(),software=True)
    previous=None
    for words in inputs:
        arena.zero_();encoded=codec.encode(words,out=arena)
        digest=hashlib.sha256(words.numpy().tobytes()).hexdigest()
        canonical,_=w.pack(codec,*encoded,T,pred,p,digest)
        packet=path.pack(encoded,digest)
        assert packet.view()==canonical
        path.poison_receiver();received,expect=path.receive(packet)
        assert received[0].data_ptr()!=arena.data_ptr()
        assert expect==digest and torch.equal(codec.decode(*received,T),words)
        oldarena,oldstarts,_=w.unpack(codec,bytes(packet.view()),pred)
        assert torch.equal(codec.decode(oldarena,oldstarts,T),words)
        if previous is None:previous=(bytes(packet.view()),words.clone())
    arena.fill_(-222);path.poison_receiver()
    received,_=path.receive(HostPacket.copy_from(previous[0],pin=False))
    assert torch.equal(codec.decode(*received,T),previous[1])
print('global and learned T: exact canonical bytes, changed inputs, poisoned receiver, original receiver passed')
'''
        result=subprocess.run([sys.executable,"-c",script],cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ,TRITON_INTERPRET="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),
            capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_fresh_retrieval_aggregation_and_full_gate(self):
        rows=[]
        for case in ("a","b"):
            for ri in range(3):
                for name,ms in (("legacy_checked",100),("pinned_checked",60)):
                    fresh=[dict(total_ms=ms,encode_ms=5,pack_D2H_ms=20,restore_H2D_ms=30,decode_ms=5,bit_exact=True)]*2
                    retrieval=[dict(total_ms=35,restore_H2D_ms=30,decode_ms=5,bit_exact=True)]*2
                    rows.append(dict(method="t_fast",path=name,case=case,round=ri,packet_bytes=100,values=100,
                        bit_exact=True,canonical_packet_identical=True,fresh=fresh,retrieval=retrieval,
                        packet_sha256="packet",source_words_sha256="words",
                        packet_only_proofs={k:dict(bit_exact=True,packet_sha256="packet",source_words_sha256="words")
                            for k in ("legacy","pinned")} if name=="pinned_checked" and ri==0 else None))
        table=aggregate(rows,["t_fast"],["a","b"],3,2)
        self.assertEqual(table[1]["packet_bytes"],200)
        self.assertEqual(table[1]["fresh_total_ms"],60)
        self.assertEqual(table[1]["retrieval_total_ms"],35)
        r=dict(gpu_executed=True,full_reference_coverage=True,errors=[],arguments=dict(rounds=3,methods=["t_fast"]),aggregate=table)
        self.assertTrue(decision(r)["t_fast"]["candidate_for_review"])
        r["gpu_executed"]=False
        self.assertFalse(decision(r)["t_fast"]["candidate_for_review"])
        r["gpu_executed"]=True;r["aggregate"]=aggregate(rows[:-1],["t_fast"],["a","b"],3,2)
        self.assertFalse(decision(r)["t_fast"]["candidate_for_review"])
        rows[1]["packet_only_proofs"]=None
        r["aggregate"]=aggregate(rows,["t_fast"],["a","b"],3,2)
        self.assertFalse(decision(r)["t_fast"]["candidate_for_review"])

    def test_retrieval_really_omits_encode_and_still_restores_the_packet(self):
        class TrackedRaw(native.Raw):
            def __init__(self):super().__init__((1,1,8),"cpu");self.encodes=0
            def encode(self,words):self.encodes+=1;return super().encode(words)
        adapter=TrackedRaw();words=torch.arange(8,dtype=torch.int16).reshape(1,1,8)
        path=PinnedParts(adapter,"raw_bf16",words.shape,"a"*64,1024,software=True)
        with patch("torch.cuda.synchronize"):
            output,packet,fresh=run_fresh(adapter,path,words,"b"*64,torch.device("cpu"))
            self.assertTrue(torch.equal(output,words));self.assertEqual(adapter.encodes,1)
            adapter.out.fill_(-1);path.poison_receiver()
            output,retrieval=run_retrieval(adapter,path,packet,"b"*64,torch.device("cpu"))
            self.assertTrue(torch.equal(output,words));self.assertEqual(adapter.encodes,1)
        self.assertAlmostEqual(fresh["total_ms"],sum(fresh[k] for k in ("encode_ms","pack_D2H_ms","restore_H2D_ms","decode_ms")))
        self.assertAlmostEqual(retrieval["total_ms"],retrieval["restore_H2D_ms"]+retrieval["decode_ms"])

    def test_gpu_CLI_rejects_interpreter_before_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"must_not_exist"
            r=subprocess.run([sys.executable,"scripts/bench_t_transfer_path.py","--reference-run","unused","--output",str(out)],
                cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,TRITON_INTERPRET="1"),capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn("GPU results require compiled CUDA kernels",r.stderr)
            self.assertFalse(out.exists())


if __name__=="__main__":unittest.main()
