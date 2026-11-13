"""Transfer completeness and measurement decisions; GPU checks are in the runner."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from .native_compare import (ExANS, ExEncoded, PREFIX, Raw, block_layout, exans_paid_bytes,
                             pack_parts, unpack_parts)
from scripts.bench_t_baselines import aggregate, packet_from, packet_to, summary


class TransferTests(unittest.TestCase):
    def test_every_bf16_word_and_empty_directory_roundtrip(self):
        words=torch.arange(65536,dtype=torch.int64).to(torch.int16).reshape(2,128,256)
        codec=Raw(words.shape,"cpu")
        digest=hashlib.sha256(words.numpy().tobytes()).hexdigest()
        packet,cost=pack_parts("raw_bf16",words.shape,"a"*64,digest,
            [("words",words),("empty",torch.empty(0,dtype=torch.int32))])
        parts,meta=unpack_parts(packet,"raw_bf16",words.shape,"a"*64,"cpu")
        self.assertTrue(torch.equal(parts["words"],words))
        self.assertEqual(parts["empty"].numel(),0)
        self.assertEqual(len(packet),cost["header_bytes"]+words.numel()*2)
        self.assertEqual(meta["words_sha256"],digest)

    def test_packet_identity_corruption_and_trailing_bytes_rejected(self):
        packet,_=pack_parts("raw_bf16",[4],"a"*64,"b"*64,[("words",torch.arange(4,dtype=torch.int16))])
        for bad in (packet[:-1],packet+b"x",packet[:50]+bytes([packet[50]^1])+packet[51:]):
            with self.assertRaises(ValueError):unpack_parts(bad,"raw_bf16",[4],"a"*64,"cpu")
        for method,shape,state in (("exans",[4],"a"*64),("raw_bf16",[5],"a"*64),("raw_bf16",[4],"c"*64)):
            with self.assertRaises(ValueError):unpack_parts(packet,method,shape,state,"cpu")

    def test_unbounded_descriptor_product_cannot_overflow_into_small_payload(self):
        packet,_=pack_parts("raw_bf16",[1],"a"*64,"b"*64,[("words",torch.zeros(1,dtype=torch.int16))])
        _,hlen,_=PREFIX.unpack_from(packet)
        meta=json.loads(packet[PREFIX.size:PREFIX.size+hlen])
        meta["parts"][0]["shape"]=[2**32,2**32]
        header=json.dumps(meta).encode()
        body=header+packet[PREFIX.size+hlen:]
        forged=PREFIX.pack(b"TCMPV001",len(header),hashlib.sha256(body).digest())+body
        with self.assertRaises(ValueError):unpack_parts(forged,"raw_bf16",[1],"a"*64,"cpu")

    def test_full_real_window_has_paid_partial_exans_block(self):
        layout=block_layout(28*8192*2048,122*4096)
        self.assertEqual(layout["blocks"],940)
        self.assertEqual(layout["tail_values"],32768)
        self.assertEqual(layout["tail_raw_bytes"],65536)
        self.assertEqual(exans_paid_bytes([6,0],3,4),8+6+8+6)
        self.assertEqual(block_layout(3,4)["blocks"],0)
        with self.assertRaises(ValueError):exans_paid_bytes([9],0,4)

    def test_exans_subset_decode_copies_rejected_blocks_and_tail(self):
        # Native API contract: decode only fitted record 1; rejected record 0 must
        # come from paid raw words, and all values after the final block survive.
        codec=ExANS.__new__(ExANS)
        codec.ex=SimpleNamespace(BLOCK_WORDS=4,BLOCK_BYTES=8)
        codec.shape=(11,);codec.device=torch.device("cpu");codec.nb=2
        codec.layout=block_layout(11,4);codec.out=torch.full((11,),-1,dtype=torch.int16)
        codec.handle=ctypes.c_void_p(1)
        called=[]
        def native(handle,records,indices,count,addrs,sizes,offs,stream):
            called.append(([indices[i] for i in range(count)],count))
            self.assertEqual(addrs[0],codec.out.data_ptr()+8)
            self.assertEqual(list(sizes),[8])
            codec.out[4:8]=torch.tensor([40,41,42,43],dtype=torch.int16)
            return 0
        codec.lib=SimpleNamespace(openlake_expans_decode=native)
        encoded=ExEncoded(torch.zeros(2,8,dtype=torch.uint8),(False,True),
            torch.tensor([[0,1,2,3]],dtype=torch.int16),torch.tensor([80,81,82],dtype=torch.int16))
        with patch("torch.cuda.current_stream",return_value=SimpleNamespace(cuda_stream=0)):
            actual=codec.decode(encoded)
        self.assertEqual(called,[([1],1)])
        self.assertEqual(actual.tolist(),[0,1,2,3,40,41,42,43,80,81,82])

    def test_empty_fitted_subset_never_calls_native_decoder(self):
        codec=ExANS.__new__(ExANS)
        codec.ex=SimpleNamespace(BLOCK_WORDS=4,BLOCK_BYTES=8)
        codec.device=torch.device("cpu");codec.nb=0;codec.layout=block_layout(3,4)
        codec.out=torch.empty(3,dtype=torch.int16)
        out=codec.decode(ExEncoded(torch.empty(0,8,dtype=torch.uint8),(),
            torch.empty(0,4,dtype=torch.int16),torch.tensor([-32768,32640,32705],dtype=torch.int16)))
        self.assertEqual(out.tolist(),[-32768,32640,32705])

    def test_raw_packet_api_used_by_fresh_receiver(self):
        words=torch.tensor([[[0,-32768,32640,32705]]],dtype=torch.int16)
        codec=Raw(words.shape,"cpu")
        job=dict(shape=list(words.shape),state_sha256="a"*64)
        packet,_=packet_from(codec,"raw_bf16",codec.encode(words),job,"b"*64)
        restored,expected=packet_to(Raw(words.shape,"cpu"),"raw_bf16",packet,job,"cpu")
        self.assertTrue(torch.equal(codec.decode(restored),words))
        self.assertEqual(expected,"b"*64)

    def test_both_T_adapters_preserve_reference_packets_and_decode(self):
        if os.environ.get("TRITON_INTERPRET")!="1":
            env=dict(os.environ,TRITON_INTERPRET="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
            r=subprocess.run([sys.executable,"-m","unittest",
                "kv_compare.test_native_compare.TransferTests.test_both_T_adapters_preserve_reference_packets_and_decode"],
                env=env,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            return
        from .head_spread import Policy
        from .head_spread_wire import save_state,consts_digest,load_state,codec_for_state,pack
        from .native_compare import make
        layers,tokens,hd,heads=2,8,8,16
        width=hd*heads;channels=width//2
        gen=torch.Generator().manual_seed(488)
        consts=dict(Aq=torch.randint(-1,2,(layers,width,width),dtype=torch.int8,generator=gen),
            sa=torch.full((layers,width),.002),b=torch.zeros(layers,width),
            sp=torch.full((layers,width),.3),isx=torch.full((layers,width),12.),
            cos=torch.ones(tokens,hd//2),sin=torch.zeros(tokens,hd//2))
        consts["Aq"][0].zero_()
        consts["inv"]=(4096/(2*math.log(2)*consts["sp"].double())).float()
        words=(.1*torch.randn((layers,tokens,width),generator=gen)).bfloat16().view(torch.int16)
        words[0,0,:5]=torch.tensor([0,-32768,32640,32705,-128],dtype=torch.int16)
        word_hash=hashlib.sha256(words.numpy().tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            for method in ("t_fast","t_hbh"):
                k=dict(consts)
                policy=Policy()
                if method=="t_hbh":
                    k.update(groups=16,Acq=torch.zeros_like(k["Aq"]))
                    policy=Policy(kind=2,alpha=.75,weights=np.tile(np.eye(heads,dtype=np.float32),(layers,1,1)),
                                  bias=np.zeros((layers,heads),np.float32))
                state_path=Path(td)/(method+".pt")
                pred=consts_digest(k,channels,hd,8,"uniform")
                save_state(state_path,k,channels,hd,8,"uniform",policy,pred)
                original=codec_for_state(load_state(state_path,"cpu"),tokens)
                segments,capacity=original.layout(tokens)
                arena=torch.zeros(segments*capacity,dtype=torch.int16)
                expected,_=pack(original,*original.encode(words,out=arena),tokens,pred,policy,word_hash)
                codec=make(method,state_path,words.shape,"cpu")
                packet,_=codec.pack(codec.encode(words),word_hash)
                self.assertEqual(packet,expected)
                restored,meta=codec.receive_packet(packet)
                self.assertEqual(meta["words_sha256"],word_hash)
                self.assertTrue(torch.equal(codec.decode(restored),words))

    def test_partial_round_cannot_be_complete_and_pair_is_independent(self):
        timing=lambda n:dict(cuda_median_ms=n,wall_median_ms=n+1)
        rows=[dict(method="t_fast",case="r0",round=0,packet_bytes=50,values=50,bit_exact=True,
            encode=timing(4),decode=timing(3),pair=timing(8))]
        r=aggregate(rows,["t_fast","splitzip"],["r0","r1"],3)
        self.assertFalse(r[0]["complete"])
        self.assertFalse(r[1]["complete"])
        self.assertEqual(r[0]["pair_cuda_ms"],8)
        self.assertEqual(r[0]["saved_percent"],50)
        self.assertAlmostEqual(r[0]["ideal_serial_link_ms"]["100"],9+50*8/100e6)
        self.assertIn("unavailable",summary(dict(status="incomplete",aggregate=r)))


if __name__=="__main__":unittest.main()
