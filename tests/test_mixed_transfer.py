"""Checked transfer tests; CPU fake SplitZip is never performance evidence."""
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from kv_compare import mixed_transfer as mt
from kv_compare.causal_packet import Packet, sha
from kv_compare.framed_baselines import Baseline, LUTS, unframe
from kv_compare.team_packet import TeamCodec, save_state
from tests.test_team_packet import fixed_state


class FakeEncoded:
    """All-escape semantic fixture; not the author implementation or a benchmark."""
    def __init__(self, pk, sm, counts, starts, chunk_id, local_pos, esc_val, n, n_esc, chunk_size):
        for name,value in zip(('pk','sm','counts','starts','chunk_id','local_pos','esc_val','n','n_esc','chunk_size'),
            (pk,sm,counts,starts,chunk_id,local_pos,esc_val,n,n_esc,chunk_size)):
            setattr(self,name,value)
    @property
    def compressed_bytes(self):
        return (self.n+1)//2+self.n+7*self.n_esc


class FakeSZ:
    def __init__(self, device): self.device=torch.device(device)
    def encode(self, tensor):
        words=tensor.contiguous().view(torch.int16).flatten().long()&65535;n=words.numel()
        i=torch.arange(n,device=self.device);empty=torch.empty(0,dtype=torch.int32,device=self.device)
        return FakeEncoded(torch.full(((n+1)//2,),255,dtype=torch.uint8,device=self.device),
            (((words>>8)&128)|(words&127)).to(torch.uint8),empty,empty,(i//1024).to(torch.int32),
            (i%1024).to(torch.uint16),((words>>7)&255).to(torch.uint8),n,n,1024)
    def decode(self, e):
        words=((e.sm.long()&128)<<8)|(e.sm.long()&127)
        i=e.chunk_id.long()*1024+e.local_pos.long()
        words[i] |= e.esc_val.long()<<7
        return words.to(torch.int16).view(torch.bfloat16)


class FakeBaseline:
    def __init__(self, method, device, *, source=None, state=None):
        self.method=method;self.device=torch.device(device);self.codec=FakeSZ(device)
        self.module=SimpleNamespace(ChunkLocalGPUEncoded=FakeEncoded)
        self.state_sha256=sha(state);self.shared_bytes=sum(LUTS.values())
    encode=Baseline.encode
    decode=Baseline.decode


def state(root, layers=2, tokens=64, rank=False):
    path=Path(root)/'state'; consts=fixed_state(layers=layers,tokens=tokens)
    rc=None
    if rank:
        rc=dict(rank=256,Bq=torch.zeros(layers,256,256,dtype=torch.int8),
            Uq=torch.zeros(layers,256,256,dtype=torch.int8),m1=torch.ones(layers,256),
            m0=torch.zeros(layers,256),sa=torch.ones(layers,256),b=torch.zeros(layers,256),
            sp=consts['sp'].clone(),inv=consts['inv'].clone())
    save_state(path,consts,rc)
    lut=Path(root)/'luts.npz';np.savez(lut,**{k:np.zeros(n,dtype=np.uint8) for k,n in LUTS.items()})
    return path,lut


class ContractTests(unittest.TestCase):
    def test_extension_identity_and_source_tampering(self):
        identity=mt.extension_identity()
        self.assertEqual(identity['commit'],'8a40dd98282ed364a8fe1077d171a1922277c1ec')
        import shutil
        with tempfile.TemporaryDirectory() as raw:
            p=Path(raw)/'vendor';shutil.copytree(mt.EXTENSION,p)
            (p/'mixed.py').write_bytes((p/'mixed.py').read_bytes()+b'\n# changed')
            with mock.patch.object(mt,'EXTENSION',p):
                with self.assertRaises(ValueError):mt.extension_identity()

    def test_masks_are_explicit_and_all_zero_has_no_predictor_installation(self):
        for mask in ([],[1,0],'ab',[True,None]):
            with self.assertRaises(ValueError):mt.MixedTransfer('missing','missing','missing','x',mask,'cpu')
        with tempfile.TemporaryDirectory() as raw:
            path,lut=state(raw)
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',FakeBaseline),mock.patch.object(mt,'load_state') as load:
                codec=mt.MixedTransfer('missing','unused',lut,sha(lut),[False,False],'cpu')
                load.assert_not_called()
                words=torch.arange(2*64*256,dtype=torch.int32).to(torch.int16).reshape(2,64,256)
                baseline=FakeBaseline('splitzip_official_framed','cpu',state=lut)
                packet=codec.encode(words)
                self.assertEqual(packet.export(),baseline.encode(words).export())
                self.assertTrue(torch.equal(codec.decode(Packet.restore(packet.export(),'cpu')),words))
                self.assertEqual(codec.accounting(packet)['shared_predictor_bytes'],0)
                self.assertEqual(codec.accounting(packet)['packet_bytes'],len(packet.export()))
                self.assertEqual(codec.accounting(packet)['fallback_layers'],[0,1])
                self.assertEqual(codec.accounting(packet)['rans_layers'],[])
                self.assertEqual(codec.accounting(packet)['prior_reset_layers'],[])
                self.assertEqual(codec.accounting(packet)['mask_transitions'],0)
                with self.assertRaises(ValueError):codec.encode(words,start=1)
                with self.assertRaises(ValueError):codec.encode(words[:1])


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_native_mixed_partial_and_all_rans_real_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            path,lut=state(raw);gen=torch.Generator().manual_seed(404)
            words=torch.randn(2,64,256,generator=gen).bfloat16().view(torch.int16)
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',FakeBaseline):
                for mask in ([True,False],[False,True],[True,True]):
                    with self.subTest(mask=mask):
                        codec=mt.MixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform')
                        packet=codec.encode(words);wire=packet.export();restored=Packet.restore(wire,'cpu')
                        held=codec.decode(restored);other=codec.decode(codec.encode(-words))
                        self.assertTrue(torch.equal(held,words));self.assertTrue(torch.equal(other,-words))
                        self.assertNotEqual(held.data_ptr(),other.data_ptr())
                        self.assertEqual(packet.export(),wire)
                        self.assertEqual(codec.accounting(packet)['packet_bytes'],len(wire))
                        self.assertEqual(codec.accounting(packet)['expanded_arena_bytes'],0)
                        account=codec.accounting(packet)
                        self.assertEqual(account['rans_layers'],[l for l,v in enumerate(mask) if v])
                        self.assertEqual(account['prior_reset_layers'],[l for l,v in enumerate(mask) if v and (l==0 or not mask[l-1])])
                        self.assertEqual(account['predecessor_fallback_layers'],[l for l,v in enumerate(mask) if v and l>0 and not mask[l-1]])
                        self.assertEqual(account['mask_transitions'],int(mask[0]!=mask[1]))
                        if all(mask):
                            original=TeamCodec(path,'cpu','dense_uniform').encode(words)
                            _,parts,_=unframe(packet,mt.METHOD,codec.state_sha256)
                            self.assertTrue(torch.equal(parts['rans_words'],original.payload))

    def test_corrupt_directory_or_escape_is_rejected_before_native(self):
        with tempfile.TemporaryDirectory() as raw:
            path,lut=state(raw)
            words=torch.ones(2,64,256,dtype=torch.bfloat16).view(torch.int16)
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',FakeBaseline):
                codec=mt.MixedTransfer(path,'unused',lut,sha(lut),[True,False],'cpu',arm='dense_uniform')
                wire=codec.encode(words).export()
                for name in ('rans_offsets','l1_chunk_id'):
                    packet=Packet.restore(wire,'cpu');_,parts,_=unframe(packet,mt.METHOD,codec.state_sha256)
                    parts[name][0]=-1
                    with mock.patch.object(codec.native,'decode_mixed') as native:
                        with self.assertRaises(ValueError):codec.decode(packet)
                        native.assert_not_called()
                packet=Packet.restore(wire,'cpu');packet.header['extra']['mask']=[False,True]
                with self.assertRaises(ValueError):codec.decode(packet)
                with self.assertRaises(ValueError):codec.decode_native(object())
                bad=bytearray(wire);bad[-33]^=1
                with self.assertRaises(ValueError):Packet.restore(bad,'cpu')


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET')!='1','actual CUDA required')
class CudaTests(unittest.TestCase):
    def _source_and_lut(self, root):
        source=Path(__file__).resolve().parents[1]/'runs/_llama_official/sources/splitzip'
        self.assertTrue((source/'codec_gpu.py').is_file(),'pinned author SplitZip checkout required')
        mt.verify_splitzip_source(source)
        codec=Baseline('splitzip_official_framed','cuda:0',source=source)
        lut=Path(root)/'actual_luts.npz'
        gen=torch.Generator(device='cuda:0').manual_seed(709)
        codec.calibrate(torch.randn(65536,generator=gen,device='cuda:0').bfloat16().view(torch.int16),lut)
        return source,lut

    def test_actual_gpu_mixed_roundtrip_and_owned_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            path,_=state(raw,tokens=256,rank=True);source,lut=self._source_and_lut(raw)
            gen=torch.Generator(device='cuda:0').manual_seed(991)
            words=[torch.randn(2,256,256,generator=gen,device='cuda:0').bfloat16().view(torch.int16),
                   torch.full((2,256,256),4.,dtype=torch.bfloat16,device='cuda:0').view(torch.int16)]
            for arm in ('dense_uniform','rank256_uniform'):
                for mask in ([False,False],[True,False],[False,True],[True,True]):
                    with self.subTest(arm=arm,mask=mask):
                        codec=mt.MixedTransfer(path,source,lut,sha(lut),mask,'cuda:0',arm=arm)
                        packets=[Packet.restore(codec.encode(w).export(),'cuda:0') for w in words]
                        held=codec.decode(packets[0])
                        for i in (1,0):
                            self.assertTrue(torch.equal(codec.decode(packets[i]),words[i]))
                            self.assertTrue(torch.equal(held,words[0]))
                        self.assertEqual(codec.accounting(packets[0])['packet_bytes'],len(packets[0].export()))

    def test_actual_gpu_extrema_and_boundary_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            path,_=state(raw,layers=1,tokens=256);source,lut=self._source_and_lut(raw)
            words=torch.arange(65536,dtype=torch.int32,device='cuda:0').to(torch.int16).reshape(1,256,256)
            for mask in ([False],[True]):
                codec=mt.MixedTransfer(path,source,lut,sha(lut),mask,'cuda:0',arm='dense_uniform')
                packet=Packet.restore(codec.encode(words).export(),'cuda:0')
                stream=torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    restored=codec.decode(packet)
                    del packet
                gc.collect();torch.cuda.synchronize()
                self.assertTrue(torch.equal(restored,words))
                if mask[0]:
                    packet=codec.encode(words);_,parts,_=unframe(packet,mt.METHOD,codec.state_sha256)
                    parts['rans_offsets'][0]=-1
                    with mock.patch.object(codec.native,'decode_mixed') as native:
                        with self.assertRaises(ValueError):codec.decode(packet)
                        native.assert_not_called()


if __name__=='__main__':unittest.main()
