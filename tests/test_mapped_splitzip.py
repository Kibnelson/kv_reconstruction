"""Exact mapped I/O checks; CPU interpreter results are not GPU performance."""
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kv_compare import mapped_splitzip as mapped


@dataclass
class Encoded:
    pk: torch.Tensor
    sm: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    chunk_id: torch.Tensor
    local_pos: torch.Tensor
    esc_val: torch.Tensor
    n: int
    n_esc: int
    chunk_size: int


MODULE = SimpleNamespace(ChunkLocalGPUEncoded=Encoded)
FIELDS = ('pk', 'sm', 'counts', 'starts', 'chunk_id', 'local_pos', 'esc_val')


def tables(device='cpu'):
    book = [0,1,2,3,4,5,6,7,120,121,122,123,124,125,126,127]
    marked = torch.full((256,),16,dtype=torch.uint8,device=device)
    plain = torch.zeros(256,dtype=torch.uint8,device=device)
    common = torch.zeros(256,dtype=torch.uint8,device=device)
    for code, exponent in enumerate(book):
        marked[exponent] = code;plain[exponent] = code;common[exponent] = 1
    return SimpleNamespace(chunk_size=1024, enc_lut=plain, enc_lut_marked=marked,
        common_lut=common, dec_lut=torch.tensor(book,dtype=torch.uint8,device=device))


def oracle(codec, words, index):
    """Independent scalar-field definition of the released author's bundle."""
    raw = words.index_select(0,index).flatten().long() & 65535
    exp = (raw >> 7) & 255
    codes = codec.enc_lut_marked[exp].long() & 15
    pk = ((codes[0::2] << 4) | codes[1::2]).to(torch.uint8)
    sm = (((raw >> 8) & 128) | (raw & 127)).to(torch.uint8)
    escape = codec.common_lut[exp] == 0
    positions = torch.nonzero(escape).flatten()
    counts = escape.reshape(-1,1024).sum(1).to(torch.int32)
    starts = torch.cumsum(counts,0)-counts
    return Encoded(pk,sm,counts,starts,(positions//1024).to(torch.int32),
        (positions%1024).to(torch.uint16),exp[positions].to(torch.uint8),raw.numel(),positions.numel(),1024)


class ContractTests(unittest.TestCase):
    def test_mapping_is_owned_bounded_unique_sorted_and_version_guarded(self):
        original=torch.tensor([0,3],dtype=torch.int64)
        value=mapped.prepare_mapping(original,4)
        original[0]=1
        self.assertEqual(value.layers,(0,3));self.assertEqual(value.tensor.tolist(),[0,3])
        for indices in ([],[1,1],[2,0],[-1],[4]):
            with self.assertRaises(ValueError):mapped.prepare_mapping(torch.tensor(indices,dtype=torch.int64),4)
        with self.assertRaises(ValueError):mapped.prepare_mapping(torch.tensor([0],dtype=torch.int32),4)
        value.tensor[0]=1
        with self.assertRaises(ValueError):mapped.encode_layers(tables(),MODULE,torch.zeros(4,4,256,dtype=torch.int16),value)

    def test_no_implicit_cpu_performance_fallback_or_invalid_geometry(self):
        index=torch.tensor([0],dtype=torch.int64)
        with mock.patch.dict(os.environ,{'TRITON_INTERPRET':'0'}):
            with self.assertRaises(RuntimeError):mapped.encode_layers(tables(),MODULE,torch.zeros(1,4,256,dtype=torch.int16),index)
        for words in (torch.zeros(1,4,256),torch.zeros(1,3,256,dtype=torch.int16),torch.zeros(1,1024,1,dtype=torch.int16)):
            with self.assertRaises(ValueError):mapped.encode_layers(tables(),MODULE,words,index)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def assert_fields(self,a,b):
        self.assertEqual((a.n,a.n_esc,a.chunk_size),(b.n,b.n_esc,b.chunk_size))
        for name in FIELDS:self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)),name)

    def test_all_bf16_patterns_match_independent_fields_and_decode_noncontiguous_layers(self):
        patterns=torch.arange(65536,dtype=torch.int32).to(torch.int16).reshape(256,256)
        words=torch.stack((patterns,torch.zeros_like(patterns),torch.ones_like(patterns),patterns.flip(0)))
        index=torch.tensor([0,3],dtype=torch.int64);mapping=mapped.prepare_mapping(index,4);codec=tables()
        expected=words.clone();encoded=mapped.encode_layers(codec,MODULE,words,mapping)
        self.assert_fields(encoded,oracle(codec,words,index))
        output=torch.full_like(words,1234)
        result=mapped.decode_layers_into(codec,encoded,mapping,output)
        self.assertIs(result,output)
        self.assertTrue(torch.equal(result[0],words[0]));self.assertTrue(torch.equal(result[3],words[3]))
        self.assertTrue(bool((result[1:3]==1234).all()));self.assertTrue(torch.equal(words,expected))
        held=output.clone();other=torch.full_like(words,-321)
        mapped.decode_layers_into(codec,encoded,mapping,other,validated=True)
        self.assertTrue(torch.equal(output,held));self.assertTrue(torch.equal(other[3],words[3]))

    def test_no_escape_and_all_escape_native_field_order(self):
        codec=tables();index=torch.tensor([1,3],dtype=torch.int64)
        for bit,escaped in ((0x3f80,False),(0x7fc1,True)):
            words=torch.full((4,8,256),bit,dtype=torch.int16)
            words[3,:,0] |= 0x40
            encoded=mapped.encode_layers(codec,MODULE,words,index)
            self.assert_fields(encoded,oracle(codec,words,index))
            self.assertEqual(encoded.n_esc,encoded.n if escaped else 0)
            out=torch.zeros_like(words);mapped.decode_layers_into(codec,encoded,index,out)
            self.assertTrue(torch.equal(out[index],words[index]))

    def test_malformed_escape_fields_are_rejected_before_kernel(self):
        codec=tables();words=torch.full((2,4,256),0x7fc1,dtype=torch.int16)
        index=torch.tensor([1],dtype=torch.int64);encoded=mapped.encode_layers(codec,MODULE,words,index)
        original=encoded.chunk_id.clone();encoded.chunk_id[0]=-1
        with self.assertRaises(ValueError):mapped.decode_layers_into(codec,encoded,index,torch.empty_like(words))
        encoded.chunk_id=original;encoded.local_pos[0]=1024
        with self.assertRaises(ValueError):mapped.decode_layers_into(codec,encoded,index,torch.empty_like(words))
        encoded.local_pos[0]=0;encoded.n+=4
        with self.assertRaises(ValueError):mapped.decode_layers_into(codec,encoded,index,torch.empty_like(words))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET')!='1','actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_author_field_parity_all_patterns_and_direct_owned_output(self):
        from kv_compare.framed_baselines import Baseline
        from kv_compare.mixed_transfer import verify_splitzip_source
        source=Path(__file__).resolve().parents[1]/'runs/_llama_official/sources/splitzip'
        verify_splitzip_source(source)
        baseline=Baseline('splitzip_official_framed','cuda:0',source=source)
        fixture=tables('cuda:0')
        for name in ('enc_lut','enc_lut_marked','common_lut','dec_lut'):
            setattr(baseline.codec,name,getattr(fixture,name))
        patterns=torch.arange(65536,dtype=torch.int32,device='cuda:0').to(torch.int16).reshape(256,256)
        words=torch.stack((patterns,torch.zeros_like(patterns),torch.ones_like(patterns),patterns.flip(0)))
        index=torch.tensor([0,3],dtype=torch.int64,device='cuda:0');mapping=mapped.prepare_mapping(index,4)
        original=words.clone()
        for current in (words,torch.full_like(words,0x3f80),torch.full_like(words,0x7fc1)):
            reference=baseline.codec.encode(current.index_select(0,index).view(torch.bfloat16).flatten())
            actual=mapped.encode_layers(baseline.codec,baseline.module,current,mapping)
            for name in FIELDS:self.assertTrue(torch.equal(getattr(actual,name),getattr(reference,name)),name)
            out=torch.full_like(current,2468)
            mapped.decode_layers_into(baseline.codec,actual,mapping,out)
            self.assertTrue(torch.equal(out[index],current[index]))
            self.assertTrue(bool((out[1:3]==2468).all()))
            held=out.clone();stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                next_out=torch.empty_like(current)
                mapped.decode_layers_into(baseline.codec,actual,mapping,next_out)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out,held));self.assertTrue(torch.equal(next_out[index],current[index]))
        self.assertTrue(torch.equal(words,original))


if __name__=='__main__':unittest.main()
