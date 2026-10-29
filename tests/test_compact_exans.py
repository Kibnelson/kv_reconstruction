import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import unittest

import torch

from kv_compare import compact_exans as compact
from kv_compare.causal_packet import Packet
from kv_compare.framed_baselines import frame, unframe


def record(length, salt=1):
    value=torch.zeros(compact.RECORD,dtype=torch.uint8)
    value[:length]=(torch.arange(length,dtype=torch.int32)*salt).to(torch.uint8)
    header=compact.HEADER.pack(compact.MAGIC,2,64,compact.layout_id(),compact.RAW,length,
        64,compact.SYMBOLS,64+compact.SYMBOLS,compact.PHASE_BYTES,compact.FRAME_OFFSET,
        length-compact.FRAME_OFFSET,0,0,0,0)
    value[:64]=torch.tensor(list(header),dtype=torch.uint8)
    return value


def fixed_packet(shape, good, bad, records, raw):
    n=2
    for value in shape:n*=value
    count=(n+compact.RAW-1)//compact.RAW
    return frame(compact.FIXED_METHOD,shape,compact.FIXED_STATE,
        [('fit_indices',torch.tensor(good,dtype=torch.int32)),('raw_indices',torch.tensor(bad,dtype=torch.int32)),
         ('records',records),('raw',raw)],
        dict(blocks=count,raw_block_bytes=compact.RAW,record_bytes=compact.RECORD,input_padding_bytes=count*compact.RAW-n))


class CompactExansTests(unittest.TestCase):
    def test_pack_scatter_keeps_headers_actual_bytes_and_zero_tails(self):
        sizes=[compact.FRAME_OFFSET+7,compact.FRAME_OFFSET+123]
        source=torch.cat([record(size,i+1) for i,size in enumerate(sizes)])
        offsets,payload=compact.compact_records(source)
        self.assertEqual(offsets.tolist(),[0,sizes[0],sum(sizes)])
        self.assertEqual(payload.numel(),sum(sizes))
        target=torch.full((3*compact.RECORD,),255,dtype=torch.uint8)
        compact.expand_records(payload,offsets,torch.tensor([0,2],dtype=torch.int32),target)
        self.assertTrue(torch.equal(target[:compact.RECORD],source[:compact.RECORD]))
        self.assertTrue(torch.equal(target[2*compact.RECORD:],source[compact.RECORD:]))
        self.assertTrue(bool((target[compact.RECORD:2*compact.RECORD]==255).all()))

    def test_complete_carrier_charges_tail_raw_fallback_directories_and_wire(self):
        shape=[2*compact.SYMBOLS+17]  # Three blocks, with a 34-byte original tail.
        source=torch.cat([record(compact.FRAME_OFFSET+3),record(compact.FRAME_OFFSET+19,3)])
        raw=(torch.arange(compact.RAW,dtype=torch.int32)*7).to(torch.uint8)
        old=fixed_packet(shape,[0,2],[1],source,raw)
        packet=compact.compact_packet(old,3)
        recovered=Packet.restore(packet.export(),'cpu')
        actual_shape,parts,good,bad,layout=compact._parts(recovered,3)
        self.assertEqual(actual_shape,shape);self.assertEqual(good,[0,2]);self.assertEqual(bad,[1])
        self.assertEqual(layout['input_padding_bytes'],compact.RAW-34)
        self.assertTrue(torch.equal(parts['raw'],raw))
        self.assertEqual(parts['record_offsets'].numel(),3)
        self.assertEqual(compact.packet_bytes(packet),len(packet.export()))
        self.assertLess(len(packet.export()),len(old.export()))

    def test_all_raw_partial_tail_is_preserved_without_native_records(self):
        shape=[513]
        raw=(torch.arange(compact.RAW,dtype=torch.int32)*11).to(torch.uint8)
        old=fixed_packet(shape,[],[0],torch.empty(0,dtype=torch.uint8),raw)
        packet=compact.compact_packet(old,1)
        _,parts,good,bad,layout=compact._parts(packet,1)
        self.assertEqual(good,[]);self.assertEqual(bad,[0]);self.assertEqual(parts['record_offsets'].tolist(),[0])
        self.assertEqual(parts['records'].numel(),0)
        self.assertEqual(layout['nbytes'],1026)
        self.assertTrue(torch.equal(parts['raw'][:1026],raw[:1026]))

    def test_bad_native_header_and_actual_length_are_rejected(self):
        data=record(compact.FRAME_OFFSET+7)
        broken=data.clone();broken[0]^=1
        with self.assertRaises(ValueError):compact.compact_records(broken)
        header=bytearray(data[:64].numpy().tobytes())
        header[20:24]=(compact.RECORD+1).to_bytes(4,'little')
        with self.assertRaises(ValueError):compact.validate_header(header)
        with self.assertRaises(ValueError):compact.validate_header(data[:64].numpy().tobytes(),compact.FRAME_OFFSET+8)
        header=bytearray(data[:64].numpy().tobytes());header[24:28]=(65).to_bytes(4,'little')
        with self.assertRaises(ValueError):compact.validate_header(header)

    def test_duplicate_missing_overlapping_and_overflow_directories_are_rejected(self):
        n=compact.FRAME_OFFSET+7
        cases=[([0],[0],[0,n],n,compact.RAW), # duplicate physical block
            ([0],[],[0,n],n,0),              # missing second block
            ([0,1],[],[0,n,n],n,0),          # zero-length/overlap
            ([0,1],[],[0,-1,2*n],2*n,0),
            ([0,1],[],[0,n,2*n],2*n+1,0),
            ([0,1],[],[0,n,compact.MAX_OFFSET+1],compact.MAX_OFFSET+1,0)]
        for good,bad,offsets,payload,raw in cases:
            with self.subTest(offsets=offsets),self.assertRaises(ValueError):
                compact.validate_directory([2*compact.SYMBOLS],good,bad,offsets,payload,raw,2)
        with self.assertRaises(ValueError):
            compact.validate_directory([compact.SYMBOLS+1],[0],[1],[0,n],n,compact.RAW,1)

    def test_compact_packet_damage_rejected_before_native_expansion(self):
        old=fixed_packet([compact.SYMBOLS],[0],[],record(compact.FRAME_OFFSET+7),torch.empty(0,dtype=torch.uint8))
        packet=compact.compact_packet(old,1)
        _,parts,_=unframe(packet,compact.METHOD,compact.STATE)
        parts['record_offsets'][1]+=1
        with self.assertRaises(ValueError):compact._parts(packet,1)
        packet=compact.compact_packet(old,1)
        _,parts,_=unframe(packet,compact.METHOD,compact.STATE)
        parts['records'][0]^=1
        with self.assertRaises(ValueError):compact._parts(packet,1)

    def test_accounting_includes_native_arena_output_and_complete_wire(self):
        old=fixed_packet([compact.SYMBOLS],[0],[],record(compact.FRAME_OFFSET+7),torch.empty(0,dtype=torch.uint8))
        packet=compact.compact_packet(old,2)
        instance=object.__new__(compact.CompactExans)
        instance.records=torch.empty(2*compact.RECORD,dtype=torch.uint8)
        a=instance.accounting(packet)
        self.assertEqual(a['packet_bytes'],len(packet.export()))
        self.assertEqual(a['native_record_arena_bytes_at_capacity'],2*compact.RECORD)
        self.assertEqual(a['padded_bf16_output_bytes'],compact.RAW)
        self.assertGreater(a['framing_and_alignment_bytes'],0)

    def test_native_codec_has_no_cpu_performance_fallback(self):
        with self.assertRaises(RuntimeError):compact.CompactExans('/missing','/missing','cpu',1)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1' and compact.triton is not None,'explicit software interpreter only')
    def test_triton_pack_scatter_and_header_kernels_match_cpu_reference(self):
        source=torch.cat([record(compact.FRAME_OFFSET+15),record(compact.FRAME_OFFSET+37,3)])
        offsets,expected=compact.compact_records(source)
        payload=torch.empty_like(expected)
        compact._pack[(2,compact.triton.cdiv(compact.RECORD,compact.BLOCK))](
            source,offsets,payload,STRIDE=compact.RECORD,B=compact.BLOCK,num_warps=4)
        self.assertTrue(torch.equal(payload,expected))
        headers=torch.empty((2,64),dtype=torch.uint8)
        compact._headers[(2,)](payload,offsets,headers,B=64,num_warps=4)
        self.assertTrue(torch.equal(headers,source.view(2,compact.RECORD)[:,:64]))
        target=torch.full((3*compact.RECORD,),255,dtype=torch.uint8)
        indexes=torch.tensor([0,2],dtype=torch.int32)
        compact._scatter[(2,compact.triton.cdiv(compact.RECORD,compact.BLOCK))](
            payload,offsets,indexes,target,STRIDE=compact.RECORD,B=compact.BLOCK,num_warps=4)
        self.assertTrue(torch.equal(target[:compact.RECORD],source[:compact.RECORD]))
        self.assertTrue(torch.equal(target[2*compact.RECORD:],source[compact.RECORD:]))
        self.assertTrue(bool((target[compact.RECORD:2*compact.RECORD]==255).all()))


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'),'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_compaction_and_scatter_match_cpu_bytes(self):
        source=torch.cat([record(compact.FRAME_OFFSET+15),record(compact.FRAME_OFFSET+37,3)])
        expected_offsets,expected_payload=compact.compact_records(source)
        offsets,payload=compact.compact_records(source.to('cuda:0'))
        self.assertTrue(torch.equal(offsets.cpu(),expected_offsets))
        self.assertTrue(torch.equal(payload.cpu(),expected_payload))
        target=torch.full((3*compact.RECORD,),255,dtype=torch.uint8,device='cuda:0')
        compact.expand_records(payload,offsets,torch.tensor([0,2],dtype=torch.int32,device='cuda:0'),target)
        recovered=target.cpu()
        self.assertTrue(torch.equal(recovered[:compact.RECORD],source[:compact.RECORD]))
        self.assertTrue(torch.equal(recovered[2*compact.RECORD:],source[compact.RECORD:]))
        self.assertTrue(bool((recovered[compact.RECORD:2*compact.RECORD]==255).all()))

        # The same mandatory CUDA case also exercises the actual unchanged C ABI,
        # including complete original tails and a genuinely non-fitting block.
        from scripts.run_exans_official_preflight import source_receipt
        root=Path(__file__).resolve().parents[1]
        reference=root/'benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153/results.json'
        q=json.loads(reference.read_text())['recipe']['selections']['exans_official_fixed_framed_t512']['codec']
        self.assertEqual(importlib.metadata.version('nvidia-libnvcomp-cu12'),'5.3.0.16')
        self.assertEqual(torch.cuda.get_device_capability(0),(8,6))
        self.assertIn('RTX A4500',torch.cuda.get_device_name(0))
        source_receipt(Path(q['source']))
        self.assertEqual(hashlib.sha256(Path(q['library']).read_bytes()).hexdigest(),q['library_sha256'])
        codec=compact.CompactExans(q['source'],q['library'],'cuda:0',2)
        try:
            words=torch.full((1,compact.SYMBOLS+17),0x3F80,dtype=torch.int16,device='cuda:0')
            tail=[0,0x8000,1,0x7F,0x80,0x3F80,0xBF80,0x7F80,0xFF80,
                  0x7FC1,0xFFFF,0xFF81,0x7F7F,0xFF7F,0x0081,0x8081,0x3F81]
            words[0,-17:]=torch.tensor([x if x<32768 else x-65536 for x in tail],dtype=torch.int16,device='cuda:0')
            expected=words.clone()
            first=codec.encode(words);wire=first.export()
            restored=Packet.restore(wire,'cuda:0')
            self.assertEqual(restored.export(),wire)
            value=codec.decode(restored)
            self.assertTrue(torch.equal(value,expected))
            self.assertTrue(torch.equal(words,expected))
            self.assertEqual(tuple(value.shape),tuple(words.shape))
            self.assertEqual(compact.packet_bytes(first),len(wire))
            a=codec.accounting(first)
            self.assertEqual(a['input_padding_bytes'],compact.RAW-34)
            self.assertEqual(a['fitting_blocks']+a['raw_fallback_blocks'],2)

            index=torch.arange(compact.SYMBOLS+17,dtype=torch.int32,device='cuda:0')
            wide=(((index%256)<<7)|(index&127)).to(torch.int16).view(1,-1)
            expected_wide=wide.clone()
            second=codec.encode(wide);second_wire=second.export()
            self.assertGreaterEqual(codec.accounting(second)['raw_fallback_blocks'],1)
            second_value=codec.decode(Packet.restore(second_wire,'cuda:0'))
            self.assertTrue(torch.equal(second_value,expected_wide))
            self.assertTrue(torch.equal(wide,expected_wide))
            self.assertTrue(torch.equal(value,expected))  # First output owns its storage.
            self.assertTrue(torch.equal(codec.decode(restored),expected))
            self.assertEqual(first.export(),wire)
            self.assertEqual(second.export(),second_wire)
        finally:
            codec.close()


if __name__=='__main__':unittest.main()
