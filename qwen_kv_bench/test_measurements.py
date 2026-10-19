"""Byte-accounting, sampling, dependencies, IO and native functional checks."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from qwen_kv_bench.measurements import measure_reference,measure_native

class RawAdapter:
    name='test_raw';state_bytes=11;state_info={'source':'test'};traits={'family':'test'};parent=None;dependency=None
    def __init__(self): self.encode_calls=0;self.decode_calls=0
    def encode(self,words,layer,kind):
        self.encode_calls+=1
        data=struct.pack('<II',*words.shape)+words.astype('<u2').tobytes()
        return SimpleNamespace(data=data,stats={'component_bytes':{'header':8,'words':words.nbytes},
            'mechanism_counts':{'record':1}})
    def decode(self,blob,layer,kind):
        self.decode_calls+=1
        r,c=struct.unpack_from('<II',blob)
        return np.frombuffer(blob,dtype='<u2',offset=8).reshape(r,c).copy()

class WrappedAdapter(RawAdapter):
    name='test_wrapper'
    def __init__(self,parent):
        super().__init__();self.parent=parent;self.dependency=parent.name
    def encode(self,*args): raise AssertionError('wrapper re-encoded parent')
    def encode_from_parent(self,parent):
        self.encode_calls+=1
        data=b'wrap'+parent.data
        return SimpleNamespace(data=data,stats={'component_bytes':{'wrapper':4,'parent':len(parent.data)},
            'wrapper_inner_byte_identity':True})
    def decode(self,blob,layer,kind):
        if blob[:4]!=b'wrap': raise AssertionError('wrong wrapper')
        return self.parent.decode(blob[4:],layer,kind)

class MeasurementTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(92)
        self.matrices=[tuple(rng.integers(0,65536,(25,14),dtype=np.uint16) for _ in range(2)) for _ in range(2)]
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_full_exact_components_and_buffered_io(self):
        adapter=RawAdapter()
        result=measure_reference(self.matrices,adapter,'cpu',self.out,repeats=2,measure_io=True)
        self.assertTrue(result['exact_verified']);self.assertEqual(result['coverage'],'full')
        self.assertFalse(result['size_is_estimated']);self.assertEqual(result['records_checked'],8)
        self.assertEqual(adapter.encode_calls,8)
        self.assertEqual(sum(result['component_bytes'].values()),result['serialized_bytes'])
        self.assertEqual(result['decoder_state_bytes'],11)
        self.assertTrue(result['buffered_io']['measured']);self.assertFalse(result['transfers']['measured'])
        self.assertEqual(list(self.out.iterdir()),[])
        json.dumps(result)
    def test_sampled_estimate_and_no_sample_budget(self):
        result=measure_reference(self.matrices,RawAdapter(),'cpu',self.out,max_records=3,io_bytes_limit=0)
        self.assertEqual(result['coverage'],'sampled');self.assertTrue(result['size_is_estimated'])
        self.assertEqual(result['records_measured'],3);self.assertEqual(result['records_total'],8)
        self.assertIsNotNone(result['estimated_full_cache_serialized_bytes'])
        self.assertFalse(result['sample_decode']['measured'])
        self.assertEqual(sum(result['component_bytes'].values()),result['serialized_bytes'])
        json.dumps(result)
    def test_wrappers_reuse_same_parent_bytes(self):
        parent=RawAdapter();wrapper=WrappedAdapter(parent)
        result=measure_reference(self.matrices,wrapper,'cpu',self.out,repeats=1)
        self.assertTrue(result['wrapper_parent_bytes_reused']);self.assertEqual(parent.encode_calls,8)
        self.assertEqual(wrapper.encode_calls,8);self.assertTrue(result['exact_verified'])
    def test_original_layer_ids_survive_selection(self):
        mapping={f'layer_{l:02d}_{k}':w for l,p in zip((3,7),self.matrices) for k,w in zip(('K','V'),p)}
        adapter=RawAdapter();seen=[];original=adapter.encode
        def enc(w,l,k): seen.append(l);return original(w,l,k)
        adapter.encode=enc
        result=measure_reference(mapping,adapter,'cpu',self.out,io_bytes_limit=0)
        self.assertEqual(sorted(set(seen)),[3,7]);self.assertEqual(result['layer_indices'],[3,7])
    def test_exact_failure_raises(self):
        adapter=RawAdapter();original=adapter.decode
        def bad(b,l,k):
            x=original(b,l,k);x.reshape(-1)[0]^=1;return x
        adapter.decode=bad
        with self.assertRaises(AssertionError): measure_reference(self.matrices,adapter,'cpu',self.out)
    def test_calculated_framing_is_distinct_from_record_payload(self):
        result=measure_reference(self.matrices,RawAdapter(),'cpu',self.out,io_bytes_limit=0)
        framing=32+8*(result['records_measured']+1)
        self.assertEqual(result['calculated_outer_framing_bytes'],framing)
        self.assertEqual(result['serialized_bytes'],result['actual_record_payload_bytes']+framing)
        self.assertIn('not serialized or persisted',result['byte_accounting'])
        self.assertEqual(list(self.out.iterdir()),[])
    def test_native_descriptor_and_explicit_rotation_provenance(self):
        from qwen_kv_bench import gpu_ops
        key=(8,'identity',104729)
        rotation=np.eye(8,dtype=np.float64)
        books={f'layer_00_{k}':np.full((16,8),16000,dtype=np.uint16) for k in ('K','V')}
        state=gpu_ops.make_state(books,device='cpu',rotations={key:rotation})
        matrices=[tuple(np.full((1,8),16001,dtype=np.uint16) for _ in range(2))]
        spec={'name':'test_binary','method':'binary','options':{'dimension':8,'transform':'identity'}}
        result=measure_native(matrices,spec,state,'cpu',self.out,repeats=1)
        self.assertTrue(result['host_metadata_prefetched'])
        self.assertTrue(result['runtime_diagnostics']['host_descriptor_guard_cost_in_decode'])
        self.assertEqual(result['native_entropy_backend'],'grouped_fixed_width_bitpacking')
        self.assertEqual(result['rotation_ids'][0]['sha256'],state.rotation_ids[key])
        self.assertTrue(result['rotation_ids'][0]['preloaded_before_measurement'])
        self.assertEqual(result['native_state_physical_bytes'],state.physical_nbytes)
        json.dumps(result)
    def test_native_cpu_packet_storage_and_exact(self):
        from qwen_kv_bench import gpu_ops
        for method in ('raw','exponent','e8_symbol'):
            state=gpu_ops.make_state({},device='cpu')
            result=measure_native(self.matrices,{'name':'test_native_'+method,'method':method,'options':{'step_log2':6} if method=='e8_symbol' else {}},state,'cpu',self.out,repeats=1,max_records=3,measure_io=True)
            self.assertTrue(result['exact_verified']);self.assertEqual(result['execution'],'tensor_native_cpu_functional')
            self.assertEqual(sum(result['component_bytes'].values()),result['serialized_bytes'])
            self.assertFalse(result['sample_packet_transfer_then_decode']['device_transfer_measured'])
            json.dumps(result)

if __name__=='__main__': unittest.main()
