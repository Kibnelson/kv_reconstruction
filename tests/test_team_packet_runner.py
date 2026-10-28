"""Protocol and measurement checks for the matched one-step packet runner."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.bench_team_packet import (LINKS, METHODS, aggregate, check_fit_metadata,
    cuda_check, identity, main, normalize_row, receive, setup_decode, transfer_models, validate_lengths)
from scripts.package_team_packet import verify


class RunnerTests(unittest.TestCase):
    def test_requested_link_models_keep_serial_overlap_and_fresh_separate(self):
        self.assertEqual(LINKS,(1.,100.,200.,300.,400.))
        a=dict(tokens=512,method='team_dense_uniform',packet_bytes=1_000_000,
               encode_ms=3.,decode_ms=2.)
        result=transfer_models([a])
        r=next(x for x in result['rows'] if x['effective_gbps']==1.)
        self.assertEqual((r['wire_ms'],r['precompressed_serial_ms'],r['ideal_overlap_ms'],r['fresh_serial_ms']),
                         (8.,10.,8.,13.))
        self.assertFalse(result['measured_network'])
        for bad in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):transfer_models([a],[bad])

    def test_lengths_are_validated_for_unchanged_unmasked_rank_tiles(self):
        self.assertEqual(validate_lengths('512,8192'),(512,8192))
        self.assertEqual(validate_lengths('256,2048'),(256,2048))
        for bad in ['64','128','512,512','768,8200','0','512,']:
            with self.assertRaises(ValueError):validate_lengths(bad)

    def test_training_receipt_preserves_caller_fields_added_by_state_writer(self):
        expected=dict(recipe_identity='a',capture_source={'model':'original'},calibration={'train_tokens':8192})
        meta=dict(expected,format='team_onestep_state_v1',vendor_source={'commit':'frozen'},rank=256)
        self.assertTrue(check_fit_metadata({'metadata':meta},expected))
        changed=dict(expected,capture_source={'model':'other'})
        with self.assertRaises(ValueError):check_fit_metadata({'metadata':meta},changed)
        with self.assertRaises(ValueError):check_fit_metadata({'metadata':dict(meta,rank=512)},expected)

    def test_graph_setup_receipt_is_json_serializable_when_codec_returns_self(self):
        class Codec:
            graph_cold_wall_ms=12.5
            graph_capture_count=1
            def graph_prepare(self,packet):
                self.packet=packet
                return self
        codec=Codec();receipt=setup_decode({'kind':'team'},codec,'wire')
        self.assertEqual(json.loads(json.dumps(receipt)),dict(graph_cold_wall_ms=12.5,graph_capture_count=1))
        self.assertEqual(codec.packet,'wire')

    def test_lengths_never_average_into_one_latency(self):
        rows=[]
        for n in (512,8192):
            for d in ('test_text','test_code'):
                row=dict(method='team_dense_uniform',document=f'{d}_t{n}',shape=[28,n,2048],
                    values=28*n*2048,packet_bytes=n*100,complete_bpv=8.,saved_percent=50.,original_exact=True,
                    encode={'wall_median_ms':n/100},decode={'wall_median_ms':n/200})
                rows.append(normalize_row(row))
        result=aggregate(rows)
        self.assertEqual([r['tokens'] for r in result],[512,8192])
        self.assertEqual([r['decode_ms'] for r in result],[2.56,40.96])
        self.assertTrue(all(r['documents']==['test_text','test_code'] for r in result))

    def test_kernel_diagnostic_uses_cuda_interval_not_host_wall(self):
        r=normalize_row(dict(method='team_dense_uniform',document='test_text_t512',shape=[28,512,2048],
            values=28*512*2048,packet_bytes=100,complete_bpv=8.,saved_percent=50.,original_exact=True,
            encode={'wall_median_ms':3.},decode={'wall_median_ms':2.},
            kernel_decode={'wall_median_ms':99.,'samples':[{'cuda_interval_ms':1.},{'cuda_interval_ms':3.}]}))
        self.assertEqual(aggregate([r])[0]['kernel_decode_ms'],2.)

    def test_cpu_is_rejected_before_parent_or_capture_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'
            with contextlib.redirect_stderr(io.StringIO()):
                code=main(['--device','cpu','--output',str(output),
                           '--source-run',str(Path(folder)/'missing'),
                           '--parent-run',str(Path(folder)/'missing-parent')])
            self.assertEqual(code,1)
            data=json.loads((output/'results.json').read_text())
            self.assertEqual(data['status'],'failed')
            self.assertFalse(data['gpu_measured'])
            self.assertIn('actual CUDA required',data['error'])
            self.assertFalse((output/'team_state').exists())
            self.assertTrue(verify(output,data))

    def test_receiver_schema_rejects_source_arrays_before_loading_codec(self):
        torch=SimpleNamespace()
        bad=dict(codec={'kind':'team'},entries=[],output='unused',inputs=['secret.npy'])
        with self.assertRaisesRegex(ValueError,'receiver request'):
            receive(bad,torch,SimpleNamespace(type='cuda'))
        q=dict(codec={'kind':'team'},entries=[{'path':'source.npy'}],output='unused')
        with self.assertRaisesRegex(ValueError,'source data'):
            receive(q,torch,SimpleNamespace(type='cuda'))

    def test_packager_rejects_cpu_completed_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'CPU evidence'):
                verify(Path(folder),dict(format='team_packet_v1',status='completed',gpu_measured=False))

    def test_recipe_hash_is_canonical_and_binds_lengths_methods_and_source(self):
        a=dict(lengths=[512,8192],methods=list(METHODS),source={'words_sha256':'a'})
        self.assertEqual(identity(a),identity(dict(reversed(list(a.items())))))
        self.assertNotEqual(identity(a),identity(dict(a,lengths=[8192])))
        self.assertNotEqual(identity(a),identity(dict(a,source={'words_sha256':'b'})))


if __name__=='__main__':unittest.main()
