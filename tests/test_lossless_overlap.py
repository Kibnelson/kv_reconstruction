"""Lossless overlap experiment protocol checks, without a tensor runtime."""
import copy
import json
import subprocess
import sys
import unittest

from scripts import bench_lossless_overlap as bench


def inputs():
    return json.loads((bench.ROOT/bench.frozen.FROZEN/'results.json').read_text())['recipe']['selections']


def policy():
    return bench.verify_previous()[0]


def paired_rows():
    return [dict(method=f'overlap_{mode}_k{k}',tokens=n,document=f'{doc}_t{n}',
        packet_bytes=1000,packet_sha256='a'*64,original_word_sha256='b'*64,
        rans_fields={f:dict(bytes=64,sha256='c'*64) for f in ('rans_words','rans_offsets','rans_relative')})
        for n in bench.LENGTHS for k in bench.KGRID for doc in ('test_text','test_code') for mode in bench.MODES]


def aggregate(method,e,d,phase='paired',n=512):
    return dict(tokens=n,method=method,round=phase,packet_bytes=1000,
        batched=dict(encode_ms=e,decode_ms=d),isolated=dict(encode_ms=e,decode_ms=d))


class PreviousAndPlanTests(unittest.TestCase):
    def test_completed_prior_words_policy_and_receipts_are_bound(self):
        p,r=bench.verify_previous()
        self.assertEqual(r['package']['files'],214)
        self.assertEqual([i for i,x in enumerate(p['masks']['2']) if x],[1,27])
        self.assertEqual(sum(p['masks']['14']),14)
        value=dict(p);digest=value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value),digest)
        self.assertEqual(set(p['reference_packets']),{f'decode_both_k{k}_t{n}' for k in (2,14) for n in (512,8192)})

    def test_all_modes_keep_existing_words_masks_encoder_and_bundle(self):
        source,p=inputs(),policy();original=copy.deepcopy(source)
        jobs=bench.job_plan(source,p)
        self.assertEqual(len(jobs),8)
        self.assertEqual(len(jobs)+len(bench.compact_plan(source,p))+len(bench.frozen.job_plan(source)),26)
        for _,q in jobs:
            mode,k=bench.method_parts(q['method'])
            self.assertEqual(q['codec']['mask'],p['masks'][str(k)])
            self.assertIs(q['codec']['selective'],True);self.assertIs(q['codec']['bundle'],True)
            self.assertIs(q['codec']['mapped_io'],True);self.assertIs(q['codec']['selective_decode'],True)
            self.assertIs(q['codec']['overlap_decode'],bench.FLAGS[mode])
            self.assertEqual(q['entries'],source[f"team_rank256_uniform_t{q['tokens']}"]['entries'])
            self.assertEqual(len(q['reference_entries']),2)
            self.assertTrue(all(x['method']==f'decode_both_k{k}' for x in q['reference_entries']))
        self.assertEqual(source,original)

    def test_undeclared_or_lossy_method_names_are_rejected(self):
        for value in ('overlap_lossy_k2','overlap_overlap_k28','overlap_overlap_k02','decode_mapped_k14','overlap_overlap_k14_extra'):
            with self.assertRaises(ValueError):bench.method_parts(value)


class ReceiverAndParityTests(unittest.TestCase):
    def requests(self):
        for _,q in bench.job_plan(inputs(),policy()):
            entries=[{**{k:e[k] for k in ('id','document','shape','values','original_word_sha256')},
                'packet':'/actual.kvc','packet_bytes':1000,'packet_sha256':'a'*64} for e in q['entries']]
            yield {k:q[k] for k in ('method','tokens','codec','execution')}|dict(entries=entries,output='/receiver.json')

    def test_fresh_receiver_restricts_flags_geometry_and_forbids_original_data(self):
        requests=list(self.requests());self.assertEqual(len(requests),8)
        for q in requests:bench.validate_receiver_request(q)
        base=requests[-1];bad=[]
        x=copy.deepcopy(base);x['raw_entries']=[];bad.append(x)
        x=copy.deepcopy(base);x['codec']['raw_words']='/oracle.npy';bad.append(x)
        x=copy.deepcopy(base);x['entries'][0]['predictions']='/oracle.npy';bad.append(x)
        x=copy.deepcopy(base);x['codec']['selective']=False;bad.append(x)
        x=copy.deepcopy(base);x['codec']['bundle']=False;bad.append(x)
        x=copy.deepcopy(base);x['codec']['mapped_io']=False;bad.append(x)
        x=copy.deepcopy(base);x['codec']['selective_decode']=1;bad.append(x)
        x=copy.deepcopy(base);x['codec']['overlap_decode']=False;bad.append(x)
        x=copy.deepcopy(base);x['codec']['overlap_decode']=1;bad.append(x)
        x=copy.deepcopy(base);x['entries'][0]['shape'][1]+=1;bad.append(x)
        x=copy.deepcopy(base);x['entries'][1]['document']=x['entries'][0]['document'];bad.append(x)
        for q in bad:
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)

    def test_every_mode_requires_identical_complete_wire_and_rans_fields(self):
        rows=paired_rows();proofs=bench.validate_packet_equivalence(rows)
        self.assertEqual(len(proofs),8)
        self.assertTrue(all(p['all_wire_equal'] and p['all_rans_fields_equal'] for p in proofs))
        bad=[]
        for key,value in (('packet_bytes',1001),('packet_sha256','d'*64),('original_word_sha256','d'*64)):
            x=copy.deepcopy(rows);x[1][key]=value;bad.append(x)
        x=copy.deepcopy(rows);x[1]['rans_fields']['rans_relative']['sha256']='d'*64;bad.append(x)
        x=copy.deepcopy(rows);x.append(copy.deepcopy(x[0]));bad.append(x)
        x=copy.deepcopy(rows);x.pop();bad.append(x)
        for values in bad:
            with self.assertRaises(ValueError):bench.validate_packet_equivalence(values)

    def test_progress_is_partial_until_all_groups_are_complete(self):
        self.assertEqual(bench.validate_packet_equivalence(paired_rows()[:1],complete=False),[])
        self.assertEqual(len(bench.validate_packet_equivalence(paired_rows()[:2],complete=False)),1)
        with self.assertRaises(ValueError):bench.validate_packet_equivalence(paired_rows()[:2])


class MeasurementTests(unittest.TestCase):
    def test_link_units_and_full_control_rounds_preserve_objectives(self):
        data=[aggregate('splitzip_official_framed',1,8,'before'),
              aggregate('splitzip_official_framed',9,2,'after'),aggregate('overlap_overlap_k2',4,4)]
        result=bench.link_models(data);self.assertFalse(result['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in result['rows']},{1,5,10,25,50,75,100,200,300,400})
        row=next(r for r in result['rows'] if r['method']=='overlap_overlap_k2' and r['effective_gbps']==1 and r['timing_basis']=='isolated')
        self.assertAlmostEqual(row['wire_ms'],.008)
        self.assertAlmostEqual(row['precompressed_serial_ms'],4.008)
        self.assertAlmostEqual(row['fresh_serial_ms'],8.008)
        self.assertEqual(row['splitzip_targets']['precompressed']['round'],'after')
        self.assertEqual(row['splitzip_targets']['fresh']['round'],'before')
        self.assertAlmostEqual(row['splitzip_targets']['precompressed']['decode_budget_ms'],2)

    def test_paired_insights_compare_fixed_changes_without_test_selection(self):
        rows=[aggregate(f'overlap_{mode}_k{k}',20 if mode == 'overlap' else 40,
            10 if mode == 'overlap' else 20,n=n)
            for n in bench.LENGTHS for k in bench.KGRID for mode in bench.MODES]
        result=bench.paired_insights(rows);self.assertEqual(len(result),8)
        self.assertTrue(all(r['complete_bytes_delta']==0 for r in result))
        self.assertTrue(all(r['encode_speedup']==2 and r['decode_speedup']==2 for r in result if r['mode']=='overlap'))

    def test_schedule_interval_intersection_is_nonadditive_and_validated(self):
        self.assertEqual(bench.interval_overlap_ms([0, 4], [2, 6]), 2)
        self.assertEqual(bench.interval_overlap_ms([0, 2], [2, 4]), 0)
        self.assertEqual(bench.interval_overlap_ms([0, 1], [2, 4]), 0)
        self.assertEqual(bench.interval_overlap_ms([1, 5], [2, 3]), 1)
        self.assertEqual(bench.interval_overlap_ms([2, 3], [1, 5]), 1)
        for a,b in (([2,1],[0,3]),([0,float('inf')],[1,2]),([0,float('nan')],[1,2]),
                    ([0],[1,2]),('a',[1,2]),([False,2],[1,2])):
            with self.assertRaises(ValueError):bench.interval_overlap_ms(a,b)

    def test_actual_cuda_gate_is_exactly_the_two_new_implementations(self):
        self.assertEqual(bench.CUDA_TESTS, ('tests.test_mapped_decode_partition.CudaTests',
            'tests.test_mixed_decode_overlap.CudaTests'))
        source=(bench.ROOT/'scripts/bench_lossless_overlap.py').read_text()
        self.assertIn('r.testsRun==2',source)
        self.assertIn('expected_tests=2',source)

    def test_parent_import_and_help_are_stdlib_only(self):
        subprocess.run([sys.executable,'-c','import sys; import scripts.bench_lossless_overlap; assert "torch" not in sys.modules'],
            cwd=bench.ROOT,check=True)
        value=subprocess.run([sys.executable,'scripts/bench_lossless_overlap.py','--help'],cwd=bench.ROOT,
            check=True,text=True,capture_output=True)
        self.assertIn('Lossless',value.stdout)


if __name__=='__main__':unittest.main()
