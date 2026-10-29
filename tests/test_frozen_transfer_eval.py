import hashlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import frozen_transfer_worker as worker
from scripts import bench_frozen_transfer as parent


class TimingAndReceiverTests(unittest.TestCase):
    def test_batch_statistics_keep_all_five_positive_samples(self):
        r = worker.timing_samples([1., 2., 3., 4., 5.], 1_000_000_000)
        self.assertEqual(r['wall_mean_ms'], 3000)
        self.assertEqual(r['samples_seconds'], [1., 2., 3., 4., 5.])
        self.assertAlmostEqual(r['raw_gbs']['mean'], sum(1 / x for x in range(1, 6)) / 5)
        self.assertEqual((r['warmup'], r['iters'], r['repeats']), (10, 50, 5))
        for samples in ([1.] * 4, [0.] * 5, [float('nan')] * 5, [float('inf')] * 5):
            with self.assertRaises(ValueError): worker.timing_samples(samples, 1000)
        with self.assertRaises(ValueError): worker.timing_samples([1.] * 5, 0)

    def test_author_timing_is_hash_bound_and_loads_only_selected_definitions(self):
        source = '''raise RuntimeError("unrelated module body must not execute")
def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
def bench(fn, device: str, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters
'''
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'bench_codec_throughput.py'; path.write_text(source)
            with self.assertRaises(ValueError): worker.load_author_bench(path, None)
            calls = []
            fake_torch = SimpleNamespace(device=lambda d: d,
                cuda=SimpleNamespace(synchronize=lambda d: calls.append('sync')))
            ticks = iter([100., 200.])
            def clock():
                calls.append('clock'); return next(ticks)
            fake_time = SimpleNamespace(perf_counter=clock)
            with patch.object(worker, 'AUTHOR_BENCH_SHA', hashlib.sha256(path.read_bytes()).hexdigest()), patch.object(worker, 'time', fake_time):
                bench, receipt = worker.load_author_bench(Path(raw), fake_torch)
                actual = bench(lambda: calls.append('fn'), 'cuda:0', 2, 3)
            self.assertEqual(calls, ['fn', 'fn', 'sync', 'clock', 'fn', 'fn', 'fn', 'sync', 'clock'])
            self.assertAlmostEqual(actual, 100 / 3)
            self.assertTrue(receipt['unchanged_ast'])

    def test_fresh_receiver_cannot_accept_source_kv_references(self):
        entry = dict(id='a', document='test_text_t512', shape=[28, 512, 2048],
            values=28 * 512 * 2048, packet='/compressed.kvc', packet_sha256='a' * 64,
            packet_bytes=1000, original_word_sha256='b' * 64, complete_bpv=8., state_sha256='c' * 64)
        q = dict(method='raw_bf16', tokens=512, codec={'kind':'raw_bf16'},
            entries=[entry, dict(entry, id='b')], execution='native', output='/receiver.json')
        worker.valid_receiver_request(q)
        with self.assertRaises(ValueError): worker.valid_receiver_request(dict(q, raw_entries=[]))
        with self.assertRaises(ValueError):
            worker.valid_receiver_request(dict(q, entries=[dict(entry, path='/original.npy'), entry]))
        with self.assertRaises(ValueError): worker.valid_receiver_request(dict(q, entries=[entry]))


def selections():
    result = {}
    for n in parent.LENGTHS:
        for method in parent.METHODS:
            entries = [dict(id=method + '_' + doc, document=doc + f'_t{n}',
                shape=[28, n, 2048], values=28*n*2048, packet='/frozen/' + method + doc,
                packet_sha256='a'*64, original_word_sha256=digest*64, packet_bytes=1000,
                complete_bpv=8*1000/(28*n*2048), state_sha256='b'*64)
                for doc, digest in [('test_text','c'), ('test_code','d')]]
            result[f'{method}_t{n}'] = dict(method=method, tokens=n, codec={'kind':method}, entries=entries)
    return result


class MatchedProtocolTests(unittest.TestCase):
    def test_worker_requires_actual_packets_matching_receivers_and_timing_scope(self):
        with tempfile.TemporaryDirectory() as raw:
            job=Path(raw); entries=[]; rows=[]; checks=[]
            for i,doc in enumerate(('test_text_t512','test_code_t512')):
                payload=('unit-test packet '+doc).encode(); path=job/f'{i}.kvc';path.write_bytes(payload)
                entry=dict(id=str(i),document=doc,shape=[28,512,2048],values=28*512*2048,
                    packet=str(path),packet_bytes=len(payload),packet_sha256=hashlib.sha256(payload).hexdigest(),
                    original_word_sha256=str(i+1)*64,state_sha256='c'*64,
                    complete_bpv=8*len(payload)/(28*512*2048))
                entries.append(entry)
                timing=dict(batched=worker.timing_samples([.01]*5,2*entry['values']),
                    isolated=dict(wall_median_ms=11.,samples=[{}]*5,warmup_samples=[{}]*2))
                rows.append(dict(entry,method='raw_bf16',tokens=512,round='before',execution='native',
                    original_word_exact=True,decoded_word_sha256=entry['original_word_sha256'],
                    frozen_packet_sha256=entry['packet_sha256'],frozen_packet_bytes=entry['packet_bytes'],
                    encode_packet_matches_frozen=True,encode=copy.deepcopy(timing),decode=copy.deepcopy(timing)))
                checks.append(dict(id=str(i),verified=True,shape=entry['shape'],word_sha256=entry['original_word_sha256'],packet_sha256=entry['packet_sha256']))
            request=dict(method='raw_bf16',tokens=512,codec={'kind':'raw_bf16'},entries=entries,
                execution='native',round='before')
            proof=dict(gpu_measured=True,no_source_kv_or_predictions=True,alternating_packets=True,
                independent_returned_outputs=True,runtime_policy={},runtime_policy_after={},
                graph_capture_count=0,checks=checks)
            (job/'receiver.json').write_text(json.dumps(proof))
            receiver={k:request[k] for k in ('method','tokens','codec','entries','execution')}
            receiver['output']=str(job/'receiver.json');(job/'receiver_request.json').write_text(json.dumps(receiver))
            result=dict(status='completed',gpu_measured=True,rows=rows,runtime_policy={},runtime_policy_after={},
                graph_capture_count=0,receiver_sha256=parent.sha(job/'receiver.json'),
                **{k:[dict(unchanged=True)]*2 for k in ('packet_immutability','frozen_packet_immutability','raw_packet_immutability')})
            self.assertEqual(len(parent.validate_worker(result,request,job)),2)
            variants=[]
            wrong=copy.deepcopy(result);wrong['rows'][0]['encode']['batched']['iters']=3;variants.append(wrong)
            wrong=copy.deepcopy(result);wrong['graph_capture_count']=1;variants.append(wrong)
            wrong=copy.deepcopy(result);wrong['rows'][0]['original_word_exact']=False;variants.append(wrong)
            for wrong in variants:
                with self.assertRaises(ValueError):parent.validate_worker(wrong,request,job)
            (job/'receiver_request.json').write_text(json.dumps(dict(receiver,raw_entries=entries)))
            with self.assertRaises(ValueError):parent.validate_worker(result,request,job)

    def test_all_methods_have_identical_words_and_original_geometry(self):
        frozen = selections(); parent.validate_selections(frozen)
        wrong = copy.deepcopy(frozen)
        wrong['team_dense_uniform_t512']['entries'][0]['original_word_sha256'] = 'e'*64
        with self.assertRaises(ValueError): parent.validate_selections(wrong)
        wrong = copy.deepcopy(frozen)
        wrong['raw_bf16_t512']['entries'][0]['shape'][0] = 4
        with self.assertRaises(ValueError): parent.validate_selections(wrong)

    def test_plan_brackets_own_methods_without_mutating_frozen_configuration(self):
        frozen = selections(); original = copy.deepcopy(frozen)
        plan = parent.job_plan(frozen)
        self.assertEqual(len(plan), 16)
        self.assertEqual(frozen, original)
        self.assertTrue(all(q['round']=='before' for _,q in plan[:6]))
        self.assertTrue(all(q['method'] in parent.OWN for _,q in plan[6:10]))
        self.assertTrue(all(q['round']=='after' for _,q in plan[10:]))
        for _, q in plan[6:10]:
            self.assertEqual(q['execution'], 'compact_search' if q['tokens']==512 else 'direct_search')
            self.assertEqual(q['raw_entries'], frozen[f"raw_bf16_t{q['tokens']}"]['entries'])

    def test_package_hash_paths_and_symlinks_are_checked(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); data=root/'result.json'; data.write_text('{}')
            receipt={'result.json':dict(bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())}
            manifest=root/'package_manifest.json'; manifest.write_text(json.dumps(receipt))
            self.assertEqual(parent.verify_package(root)['files'],1)
            data.write_text('xx')
            with self.assertRaises(ValueError): parent.verify_package(root)
            manifest.write_text(json.dumps({'../escape':receipt['result.json']}))
            with self.assertRaises(ValueError): parent.verify_package(root)
            manifest.write_text('[]')
            with self.assertRaises(ValueError): parent.verify_package(root)
            data.unlink(); data.symlink_to(manifest)
            manifest.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): parent.verify_package(root)

    def test_aggregation_keeps_batch_and_isolated_timing_distinct(self):
        rows=[]
        for i in range(2):
            rows.append(dict(tokens=512, method='team_dense_uniform', round='paired', document=str(i),
                packet_bytes=1000+1000*i, values=28*512*2048,
                encode=dict(batched=dict(wall_mean_ms=2+2*i),isolated=dict(wall_median_ms=5+4*i)),
                decode=dict(batched=dict(wall_mean_ms=3+2*i),isolated=dict(wall_median_ms=7+4*i))))
        r=parent.aggregate(rows)[0]
        self.assertEqual(r['batched'],dict(encode_ms=3,decode_ms=4))
        self.assertEqual(r['isolated'],dict(encode_ms=7,decode_ms=9))
        self.assertEqual(r['packet_bytes'],1500)
        with self.assertRaises(ValueError): parent.aggregate(rows[:1])

    def test_link_targets_keep_a_whole_round_for_each_objective(self):
        def item(method, phase, encode, decode):
            timing=dict(encode_ms=encode,decode_ms=decode)
            return dict(tokens=512,method=method,round=phase,packet_bytes=1000,
                batched=timing,isolated=dict(timing))
        rows=[item('splitzip_official_framed','before',1,8),
            item('splitzip_official_framed','after',9,2),
            item('team_dense_uniform','paired',4,4)]
        modeled=parent.link_models(rows)
        r=next(x for x in modeled['rows'] if x['method']=='team_dense_uniform' and x['effective_gbps']==1 and x['timing_basis']=='isolated')
        self.assertAlmostEqual(r['wire_ms'],.008)
        self.assertAlmostEqual(r['precompressed_serial_ms'],4.008)
        self.assertAlmostEqual(r['fresh_serial_ms'],8.008)
        targets=r['baseline_targets']['splitzip_official_framed']
        self.assertEqual(targets['precompressed']['round'],'after')
        self.assertEqual(targets['fresh']['round'],'before')
        self.assertAlmostEqual(targets['fresh']['serial_ms'],9.008)
        self.assertFalse(modeled['measured_network'])

    def test_summary_exposes_all_links_and_two_timing_bases(self):
        with tempfile.TemporaryDirectory() as raw:
            out=Path(raw)
            parent.report(dict(status='failed',gpu_measured=False,error='no CUDA'),out)
            text=(out/'summary.md').read_text()
            self.assertIn('batched',text);self.assertIn('isolated',text)
            self.assertIn('400',text);self.assertIn('no CUDA',text)


if __name__ == '__main__':
    unittest.main()
