"""Forward-pass parity, immutable decoder, and speed-accounting invariants."""
import json
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch


class SpeedTests(unittest.TestCase):
    def test_dense_streams_and_original_decoder_in_interpreter(self):
        env=dict(os.environ,TRITON_INTERPRET="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        run=subprocess.run([sys.executable,"-m","kv_compare.fast_dense_checks","--device","cpu","--interpret"],
            cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=180)
        self.assertEqual(run.returncode,0,run.stderr[-6000:])
        r=json.loads(run.stdout.splitlines()[-1])
        self.assertFalse(r["gpu_executed"])
        self.assertTrue(r["original_stream_bytes_exact"] and r["original_decoder_exact"] and r["features_and_priors_exact"])
        self.assertTrue(r["cached_features_and_priors_exact"] and r["cached_original_stream_bytes_exact"])

    def test_original_entropy_encoder_and_decoder_methods_are_inherited(self):
        from .fast_dense_encode import FastDenseEncode, CachedFusedDense
        from .layerwise import LayerwiseOneStep
        self.assertIs(FastDenseEncode.encode,LayerwiseOneStep.encode)
        self.assertIs(FastDenseEncode.decode,LayerwiseOneStep.decode)
        self.assertIs(CachedFusedDense.encode,LayerwiseOneStep.encode)
        self.assertIs(CachedFusedDense.decode,LayerwiseOneStep.decode)

    def test_cached_original_does_not_rebuild_forward_tables_per_call(self):
        import torch
        from .fast_dense_encode import CachedFusedDense
        consts=dict(Aq=torch.zeros(1,64,64,dtype=torch.int8))
        frozen_forward=Mock()
        frozen_forward.model.return_value=("features","priors")
        with patch("kv_compare.fast_dense_encode.fu.FusedOneStep",return_value=frozen_forward) as factory:
            codec=CachedFusedDense(consts,32,16)
            words=torch.zeros(1,32,64,dtype=torch.int16)
            self.assertEqual(codec.model(words),("features","priors"))
            self.assertEqual(codec.model(words),("features","priors"))
            factory.assert_called_once()
            self.assertEqual(frozen_forward.model.call_count,2)

    def test_known_uncapturable_original_is_not_an_unexpected_graph_error(self):
        from scripts.bench_t_speed import GRAPH_OMISSIONS,main
        self.assertEqual(set(GRAPH_OMISSIONS),{"dense_original"})
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            main(["--reference-run","unused","--output","unused", "--variants","dense_original","--executions","graph"])
        self.assertEqual(error.exception.code,2)

    def test_graphs_reject_cpu_instead_of_reporting_eager_speed(self):
        from .fast_dense_encode import CudaReplay
        with self.assertRaisesRegex(RuntimeError,"actual CUDA"):
            CudaReplay(lambda:None,"cpu")

    def test_link_model_uses_gigabits_and_milliseconds(self):
        from scripts.bench_t_speed import modeled_link_ms
        self.assertEqual(modeled_link_ms(10,125_000_000,1),1010)
        self.assertEqual(modeled_link_ms(10,125_000_000,10),110)
        with self.assertRaises(ValueError):modeled_link_ms(10,1,0)

    def test_rounds_do_not_multiply_the_paid_packet_size(self):
        from scripts.bench_t_speed import aggregate
        rows=[]
        for case in ("a","b"):
            for n,t in enumerate([10,20,90]):
                tm=dict(cuda_median_ms=t,wall_median_ms=t)
                rows.append(dict(variant="dense_fast",execution="eager",case=case,round=n,
                    values=1000,packet_bytes=1000,packet_sha256="same",bit_exact=True,
                    matches_reference_packet=True,extra_policy_tensor_bytes=0,serialized_state_bytes=5,
                    encode=tm,decode=tm,pair=tm))
        r=aggregate(rows,["a","b"],3,True)["dense_fast/eager"]
        self.assertEqual(r["packet_bytes"],2000)
        self.assertEqual(r["bpv"],8)
        self.assertEqual(r["encode_ms"],20)
        self.assertTrue(r["complete"])
        rows[-1]["packet_sha256"]="changed"
        with self.assertRaisesRegex(AssertionError,"frozen packet"):
            aggregate(rows,["a","b"],3,True)

    def test_graph_failure_does_not_make_partial_results_complete(self):
        from scripts.bench_t_speed import aggregate
        tm=dict(cuda_median_ms=10,wall_median_ms=10)
        row=dict(variant="hbh_learned",execution="graph",case="a",round=0,
            values=1000,packet_bytes=1000,packet_sha256="x",bit_exact=True,
            matches_reference_packet=True,graph_liveness_passed=True,extra_policy_tensor_bytes=20,
            serialized_state_bytes=30,encode=tm,decode=tm,pair=tm)
        r=aggregate([row],["a","b"],3,True)["hbh_learned/graph"]
        self.assertFalse(r["complete"])

    def test_cpu_and_smoke_results_cannot_pass_gpu_review_gate(self):
        from scripts.bench_t_speed import decision
        table={name:dict(complete=True,all_exact=True,packet_bytes=100,encode_ms=t,pair_ms=t)
               for name,t in [("dense_original/eager",100),("dense_cached/eager",80),("dense_fast/eager",50)]}
        r=dict(aggregate=table,gpu_executed=False,full_reference_coverage=True,arguments=dict(rounds=3))
        self.assertFalse(decision(r)["fast_dense_candidate_for_review"])
        r.update(gpu_executed=True,full_reference_coverage=True)
        self.assertTrue(decision(r)["fast_dense_candidate_for_review"])
        r.update(full_reference_coverage=False)
        self.assertFalse(decision(r)["fast_dense_candidate_for_review"])

    def test_setup_caching_gain_is_separate_from_forward_computation_gain(self):
        from scripts.bench_t_speed import decision
        def entry(t):return dict(complete=True,all_exact=True,packet_bytes=100,encode_ms=t,pair_ms=t)
        r=dict(gpu_executed=True,full_reference_coverage=True,arguments=dict(rounds=3),
               aggregate={"dense_original/eager":entry(100),"dense_fast/eager":entry(60)})
        self.assertTrue(decision(r)["fast_dense_vs_uncached_original_gain"])
        self.assertFalse(decision(r)["fast_dense_candidate_for_review"])
        r["aggregate"]["dense_cached/eager"]=entry(60)
        self.assertTrue(decision(r)["cached_dense_candidate_for_review"])
        self.assertFalse(decision(r)["fast_dense_candidate_for_review"])
        r["aggregate"]["dense_cached/eager"]=entry(80)
        self.assertTrue(decision(r)["fast_dense_candidate_for_review"])


if __name__=="__main__":
    unittest.main()
