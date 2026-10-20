"""Audit-protocol fixtures; these synthetic packets are not codec benchmarks.

The real codec's arithmetic decoder has its own tests. These fixtures target
integrity links, frozen DEV choices and comparisons/accounting in the reviewer.
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("extensions_audit",ROOT/"scripts/review_transfer_extensions.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture(root):
    states=root/"states";packets=root/"packet_checks"
    states.mkdir();packets.mkdir()
    metadata={"block2":0,"block_hidden_r1":1,"block_kv_r1":1,"block_hidden_shuffled_r1":1,
              "block2_coset_uniform_r2":0,"block2_coset_model_r2":0}
    state_info={}
    for name,rank in metadata.items():
        description=dict(format="team_transfer_reference_v1",geometry=dict(layers=2,channels=2,head_dim=2,heads=1),
                         mode="block",block_size=2,metadata_rank=rank,metadata_divisor=128,nblocks=[2,2])
        arrays={"fixture_coefficients":np.zeros((2,4+rank),np.int64),"fixture_scale":np.ones(8,np.float64)}
        digest=hashlib.sha256(audit.canonical(description))
        for key,array in sorted(arrays.items()):
            digest.update(key.encode());digest.update(array.dtype.str.encode());digest.update(audit.canonical(list(array.shape)));digest.update(array.tobytes())
        identity=digest.hexdigest();description["state_sha256"]=identity
        path=states/(name+".npz")
        np.savez_compressed(path,description=np.frombuffer(audit.canonical(description),np.uint8),**arrays)
        projector=dict(array_bytes=0,serialized_bytes=0)
        if rank:
            encoder=states/(name+"_encoder.npz")
            np.savez(encoder,mean=np.zeros(2),basis=np.zeros((2,rank)),scale=np.ones(rank))
            projector=dict(array_bytes=16+16*rank+8*rank,serialized_bytes=encoder.stat().st_size)
        state_info[name]=(path,identity,sum(x.nbytes for x in arrays.values()),projector)
    test=[]
    for start in (0,4):
        for name,rank in metadata.items():
            correction="coset_model" if "coset_model" in name else "coset_uniform" if "coset_uniform" in name else "direct"
            residue=2 if correction!="direct" else 0
            state_path,identity,resident,projector=state_info[name]
            words=np.full((2,4,4),0x3f80,np.uint16)
            raw=words.astype("<u2").tobytes()
            meta=b"\x00"*(2*4*rank)
            stream=b"\x01"*(20 if "hidden_r1" in name else 28)
            payload=meta+stream
            header=dict(format="team_transfer_reference_v1",state_sha256=identity,shape=[2,4,4],start=start,
                        kind="range",correction=correction,residue_bits=residue,metadata_bytes=len(meta),stream_bytes=len(stream),
                        decoded_sha256=audit.sha(raw),payload_sha256=audit.sha(payload))
            header_raw=audit.canonical(header)
            packet=audit.PREFIX.pack(b"KVTRF001",len(header_raw))+header_raw+payload
            stem=f"TEST_case_{name}_{start}"
            packet_path=packets/(stem+".bin");packet_path.write_bytes(packet)
            np.save(packets/(stem+".npy"),words,allow_pickle=False)
            inputs=[str(state_path),str(packet_path)]
            child=dict(inputs=inputs,decoded_sha256=hashlib.sha256(str(words.dtype).encode()+audit.canonical(words.shape)+words.tobytes()).hexdigest(),
                       stats=dict(exact_checksum_verified=True),decode_wall_seconds=.1)
            (packets/(stem+".json")).write_text(json.dumps(child))
            layer=[40.,115.] if name=="block_hidden_r1" else [96.,96.]
            stats=dict(packet_bytes=len(packet),framing_bytes=audit.PREFIX.size+len(header_raw),metadata_bytes=len(meta),entropy_stream_bytes=len(stream),
                       raw_fallback_bytes=0,fallback=None,state_sha256=identity,layer_ideal_bits=layer,ideal_bits=sum(layer))
            row=dict(phase="TEST",case="case",variant=name,start=start,tokens=4,values=32,raw_bytes=64,
                     packet_bytes=len(packet),bits_per_value=8*len(packet)/32,saving_fraction=1-len(packet)/64,
                     codec_stats=stats,metadata_rank=rank,residue_bits=residue,correction=correction,
                     bit_exact=True,packet_only_subprocess=True,decoder_inputs=inputs,
                     encode_wall_seconds=.2,decode_wall_seconds=.1,
                     shared_state_bytes=resident+projector["array_bytes"],shared_state_wire_bytes=state_path.stat().st_size+projector["serialized_bytes"])
            test.append(row)
    dev=[]
    for row in test[:len(metadata)]:
        if "coset_model" in row["variant"]:
            continue
        clone=copy.deepcopy(row);clone.update(phase="DEV",case="selection",start=8)
        dev.append(clone)
    for original in list(dev):
        if original["metadata_rank"]:
            other=copy.deepcopy(original);other["variant"]=other["variant"].replace("_r1","_r2")
            other["metadata_rank"]=2;other["packet_bytes"]+=100
            other["codec_stats"]["packet_bytes"]+=100
            other["codec_stats"]["metadata_bytes"]+=8
            other["codec_stats"]["entropy_stream_bytes"]+=92
            other["bits_per_value"]=other["packet_bytes"]*8/other["values"]
            dev.append(other)
        elif original["correction"]=="coset_uniform":
            other=copy.deepcopy(original);other["variant"]=other["variant"].replace("_r2","_r4")
            other["residue_bits"]=4;other["packet_bytes"]+=20
            other["codec_stats"]["packet_bytes"]+=20
            other["codec_stats"]["entropy_stream_bytes"]+=20
            other["bits_per_value"]=other["packet_bytes"]*8/other["values"]
            dev.append(other)
    selected={key:dict(rank=1,variant=f"block_{key}_r1",rule="minimum complete DEV packet bytes") for key in ("kv","hidden","hidden_shuffled")}
    selected["coset_plain"]=dict(residue_bits=2,variant="block2_coset_uniform_r2",rule="minimum complete DEV packet bytes")
    selected["kv_matched_hidden_rank"]=dict(rank=1,variant="block_kv_r1",rule="same transmitted-code budget")
    selected["hidden_shuffled_matched_hidden_rank"]=dict(rank=1,variant="block_hidden_shuffled_r1",rule="same transmitted-code budget")
    aggregate={}
    for name in metadata:
        rows=[r for r in test if r["variant"]==name]
        path,identity,resident,projector=state_info[name]
        packet_bytes=sum(r["packet_bytes"] for r in rows)
        aggregate[name]=dict(values=64,packet_bytes=packet_bytes,bits_per_value=8*packet_bytes/64,
                             shared_state_bytes=resident+projector["array_bytes"],shared_state_wire_bytes=path.stat().st_size+projector["serialized_bytes"],
                             model_statistics=dict(frozen_array_bytes=resident),projector_statistics=projector,
                             resident_decoder_state_bytes=resident,resident_encoder_projector_bytes=projector["array_bytes"],
                             serialized_decoder_state_bytes=path.stat().st_size,serialized_encoder_projector_bytes=projector["serialized_bytes"])
    result=dict(format="transfer_extension_study_v1",status="completed",exact_original_reproduction=False,
                arguments=dict(block_size=2,capture_tokens=16,act_dir=None),test_rows=test,dev_rows=dev,aggregate=aggregate,
                selected_on_dev=selected,development_split=dict(scale_start=0,scale_stop=8,selection_start=8,selection_tokens=4,dev_capture_tokens=16),
                scope=dict(codec_device="cpu"))
    (root/"results.json").write_text(json.dumps(result))
    return result


class TransferExtensionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.result=fixture(self.root)

    def save(self):
        (self.root/"results.json").write_text(json.dumps(self.result))

    def failures(self):
        self.save();r=audit.review(self.root)
        return [c["check"] for c in r["checks"] if not c["passed"]]

    def test_valid_evidence_does_not_require_compression_gain(self):
        r=audit.review(self.root)
        self.assertTrue(r["all_passed"])
        self.assertTrue(all(x["saving_fraction"]<0 for x in r["aggregate"]))
        self.assertFalse(r["exact_original_reproduction"])
        self.assertTrue(r["matched_hidden_kv_control"]["equal_rank"])
        self.assertEqual(len(r["packet_only_artifact_evidence"]),12)

    def test_layer_screen_pays_metadata_and_is_not_a_policy(self):
        r=audit.review(self.root)
        screen=r["layer_ideal_screening"]["block_hidden_r1"]
        self.assertEqual(screen["positive_layers"],[0])
        self.assertEqual(screen["negative_or_zero_layers"],[1])
        self.assertEqual(screen["rows"][0]["metadata_bits"],64)
        self.assertEqual(screen["rows"][0]["net_ideal_bits_saved_after_metadata"],48.)
        self.assertEqual(screen["rows"][1]["net_ideal_bits_saved_after_metadata"],-102.)
        self.assertIn("not measured layer packets",screen["scope"])
        self.assertIn("DEV",screen["required_next_step"])

    def test_nonwinning_dev_selection_fails(self):
        self.result["selected_on_dev"]["hidden"].update(rank=2,variant="block_hidden_r2")
        failures=self.failures()
        self.assertIn("DEV minimum frozen: hidden",failures)

    def test_test_residue_choice_cannot_change_after_dev(self):
        target=next(r for r in self.result["test_rows"] if r["variant"]=="block2_coset_uniform_r2")
        target["residue_bits"]=4
        self.assertTrue(any("TEST uses frozen residue_bits" in x for x in self.failures()))

    def test_variants_cannot_compare_different_original_values(self):
        # Change one variant's decoded target and update all local checksum
        # links: the cross-variant original-content check must still fail.
        row=next(r for r in self.result["test_rows"] if r["variant"]=="block_hidden_r1")
        packet=Path(row["decoder_inputs"][1]);raw=packet.read_bytes()
        _,size=audit.PREFIX.unpack_from(raw);header=json.loads(raw[audit.PREFIX.size:audit.PREFIX.size+size])
        words=np.load(packet.with_suffix(".npy"),allow_pickle=False);words[0,0,0]^=1
        np.save(packet.with_suffix(".npy"),words,allow_pickle=False)
        header["decoded_sha256"]=audit.sha(words.astype("<u2").tobytes())
        encoded=audit.canonical(header)
        packet.write_bytes(audit.PREFIX.pack(b"KVTRF001",len(encoded))+encoded+raw[audit.PREFIX.size+size:])
        child_path=packet.with_suffix(".json");child=json.loads(child_path.read_text())
        child["decoded_sha256"]=hashlib.sha256(str(words.dtype).encode()+audit.canonical(words.shape)+words.tobytes()).hexdigest()
        child_path.write_text(json.dumps(child))
        self.assertTrue(any("paired variants recover identical" in x for x in self.failures()))

    def test_dev_packet_price_cannot_omit_metadata(self):
        row=next(r for r in self.result["dev_rows"] if r["metadata_rank"])
        row["packet_bytes"]-=row["codec_stats"]["metadata_bytes"]
        self.assertTrue(any("DEV complete-packet accounting" in x for x in self.failures()))

    def test_decoder_cannot_receive_hidden_capture(self):
        self.result["test_rows"][0]["decoder_inputs"].append("hidden_capture.npz")
        self.assertTrue(any("only state and packet" in x for x in self.failures()))

    def test_packet_corruption_is_detected(self):
        packet=Path(self.result["test_rows"][0]["decoder_inputs"][1])
        raw=bytearray(packet.read_bytes());raw[-1]^=1;packet.write_bytes(raw)
        self.assertTrue(any("artifact evidence valid" in x for x in self.failures()))

    def test_state_array_corruption_is_detected(self):
        path=Path(self.result["test_rows"][0]["decoder_inputs"][0])
        with np.load(path,allow_pickle=False) as source:
            arrays={name:source[name].copy() for name in source.files}
        arrays["fixture_coefficients"][0,0]=5
        np.savez_compressed(path,**arrays)
        self.assertTrue(any("artifact evidence valid" in x for x in self.failures()))

    def test_mismatched_variant_grid_is_detected(self):
        self.result["test_rows"].pop()
        self.assertTrue(any("matched TEST grid" in x for x in self.failures()))

    def test_same_model_coset_cannot_claim_an_ideal_gain(self):
        row=next(r for r in self.result["test_rows"] if r["correction"]=="coset_model")
        row["codec_stats"]["ideal_bits"]-=1
        self.assertTrue(any("same-model coset ideal chain identity" in x for x in self.failures()))

    def test_metadata_not_free_and_endpoint_state_not_conflated(self):
        row=next(r for r in self.result["test_rows"] if r["metadata_rank"]==1)
        row["codec_stats"]["metadata_bytes"]=0
        self.result["aggregate"]["block_hidden_r1"]["resident_decoder_state_bytes"]+=100
        failures=self.failures()
        self.assertTrue(any("paid packet decomposition" in x for x in failures))
        self.assertTrue(any("metadata bytes fully charged" in x for x in failures))
        self.assertTrue(any("state separately account" in x for x in failures))

    def test_overlapping_scale_and_selection_fails(self):
        self.result["development_split"]["scale_stop"]=9
        self.assertIn("disjoint DEV scales and selection",self.failures())

    def test_cli_outputs_review_and_preserves_existing_output(self):
        output=self.root/"review"
        with mock.patch("sys.stdout",new_callable=io.StringIO) as captured:
            self.assertEqual(audit.main([str(self.root),"--output",str(output)]),0)
        self.assertTrue(json.loads(captured.getvalue())["all_passed"])
        self.assertTrue((output/"insights.json").exists())
        with mock.patch("sys.stderr",new_callable=io.StringIO), self.assertRaises(SystemExit):
            audit.main([str(self.root),"--output",str(output)])


if __name__=="__main__":
    unittest.main()
