#!/usr/bin/env python3
"""Read-only lower-stream probe on existing pinned pretrained cache captures.

CPU functional/size evidence only. The serialized lower-byte records exclude
exponents and decoder state, so this cannot demonstrate a complete-KV target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qwen_kv_bench import frontier_lower
from qwen_kv_bench.frontier_diagnostics import analyze

HEADER=struct.Struct("<4sHHIQQI")


def serialized_roundtrip(fields,meta,layer,start,kind):
    parts={name:bytes(value.cpu().numpy()) for name,value in sorted(fields.items())}
    descriptor=dict(layer=layer,start=start,kind=kind,lower=meta,field_bytes={k:len(v) for k,v in parts.items()})
    header=json.dumps(descriptor,sort_keys=True,separators=(",",":")).encode()
    payload=b"".join(parts.values())
    count=np.prod(meta["shape"]).item()
    wire=HEADER.pack(b"L8PK",1,0,len(header),count,len(payload),zlib.crc32(header+payload))+header+payload
    magic,version,flags,nheader,nvalues,npayload,crc=HEADER.unpack_from(wire)
    body=wire[HEADER.size:]
    assert magic==b"L8PK" and version==1 and flags==0 and len(body)==nheader+npayload and zlib.crc32(body)==crc
    loaded=json.loads(body[:nheader]);offset=nheader;restored={}
    for name,length in loaded["field_bytes"].items():
        restored[name]=torch.from_numpy(np.frombuffer(body[offset:offset+length],np.uint8).copy())
        offset+=length
    assert offset==len(body) and nvalues==count
    return frontier_lower.decode(restored,loaded["lower"]),len(wire)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    p.add_argument("--max-pairs",type=int,default=72)
    p.add_argument("--tile-tokens",type=int,default=64)
    p.add_argument("--head-dim",type=int,default=64)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if min(a.max_pairs,a.tile_tokens,a.head_dim)<1:p.error("Counts must be positive")
    if a.output.exists() and any(a.output.iterdir()):p.error("Output must be fresh or empty")
    torch.set_num_threads(4)
    manifest=json.loads(a.manifest.read_text());rows=[]
    for case in manifest["cases"]:
        source=a.manifest.parent/case["file"]
        with np.load(source,allow_pickle=False) as saved:
            layers=len(saved.files)//2
            matrices=[(saved[f"layer_{i:02d}_K"],saved[f"layer_{i:02d}_V"]) for i in range(layers)]
        digest=hashlib.sha256(b"".join(x.astype("<u2").tobytes() for pair in matrices for x in pair)).hexdigest()
        assert digest==case["raw_sha256"]
        slots=[(layer,start) for layer,(k,_) in enumerate(matrices) for start in range(0,len(k),a.tile_tokens)]
        chosen=np.linspace(0,len(slots)-1,min(a.max_pairs,len(slots)),dtype=np.int64)
        records=[]
        for index in chosen:
            layer,start=slots[index];k,v=matrices[layer]
            records.append(dict(layer=layer,start=start,tokens=min(a.tile_tokens,len(k)-start),head_dim=a.head_dim,
                k=k[start:start+a.tile_tokens].copy(),v=v[start:start+a.tile_tokens].copy()))
        result=dict(name=case["name"],tokens=case["tokens"],full_raw_sha256=digest,npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            pair_records=len(records),pair_records_total=len(slots),diagnostics=analyze(records),methods=[])
        values=sum(record["k"].size+record["v"].size for record in records)
        for mode in frontier_lower.MODES:
            size=0;storage={};counts={}
            for record in records:
                for kind in ("k","v"):
                    words=record[kind].astype(np.uint32)
                    lower=(((words>>8)&128)|(words&127)).astype(np.uint8)
                    fields,meta=frontier_lower.encode(torch.from_numpy(lower.copy()),mode=mode)
                    restored,bytes_=serialized_roundtrip(fields,meta,record["layer"],record["start"],kind)
                    assert np.array_equal(restored.numpy(),lower)
                    size+=bytes_;storage[meta["storage"]]=storage.get(meta["storage"],0)+1
                    for key,value in meta.get("mode_counts",{}).items():counts[key]=counts.get(key,0)+value
            framing=32+8*(2*len(records)+1)
            result["methods"].append(dict(mode=mode,roundtrip_exact=True,records_checked=2*len(records),raw_lower_bytes=values,
                actual_serialized_record_bytes=size,calculated_outer_framing_bytes=framing,total_lower_bytes=size+framing,
                lower_stream_bits_per_original_BF16_word=8*(size+framing)/values,storage_counts=storage,bitplane_mode_counts=counts,
                scope="Exact framed lower-byte streams only; exponent streams and decoder state are excluded. CPU size/functionality, no GPU or complete-KV savings claim."))
        rows.append(result);print(case["name"],[(r["mode"],round(r["lower_stream_bits_per_original_BF16_word"],4)) for r in result["methods"]],flush=True)
    report=dict(format="pretrained_frontier_lower_probe_v1",model=manifest["model"],revision=manifest["revision"],
        reused_capture_runtime=manifest["runtime"],input_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        probe_runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device="cpu",threads=torch.get_num_threads()),
        source_files_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(),Path(frontier_lower.__file__).resolve(),
                Path(__file__).resolve().parents[1]/"qwen_kv_bench/gpu_ops.py",
                Path(__file__).resolve().parents[1]/"qwen_kv_bench/frontier_diagnostics.py")},
        config=dict(tile_tokens=a.tile_tokens,max_pairs=a.max_pairs,head_dim=a.head_dim),cases=rows,
        scope="Reused held-out pretrained cache data, deterministic lossless transforms only, no fit or policy selection. Diagnostics are descriptive; no new prefill producer captures, Qwen result or GPU performance evidence. Only lower-stream sizes are encoded here.")
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
