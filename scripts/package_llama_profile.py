#!/usr/bin/env python3
"""Package small checked profile/official evidence; traces and all tensors stay local."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.package_llama_causal import sha


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path); p.add_argument("--destination",type=Path,required=True)
    args=p.parse_args(argv); run=args.run.resolve(); dest=args.destination
    if dest.exists(): raise ValueError("fresh report destination required")
    r=json.loads((run/"results.json").read_text())
    for stage in r["stages"].values():
        if stage.get("status")=="completed":
            for path,digest in stage.get("artifacts",{}).items():
                source=(run/path).resolve()
                if not source.is_relative_to(run) or sha(source)!=digest:
                    raise ValueError("stage artifact changed: "+path)
    if r["status"]=="completed" and not r["gpu_measured"]:
        raise ValueError("CPU evidence cannot be a completed GPU result")
    if r["status"]=="completed":
        required=list(r["recipe"]["stages"])
        if "aqua" in required: required.append("aqua_raw")
        if any(r["stages"].get(s,{}).get("status")!="completed" for s in required):
            raise ValueError("completion lacks requested stages")
    for filename,key in (("base_results.json","base_results_sha256"),("baseline_results.json","baseline_results_sha256")):
        if (run/filename).exists() and sha(run/filename)!=r["recipe"][key]:
            raise ValueError("reused baseline metadata changed")
    if r["stages"].get("exans",{}).get("status")=="completed":
        proof=json.loads((run/"exans_receiver.json").read_text())
        if not proof["gpu_measured"] or not proof["no_source_kv_or_predictions"]:
            raise ValueError("independent ExANS GPU receiver required")
        checks={e["id"]:e for e in proof["checks"]}
        for row in json.loads((run/"exans.json").read_text())["rows"]:
            c=checks[row["id"]]
            if not (c["verified"] and row["original_exact"] and row["status"]=="verified" and
                    c["word_sha256"]==row["original_word_sha256"]==row["decoded_word_sha256"] and
                    c["decoded_file_sha256"]==row["decoded_file_sha256"] and
                    abs(row["complete_bpv"]-8*row["packet_bytes"]/row["values"])<1e-10):
                raise ValueError("ExANS correctness/accounting proof differs")
    files=["results.json","summary.md","console.log","base_results.json","baseline_results.json",
        "profile.json","toolchain.json","exans_build.json","exans.json","exans_receiver.json","exans_validation.json",
        "aqua_environment.json","aqua_ppl_raw_official.json","aqua_ppl_higgs2_official.json","aqua_train_data.json","aqua_test_data.json","aqua_calibration_backup.json"]
    files += [str(p.relative_to(run)) for p in (run/"resume_history").glob("*.json")]
    files += [str(p.relative_to(run)) for p in (run/"logs").glob("*.log")]
    dest.mkdir(parents=True)
    manifest=dict(status=r["status"],gpu_measured=r["gpu_measured"],files={},local_traces={},
        scope="small diagnostic/native baseline evidence; no model/predictors/KV/packets/binaries or full traces")
    for name in files:
        source=run/name
        if not source.is_file(): continue
        if source.is_symlink() or not source.resolve().is_relative_to(run): raise ValueError("report symlink forbidden")
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
        manifest["files"][name]=dict(bytes=source.stat().st_size,sha256=sha(source))
    for source in (run/"traces").glob("*.json.gz"):
        manifest["local_traces"][str(source.relative_to(run))]=dict(bytes=source.stat().st_size,sha256=sha(source))
    (dest/"package_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(dest/"summary.md"); return 0


if __name__=="__main__": sys.exit(main())
