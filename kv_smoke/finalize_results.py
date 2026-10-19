"""Clarify the completed experiment report and build one reproducible archive."""
from __future__ import annotations
import datetime, hashlib, json, pathlib, zipfile

def main():
    root=pathlib.Path(__file__).resolve().parent
    out=root/"results"
    r=json.loads((out/"results.json").read_text())
    expected=len(r["configuration"]["lengths"])*3
    if len(r["cases"])!=expected:
        raise RuntimeError(f"Only {len(r['cases'])}/{expected} cases completed")
    assert all(v["all_roundtrips_exact"] for v in r["aggregate"].values())
    r["report_metadata"]={
        "finalized_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "aggregation":"Ratios use summed raw bytes divided by summed serialized bytes; long contexts have larger weight.",
        "fixtures":"Synthetic evaluation fixtures, not an untouched benchmark test split. No calibration performed.",
        "learned_or_codebook_parameter_array_bytes":0,
        "parameter_note":"No learned/codebook array in these controls; model/Python/runtime allocations are not included in serialized cache sizes.",
        "unit_tests_passed":10,
        "test_scope":"Exact codec and indexed cache, actual BF16 model integration, separate CPU tile-consumption check; no GPU/concurrency experiment."
    }
    (out/"results.json").write_text(json.dumps(r,indent=2)+"\n")
    report=(out/"smoke_test_report.md").read_text().replace("synthetic held-out","synthetic evaluation")
    rows=r["aggregate"]
    target=rows["lattice2_sign_coset"]
    correction_hist=target["accepted_correction_width_histogram"]
    accepted=sum(correction_hist.values())
    modal_width=max(correction_hist,key=correction_hist.get)
    modal_percent=100*correction_hist[modal_width]/accepted
    lines=["","## Interpretation of the completed run","",
      f"All {len(r['cases'])} configured prompt/length cases completed. Aggregate ratios are weighted by raw bytes, so the 2,048-token cases contribute more than shorter cases.",
      "",
      f"- The sign-preserving scalar lattice plus exact corrections saved **{target['space_saving_percent']:.2f}%** including all serialized framing.",
      f"- Exact exponent packing saved **{rows['exponent']['space_saving_percent']:.2f}%**. It outperformed this particular predictor/correction design.",
      f"- Symmetric scalar-lattice controls saved **{rows['lattice2_coset']['space_saving_percent']:.2f}%** and **{rows['lattice4_coset']['space_saving_percent']:.2f}%**; negative savings mean expansion after fallback/framing.",
      "- This validates the recovery mechanism, not the hypothesis that a useful lattice/learned predictor beats established lossless representations.",
      f"- The indexed raw control expands by {-rows['raw']['space_saving_percent']:.2f}% with these group and tile sizes. More compact headers or different grouping are packaging experiments; their benefit and access/workspace tradeoffs were not tested.",
      "- The generation path restores the entire prefix; the tile checker additionally holds full reference arrays. Neither is a compact-resident model or peak-GPU-memory experiment.",
      "- The reported generated-token list starts after the token chosen from prefill and contains eight following decode outputs.",
      "",
      "## What the correction widths suggest","",
      f"Among accepted sign-preserving predictor groups, {modal_percent:.1f}% require {modal_width} correction bits per value, in addition to two coordinate bits, one sign bit, scales, and framing. Corrections remain the dominant cost.",
      "",
      "The symmetric four-bit scalar-grid predictor usually needs fifteen correction bits, so the seed plus corrections already exceed sixteen raw bits before metadata. Raw fallback avoids that payload expansion but still pays container overhead.",
      "",
      "The next useful experiment is to reduce the maximum symbol prediction error within each group, rather than only average floating-point error. Rare escapes or grouping by exponent range may prevent one outlier from inflating the correction width for all 64 values; smaller groups trade that benefit against more metadata. These are proposed tests, not measured gains.",
      "",
      "## Per-case ratio ranges","",
      "| Method | Minimum ratio | Maximum ratio |",
      "|---|---:|---:|"]
    for method in rows:
        ratios=[c["codecs"][method]["ratio_vs_raw_payload"] for c in r["cases"]]
        lines.append(f"| {method} | {min(ratios):.4f}x | {max(ratios):.4f}x |")
    lines+=["","## Validation","",
       "Ten unit tests passed, including the reversible BF16 mapping, bit packing, exact correction recovery, exceptional-value fallbacks, full-cache reconstruction, partial tiles, and independent access to a later tile when an earlier payload is damaged.",
       "",
       "The included sample snapshot can be verified with NumPy alone using python verify_sample.py. Model weights and installed packages are excluded from the archive.",
       ""]
    (out/"smoke_test_report.md").write_text(report+"\n".join(lines))
    archive=root.parent/"kv_reconstruction_smoke_test.zip"
    files=[]
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix!=".pyc":
            files.append(path)
    manifest={"files":[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,
                      "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
    (out/"package_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    files.append(out/"package_manifest.json")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):
            z.write(path,"kv_smoke/"+str(path.relative_to(root)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,
                     "completed_cases":len(r["cases"]),
                     "ratios":{k:v["ratio_vs_raw_payload"] for k,v in rows.items()},
                     "savings_percent":{k:v["space_saving_percent"] for k,v in rows.items()}},indent=2))

if __name__=="__main__":
    main()

