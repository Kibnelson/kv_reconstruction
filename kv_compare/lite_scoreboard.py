"""Comparable complete-packet statistics and explicitly modeled link costs."""
import statistics

LINKS = (0.1, 0.25, 0.5, 1., 2.5, 5., 10., 25., 50., 100.)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["kappa"]), []).append(row)
    output = []
    for (method, kappa), group in groups.items():
        mean = lambda key: statistics.mean(r[key] for r in group)
        modes = {r["mode"] for r in group}
        if len(modes) != 1:
            raise ValueError("mixed lossy/lossless method group")
        output.append(dict(method=method, kappa=kappa, mode=group[0]["mode"], documents=[r["document"] for r in group],
            values=mean("values"), packet_bytes=mean("packet_bytes"), complete_bpv=mean("complete_bpv"),
            saved_percent=mean("saved_percent"), original_exact=all(r["original_exact"] for r in group),
            encode_ms=statistics.mean(r["encode"]["wall_median_ms"] for r in group),
            decode_ms=statistics.mean(r["decode"]["wall_median_ms"] for r in group),
            quality=[dict(document=r["document"], **r["quality"]) for r in group if r.get("quality")]))
    controls = {r["method"]: r for r in output if r["mode"] == "lossless"}
    for row in output:
        row["vs_baseline"] = {}
        if row["mode"] == "lossless":
            for name in ("raw_bf16", "splitzip_official_framed", "exans_official_fixed_framed"):
                if name in controls:
                    b = controls[name]
                    row["vs_baseline"][name] = dict(
                        packet_bytes_ratio=row["packet_bytes"] / b["packet_bytes"],
                        encode_time_ratio=row["encode_ms"] / b["encode_ms"],
                        decode_time_ratio=row["decode_ms"] / b["decode_ms"])
    return output


def network_model(aggregates, links=LINKS):
    """Serial estimates only. B is decimal bits/sec; packet size is bytes."""
    cases = []
    for gbps in links:
        if gbps <= 0:
            raise ValueError("positive effective bandwidth required")
        estimates = []
        for row in aggregates:
            if row["mode"] != "lossless" or not row["original_exact"] or row["kappa"] != 0:
                continue
            transfer = 8 * row["packet_bytes"] / (gbps * 1e9) * 1000
            estimates.append(dict(method=row["method"], transfer_ms=transfer,
                fresh_ms=row["encode_ms"] + transfer + row["decode_ms"],
                precompressed_ms=transfer + row["decode_ms"]))
        cases.append(dict(effective_gbps=gbps, estimates=estimates,
            best_fresh=min(estimates, key=lambda r: r["fresh_ms"])["method"] if estimates else None,
            best_precompressed=min(estimates, key=lambda r: r["precompressed_ms"])["method"] if estimates else None))
    return dict(measured_network=False,
        scope="serial size/timing model; no transport, serialization, staging, disk I/O, queueing, overlap or serving measurement",
        links=cases)


def validate_comparison(rows):
    """All methods must cover the same frozen documents and original words."""
    reference = {}
    coverage = {}
    for row in rows:
        identity = (tuple(row["shape"]), row["values"], row["original_word_sha256"])
        if row["document"] in reference and reference[row["document"]] != identity:
            raise ValueError("methods use different original KV")
        reference[row["document"]] = identity
        key = (row["method"], row["kappa"])
        if row["document"] in coverage.setdefault(key, set()):
            raise ValueError("duplicate comparison row")
        coverage[key].add(row["document"])
        if row["status"] != "verified" or (row["mode"] == "lossless" and not row["original_exact"]):
            raise ValueError("unverified or inexact lossless row")
    if not reference or any(docs != set(reference) for docs in coverage.values()):
        raise ValueError("method/document coverage differs")
