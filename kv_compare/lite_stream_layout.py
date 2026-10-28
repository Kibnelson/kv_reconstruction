"""Exact stream bookkeeping and calculated serial fetch targets; no timings."""
import math


def geometry(n, values_per_stream, block):
    if any(type(x) is not int or x < 1 for x in (n,values_per_stream,block)):
        raise ValueError('positive integer geometry required')
    streams=(n+values_per_stream-1)//values_per_stream
    programs=(streams+block-1)//block
    # All B lanes store their 32-bit rANS state, including inactive tail lanes.
    state_bytes=4*block*programs
    directory_bytes=8*programs
    return dict(values=n,values_per_stream=values_per_stream,lanes_per_program=block,
        streams=streams,programs=programs,state_bytes=state_bytes,directory_bytes=directory_bytes,
        state_and_directory_bpv=8*(state_bytes+directory_bytes)/n)


def fetch_budgets(aggregates, links=(1.,2.5,5.,10.)):
    baseline=next((r for r in aggregates if r['method']=='splitzip_official_framed'),None)
    if baseline is None: return []
    output=[]
    for row in aggregates:
        if row['mode']!='lossless' or row['kappa']!=0 or not row['method'].startswith('lite_'): continue
        for gbps in links:
            if not math.isfinite(gbps) or gbps<=0: raise ValueError('positive finite link required')
            saved_transfer=8*(baseline['packet_bytes']-row['packet_bytes'])/(gbps*1e6)
            budget=baseline['decode_ms']+saved_transfer
            output.append(dict(method=row['method'],effective_gbps=gbps,decode_budget_ms=budget,
                current_decode_ms=row['decode_ms'],required_reduction_factor=row['decode_ms']/budget if budget>0 else None,
                modeled_fetch_ms=row['decode_ms']+8*row['packet_bytes']/(gbps*1e6),
                baseline_modeled_fetch_ms=baseline['decode_ms']+8*baseline['packet_bytes']/(gbps*1e6),
                measured_network=False))
    return output
