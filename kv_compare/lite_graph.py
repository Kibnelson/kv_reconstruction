"""One-stream, one-cache CUDA graph decoder for unchanged lossless Lite math.

Received packets are validated at their actual length. Only decoder execution
is captured; input copies and independently owned output copies are paid on
every call. Fixed staging capacity lets different payload lengths share a graph.
"""
import hashlib
import json
import statistics
import time

import torch

from .causal_packet import Packet, json_bytes


def header_key(packet):
    """All semantic header fields, deliberately excluding payload length/data."""
    return json_bytes(packet.header)


def stage_packet(packet, payload, starts):
    """Copy new bytes AND offsets; caller validates the actual received packet."""
    if (packet.payload.ndim!=1 or packet.starts.ndim!=1 or
            packet.payload.dtype!=payload.dtype or packet.starts.dtype!=starts.dtype or
            packet.payload.device!=payload.device or packet.starts.device!=starts.device or
            packet.payload.numel()>payload.numel() or packet.starts.numel()!=starts.numel()):
        raise ValueError('packet differs from staging capacity/device/dtypes')
    payload[:packet.payload.numel()].copy_(packet.payload)
    starts.copy_(packet.starts)
    return 2*packet.payload.numel()+8*packet.starts.numel()


class GraphDecoder:
    def __init__(self, engine, warmup=3):
        if engine.device.type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('actual CUDA required; graph execution has no CPU substitute')
        if type(warmup) is not int or warmup<1: raise ValueError('positive warmup required')
        self.engine=engine
        self.device=torch.device('cuda',torch.cuda.current_device()) if engine.device.index is None else engine.device
        self.warmup=warmup; self.capture_count=0; self.setups=[]
        self._entry=None; self._stream=None; self._busy=False

    def _check_stream(self):
        stream=torch.cuda.current_stream(self.device)
        if self._stream is not None and self._stream!=stream.cuda_stream:
            raise RuntimeError('graph decoder is bound to one CUDA stream')
        return stream

    def _validate(self, packet):
        if (packet.payload.device!=self.device or packet.starts.device!=self.device or
                packet.payload.ndim!=1 or packet.starts.ndim!=1):
            raise ValueError('received packet must have flat buffers on the decoder device')
        # Crucially validate received length, never the padded staging packet.
        self.engine.validate(packet)
        if packet.header['kappa']!=0:
            raise ValueError('this graph experiment supports lossless mode only')

    def _release(self):
        if self._entry is not None:
            torch.cuda.synchronize(self.device)
            self._entry=None

    def close(self):
        if self._busy: raise RuntimeError('overlapping graph requests are unsupported')
        self._release()

    def _prepare_validated(self, packet):
        key=header_key(packet)
        if packet.header['raw']: return dict(rebuilt=False,raw_fallback=True,capture_count=self.capture_count)
        if self._entry is not None and self._entry['key']==key:
            return dict(rebuilt=False,capture_count=self.capture_count)
        self._release()
        current=self._check_stream(); torch.cuda.synchronize(self.device)
        before=torch.cuda.memory_allocated(self.device)
        before_reserved=torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        begin=time.perf_counter()
        shape=tuple(packet.header['shape']); _,capacity,programs=self.engine.layout(shape[1])
        payload=torch.empty(capacity,dtype=torch.int16,device=self.device)
        starts=torch.empty(programs,dtype=torch.int64,device=self.device)
        stage_packet(packet,payload,starts)
        static=Packet(json.loads(key),starts,payload)
        side=torch.cuda.Stream(device=self.device); side.wait_stream(current)
        # Initialize every decoder kernel and cuBLAS handle on the capture stream.
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                warm=self.engine.decode(static,validate=False)
            del warm
        side.synchronize(); current.wait_stream(side)
        graph=torch.cuda.CUDAGraph()
        # Exceptions propagate; there is no eager fallback labeled as a graph.
        with torch.cuda.graph(graph,stream=side):
            output=self.engine.decode(static,validate=False)
        current.wait_stream(side)
        graph.replay(); torch.cuda.synchronize(self.device)
        # RequestRope replaces its cache on range changes. Own the referenced
        # tables for the graph lifetime even if another eager call changes it.
        tables=tuple(self.engine.rope._tables.values())
        self._entry=dict(key=key,packet=static,graph=graph,output=output,tables=tables,side=side)
        self._stream=current.cuda_stream; self.capture_count+=1
        record=dict(rebuilt=True,capture_count=self.capture_count,
            header_sha256=hashlib.sha256(key).hexdigest(),shape=list(shape),start=packet.header['start'],
            cold_synchronized_wall_ms=(time.perf_counter()-begin)*1000,
            baseline_torch_bytes=before,persistent_torch_bytes=torch.cuda.memory_allocated(self.device),
            persistent_torch_bytes_delta=torch.cuda.memory_allocated(self.device)-before,
            baseline_torch_reserved_bytes=before_reserved,
            persistent_torch_reserved_bytes=torch.cuda.memory_reserved(self.device),
            persistent_torch_reserved_bytes_delta=torch.cuda.memory_reserved(self.device)-before_reserved,
            peak_torch_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            staging_payload_bytes=payload.numel()*2,staging_directory_bytes=starts.numel()*8,
            captured_output_bytes=output.numel()*2,
            scope='cold allocation + side-stream warmup + capture + first replay; Torch allocations exclude some driver graph metadata')
        self.setups.append(record)
        return record

    def prepare(self, packet):
        if self._busy: raise RuntimeError('overlapping graph requests are unsupported')
        self._busy=True
        try:
            with torch.cuda.device(self.device):
                self._check_stream(); self._validate(packet)
                return self._prepare_validated(packet)
        finally:
            self._busy=False

    def decode(self, packet):
        if self._busy: raise RuntimeError('overlapping graph requests are unsupported')
        self._busy=True
        try:
            with torch.cuda.device(self.device):
                self._check_stream(); self._validate(packet)
                if packet.header['raw']: return self.engine.decode(packet,validate=False)
                self._prepare_validated(packet)
                entry=self._entry
                stage_packet(packet,entry['packet'].payload,entry['packet'].starts)
                entry['graph'].replay()
                # Later replays must not overwrite an earlier returned result.
                return entry['output'].clone()
        finally:
            self._busy=False

    def measure_phases(self, packet, repeats=3):
        """Separate synchronized diagnostics, never ordinary benchmark samples."""
        if type(repeats) is not int or repeats<1: raise ValueError('positive repeats required')
        with torch.cuda.device(self.device):
            self.prepare(packet)
            if packet.header['raw']: raise ValueError('graph phase diagnostics require a compressed packet')
            entry=self._entry; samples={n:[] for n in ('validation','input_copy','replay','output_clone')}
            for _ in range(repeats):
                torch.cuda.synchronize(self.device); begin=time.perf_counter()
                self._validate(packet); torch.cuda.synchronize(self.device)
                samples['validation'].append(dict(wall_ms=(time.perf_counter()-begin)*1000))
                functions=(('input_copy',lambda:stage_packet(packet,entry['packet'].payload,entry['packet'].starts)),
                           ('replay',entry['graph'].replay),('output_clone',lambda:entry['output'].clone()))
                for name,fn in functions:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device); a.record(); begin=time.perf_counter()
                    value=fn(); submit=(time.perf_counter()-begin)*1000
                    b.record(); torch.cuda.synchronize(self.device)
                    samples[name].append(dict(wall_ms=(time.perf_counter()-begin)*1000,
                        cuda_ms=a.elapsed_time(b),submit_cpu_ms=submit))
                    del value
            phases={}
            for name,rows in samples.items():
                summary=dict(wall_median_ms=statistics.median(r['wall_ms'] for r in rows),samples=rows)
                if name!='validation':
                    summary.update(cuda_median_ms=statistics.median(r['cuda_ms'] for r in rows),
                        submit_cpu_median_ms=statistics.median(r['submit_cpu_ms'] for r in rows))
                phases[name]=summary
            return dict(gpu_measured=True,phases=phases,capture_count=self.capture_count,
                copied_input_bytes=2*packet.payload.numel()+8*packet.starts.numel(),
                returned_output_bytes=entry['output'].numel()*2,
                scope='separate synchronized phase diagnostics; per-phase barriers/CPU submission differ from ordinary total latency; do not add these into benchmark latency')
