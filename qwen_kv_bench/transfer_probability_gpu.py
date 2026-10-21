"""Hybrid packet backend: integer parameter work on Torch CPU/CUDA.

The CPU builds context from recovered symbols, uploads it, receives parameters,
and entropy-decodes the next original words with constriction. This exercises
the actual causal packet dependency, rather than a known-KV prediction replay.
It is deliberately instrumented reference plumbing, not a GPU entropy decoder.
"""
from __future__ import annotations

import numpy as np

from .transfer_gpu_predictor import TorchTransferPredictor, _copy_tensor, _require_torch
from .transfer_probability import Q, MAX_OFFSET


class TorchBackend:
    def __init__(self, base, state, device="cuda:0"):
        t=_require_torch(); self.base=base;self.state=state
        self.predictor=TorchTransferPredictor.from_reference(base,device=device,coefficient_dtype="int32")
        self.device=self.predictor.device;self.gpu=self.device.type=="cuda"
        self.coefficients=_copy_tensor(state.coefficients,self.device,t.int64)
        self.hint_delta=_copy_tensor(state.hint_delta,self.device,t.int64)
        self.hint_tensors=[]

    def predict(self, features, layer, group, positions):
        t=_require_torch()
        result=self.predictor.predict_block(_copy_tensor(features,self.device,t.int64),layer,group,
                                           _copy_tensor(positions,self.device,t.int64))
        return result.cpu().numpy()

    def offsets(self, context, layer, group, ids):
        t=_require_torch()
        if self.state.arm=="A":return np.zeros(len(context),np.int64)
        x=_copy_tensor(context,self.device,t.int64)
        acc=(x*self.coefficients[layer,group]).sum(dim=1,dtype=t.int64)*self.state.context_strength
        if self.state.hint_bits:
            z=_copy_tensor(ids,self.device,t.int64)
            acc=acc+self.hint_delta[layer,group].index_select(0,z)*Q*self.state.hint_strength
        return t.clamp(t.div(acc+2*Q,4*Q,rounding_mode="floor"),-MAX_OFFSET,MAX_OFFSET).cpu().numpy()

    def hint_ids(self, encoder, source):
        t=_require_torch();result=[]
        weights=_copy_tensor(encoder["weights"],self.device,t.int64)
        bias=_copy_tensor(encoder["bias"],self.device,t.int64)
        edges=_copy_tensor(encoder["edges"],self.device,t.int64)
        self.hint_tensors=[weights,bias,edges]
        for layer in range(len(source)):
            # The CPU bound is also used by the NumPy encoder. No wraparound
            # can occur in the device chunked sum, independent of sum order.
            values=source[layer]
            if encoder.get('feature_map')=='signed_absolute':values=np.concatenate((values,np.abs(values)),axis=1)
            maximum=max(abs(int(values.min())),abs(int(values.max())))
            if maximum*sum(abs(int(x)) for x in encoder["weights"][layer])+abs(int(encoder["bias"][layer]))+(1 << 15) >= 1 << 62:
                raise ValueError("hint INT64 projection overflow")
            x=_copy_tensor(values,self.device,t.int64)
            acc=t.zeros(len(x),dtype=t.int64,device=self.device)
            for begin in range(0,x.shape[1],64):
                acc+=(x[:,begin:begin+64]*weights[layer,begin:begin+64]).sum(dim=1,dtype=t.int64)
            score=((acc+(1 << 15)) >> 16)+bias[layer]
            result.append((score[:,None]>=edges[layer][None]).sum(dim=1).to(t.uint8).cpu().numpy())
        return np.stack(result)

    def synchronize(self):
        if self.gpu:_require_torch().cuda.synchronize(self.device)

    def statistics(self):
        extra=sum(x.numel()*x.element_size() for x in [self.coefficients,self.hint_delta,*self.hint_tensors])
        base=self.predictor.storage_statistics()
        return dict(device=str(self.device),gpu=self.gpu,
                    persistent_device_tensor_bytes=int(base["persistent_tensor_bytes"]+extra),
                    base_predictor_device_bytes=int(base["persistent_tensor_bytes"]),
                    probability_adapter_and_encoder_device_bytes=int(extra),
                    scope="integer parameter/hint tensors; CPU probability and entropy coder; CPU causal context with per-group H2D/D2H")
