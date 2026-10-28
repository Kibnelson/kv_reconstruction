# Allocation initialization ablation

The measured finish implementation decodes unchanged V64/B32 packets in
88.224 ms, compared with 96.206 ms for its paired reference graph. Its separate
instrumented eager profile contains 3,727 CUDA fill kernels totaling
5.926874 ms: 2,829 float, 447 BF16, 449 int16 and two int64 fills. The inclusive
host `aten::fill_` count is 4,177; it is not the CUDA kernel count. These profile
durations identify an opportunity, not a prediction of graph latency savings.

`device_check` enables deterministic algorithms. PyTorch also defaults to
initializing otherwise uninitialized allocations in this mode. Most of the
observed fills therefore need not be codec initialization. Explicit `cur` and
`surprise` zeros and initial `prior` ones remain required and unchanged.

`kv_compare.lite_allocation.allocation_policy(fill: bool)` changes only
`torch.utils.deterministic.fill_uninitialized_memory`. It requires deterministic
algorithms already enabled and never changes their setting, warning policy,
matmul policy, fitted state, arithmetic or packets. It yields a JSON-compatible
record and restores the previous fill setting on normal exit and exceptions.
Nested contexts restore the enclosing setting.

The flag is process-global. Each benchmark worker must exclusively own all
PyTorch work for the whole context, including construction, warmup, capture,
replay and checks. This context is not a concurrent serving API. Every baseline
keeps its original worker, environment and execution configuration when retimed.
The existing native baseline device check does not enable deterministic
algorithms, so its protective empty-allocation fills are already inactive.
Record each control's deterministic setting, fill flag and effective fill state;
there is no need to create duplicate native fill-on/off arms. The controlled
new context applies to Lite's fill-on and fill-off arms, which both retain
deterministic algorithms.

## Full-write and read-before-write audit

This audit covers the current `FinishEngine` and inherited custom kernels with
valid packets produced by the checked encoder. It does not turn structurally
checked directories into validation of arbitrary hostile entropy streams.

| Allocation or access | Writer and coverage before use |
|---|---|
| `staging` encoder arena | Each program in `_encode_kernel` or `_res_encode` initializes its rANS lanes in registers, writes emitted words while moving its pointer backwards, then writes both 16-bit state halves for **all** block lanes and one start offset. Unused arena prefixes are never copied into the packet. Capacity reserves two emitted words per value plus both state halves per lane. |
| Encoder `starts` | Every program writes exactly one start entry. `compact` uses starts only after the encode launch has completed on the same stream. |
| Compacted packet `packed` | Prefix-summed positive program lengths partition the entire allocation. `_compact` reads only each written stream interval and writes every logical destination element once with its tail mask. |
| Lossless decoder `words` | For `n` values, `S=ceil(n/V)` and lane/iteration index `i*S+lane` cover every logical index exactly once. The valid mask excludes padded lanes and tail values. Every destination word is written before `finish_group` reads it. |
| Decoder state/refill reads | Each program first reads both state halves for all lanes, including inactive lanes written by the encoder. `_take` reads refill words only for active lanes needing renormalization. For valid encoder streams these reads stay in the written stream interval. |
| Residual decoder `sym` and `raw` | The same lane/iteration coverage writes every logical symbol and raw word. Nonescape raw entries are explicitly zeroed by the kernel; escape raw entries receive the decoded original word. |
| `cur` and `surprise` | Real `torch.zeros` allocations are retained. No assumption about unwritten future columns is needed by this ablation. Surprise updates keep their original Torch division, reduction and accumulation order. |
| Forward rotation output | `_rotate_group` writes every element under `i<n`, matching the allocated contiguous shape. Mate columns are checked to belong to the same group at fitted-state loading. |
| Finish `xrec` | `_finish_kernel` writes every logical `rec` element once as float32. It reads only decoded words, checked mate indices, initialized RoPE tables and fixed fitted indices. |
| Recovered words / lossy reconstruction | Each checked fitted `perm` is a permutation of all `W` columns. Groups partition those columns. Finish writes all group columns into each layer before the recovered tensor is returned; lossless encoding uses the initialized source directly. |
| Per-layer `f` | `_scatter_columns` stores all `W` columns through the checked permutation before the feature tensor is added to decoder context. |
| Calibration `layer_spread` | The inherited fallback writes every fitted group column before stacking the tensor. The new context changes no calibration arithmetic. |
| Graph staging payload | Every actual received payload byte and every directory entry is copied before replay. The capacity tail can contain old or uninitialized bytes; valid entropy streams read only the actual received stream intervals. Received packets are validated at their actual length, before creating the padded static packet. |
| Graph output | The decode graph fully writes its captured recovered output before cloning. Every call returns an independently owned clone, so later replays cannot change a prior result. |
| Raw fallback | A complete initialized input or received payload is cloned. The zero-length directory has no readable elements. All BF16 bit patterns remain verbatim. |
| PyTorch operator outputs | Existing gathers, conversions, GEMMs, elementwise expressions and reductions retain the same operations and initialized inputs. The context removes only protective empty-allocation fills; it does not replace or modify their output algorithms. |

Fitted-state validation checks bounds, unique permutations and same-group RoPE
mates. This ablation intentionally leaves all those checks and every inherited
source file unchanged. CUDA execution remains ordered on the same stream.

## Verification gates

CPU protocol tests check strict bool input, deterministic-mode preconditions,
nesting, exception restoration and unchanged real zeros/ones. Explicit Triton
interpreter tests compare complete compressed packet bytes and decoded words
between fill-on and fill-off. Test-only wrappers poison Python `empty`,
`empty_like` and `empty_strided` allocations with both NaNs/integer sentinels and
different finite values, exercising the noncontiguous input path too. C++
internal allocations are not intercepted by those wrappers. A raw-fallback
round trip covers all 65,536 BF16 words under both policies.

Actual-CUDA tests are separate gates: tiny exact packet/word parity, poisoned
allocation parity and fill-on/fill-off graph replay over A/B/A packets with
different compressed sizes, one capture and independently owned outputs. CPU
interpreter or offline compilation evidence cannot substitute for these tests.
Full real-packet independent receivers and paired timings are the benchmark
runner's additional gates. Until those run successfully, this is an allocation
policy experiment with no measured GPU speedup claim.
