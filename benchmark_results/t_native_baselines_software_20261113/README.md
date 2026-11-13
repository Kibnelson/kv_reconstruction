# Native baseline comparison: software validation

The new runner compares the frozen fast dense T, learned head-by-head T, raw BF16, the authors' SplitZip and the authors' ExANS on identical captures. GPU measurements for this new comparison are pending the NUC run. AQUA-KV remains a separate lossy quality stage.

`python -m unittest kv_compare.test_native_compare -v` passes nine software tests. They cover both T adapters' byte-for-byte packet parity and exact decode in the CPU Triton interpreter, transfer of all 65,536 BF16 words, packet/state/shape corruption rejection, partial ExANS block accounting, a stubbed native subset-decode contract with paid raw fallback and tail, raw packet receiver integration, and incomplete comparison aggregation. The stub is an interface test, not native ExANS execution or timing.

The GPU benchmark rejects `TRITON_INTERPRET=1` before making a results directory. There is no CPU fallback. Six original T source blob hashes are unchanged. Shell syntax and Python compilation were checked. See `tests.txt` and `validation.json`.

No native SplitZip/ExANS GPU execution, performance gain, inference quality result or full comparison pass is claimed here. The new NUC smoke run must execute their kernels, ExANS' upstream test, native edge cases and independent packet-only receivers before the full four-window, three-round comparison.
