# Independent CUDA-runtime peer-copy diagnostic (software validation)

The user-provided tail from job 30223, attempt
`links_20261115T163810Z_708926`, reports full-buffer direct-copy parity
failures for 0->1, 1->0, 1->2 and 2->1 at 64 MiB. It shows host and host_pipe
passes for 1->0, 1->2 and 2->1. The beginning of the attempt and source PIN
were not included in that excerpt. Its final diagnostic status is failed.
These byte-copy checks do not invoke compression codecs or validate serving.

The focused diagnostic now retains all three original routes and adds
`cudart_peer` in a fresh process per pair. The new worker imports no Torch:
it reads installed-package metadata to select the pinned CUDA 12.8 runtime,
uses cudaMalloc and explicit-device cudaMemcpyPeer, and enables both peer
directions. Each repetition uses a new CPU reference, full host/local-copy
controls on both devices, a destination containing the inverse reference,
and full destination/source parity checks. Failed comparisons preserve
mismatch counts, positions, hashes and poison counts in JSON. No host route
is substituted for a failed peer copy. Original direct failures continue
to fail the whole diagnostic even if the additional route succeeds.

Read-only platform context records peer capabilities, selected IOMMU boot
flags and visible GPU group links. It establishes no hardware cause and
changes no driver, BIOS, IOMMU, ACS or scheduler setting. The CUDA guide's
Linux bare-metal IOMMU limitation is a hypothesis to investigate, not a
confirmed diagnosis of this machine.

Twenty-one CPU/software tests passed, covering complete-buffer and source
corruption, untouched poison, fresh trials, local-control failure, missing
peer access, CUDA ABI argument order/pointer widths, runtime selection,
proof rejection, failure-report retention, and launcher/bootstrap behavior.
A real subprocess with no visible GPUs verifies that the worker imports
no Torch and keeps its failed JSON. The runtime memory operations and GPU
copies in the unit tests are simulated; these tests do not validate the
new CUDA worker on a GPU. Bash syntax and Python compilation also passed.

Package pins and allocation resources are unchanged. Old published run and
software records, original GPU algorithms and serving transports are unchanged.
The full serving matrix and real-node retest remain pending.
