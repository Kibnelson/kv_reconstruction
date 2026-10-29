# Official ExANS synthetic correctness preflight

**Status: completed.** GPU performance measured: **False**.

Released author synthetic ExANS correctness/overflow test on RTX A4500. No codec throughput, Llama workload, NIC, SSD, vLLM serving or resident attention measurement.

The freshly built, unchanged author test exited zero: the synthetic compressible block round-tripped bit-exactly and the wide-exponent fixture correctly reported overflow.

Author pin: `de6c38bcc5438475bd1abf1e9b3d154cef414ef6`. Build target: `openlake_expans_test`, SM86, CUDA12.8 and nvCOMP5.3.0.16. The existing compiler-injected unique_ptr error-handler compatibility overload is disclosed; author files are unchanged.

The test uses a 999,424-byte synthetic input and a 770,048-byte fixed record. It has no model/dataset or timing loop and is silent on success; exit zero is the success evidence.

Fresh executable/library hashes, checked source files, runtime links and exact build/test commands are recorded. No prior test-executable checksum was assumed. Command wall times are provenance, not codec performance measurements.

This does not reproduce the author post's H100 throughput or establish compatible Llama/vLLM serving geometry. Keep matched framed transfer results and lossy AQUA quality results separate.
