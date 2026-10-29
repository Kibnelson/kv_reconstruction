# Combined transfer local software validation

**33 tests passed; 3 CUDA tests skipped. No GPU performance measurement.**

Checked calibration-only policy selection, receiver restrictions, full timing and link units, exact mixed decoding in the CPU Triton interpreter, compact ExANS pack/scatter/header kernels, complete wire accounting, raw overflow and physical-tail representation, source pins, and the original frozen evaluation contracts. All 12 new job schemas also validate against the actual published selections.

The 84 frozen source files match their original receipts. The other team's tracked worktree remains unchanged at `8a40dd9`. Python compilation and Bash syntax checks pass.

Actual mixed/native ExANS round trips, CUDA ownership checks and performance remain for the mandatory A4500 gate and benchmark. CPU fixtures and interpreter results do not establish GPU throughput. See `docs/combined_transfer_v1.md` for scope and the NUC command.
