# Standalone full inverse LUT: local validation

No benchmark or GPU performance measurement was run locally.

Protocol/integration: 14 passed and one actual-CUDA test skipped (15 total). Full-LUT proof and native replay: three passed and one actual-CUDA test skipped (four total). Python and Bash syntax checks passed. Logs include expected legacy interpreter warnings for NaN/Inf arithmetic; exact bit assertions passed.

The inverse matches all 655,360 fixed CDF slots. Independent AST normalization confirms that the layer decoder and post-inverse word/rANS arithmetic match the frozen source. Original packets and predictor coefficients stay fixed; the candidate adds 1.25 MiB of shared GPU lookup state and uses no baseline fallback.

All existing tracked files are unchanged. The NUC runner requires both actual-CUDA test methods before running 22 workers and 44 rows. New GPU speed remains unmeasured.
