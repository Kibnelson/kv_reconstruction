# Compact transfer results

This package contains byte-identical reports and a passing review that was executed against the full original run. It retains exact input token IDs, capture hashes, model identity and source hashes.

Frozen predictor states, encoded packets, decoded arrays and original hidden/KV captures are omitted. The compact package cannot rerun the full-artifact review without those original artifacts. Review assertions are historical executed evidence, not new checks performed by this packager. Capture NPZ hashes identify the omitted files; IDs are independently checked during extraction.
