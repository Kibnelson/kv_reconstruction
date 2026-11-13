# Cached-original dense control: software validation

Base: `b2dae7bc2446ad40b3333b322fc37e7bb7daeb39`. The earlier uploaded GPU
run is preserved. The new cached-original GPU timings and capture support
remain **pending NUC validation**. No GPU was attached for these checks.

Ten unique unittest methods passed. The explicit CPU Triton interpreter
compared original, cuBLAS and cached-original forward computations, then
compared full stream arenas and starts and decoded with original T. BF16
special patterns are included. Both alternate encoders inherit the original
entropy encoder and decoder methods unchanged.

A regression check verified that the cached fused forward object is created
once and reused, rather than constructing new CPU probability tables per
call. Review gates distinguish setup-caching improvement from a cuBLAS
computation improvement. The unchanged-original graph omission is explicit,
and unexpected graph errors retain their separate failure status.

A capture-style software workflow completed for `dense_cached` and
`dense_fast` on a small synthetic frozen reference. Both complete packets
matched the earlier reference SHA-256, and both fresh state-plus-packet
decoder processes reconstructed every word. CPU review flags stayed false.
The two strengthened gate tests were checked again separately. No
interpreter time is used as GPU performance evidence.

See [the focused NUC command](../../docs/t_frozen_speed_comparison.md).
