"""PD transfer protocol for vLLM prefill/decode disaggregation (docs/pd_protocol.md).

  wire       tensor bundles (one header + one byte blob) and the compaction of our codec's stream segments
  codecs     the transfer codecs: raw BF16, ours (T, five modes), SplitZip, AdaCode (the authors' code)
  transport  a paced virtual link (3-400 Gb/s) over a same-node shared-memory ring or TCP
  paged      gather / scatter of a request's KV between vLLM's paged cache and [layers, tokens, channels] words
  connector  the vLLM KV connector (TKVConnector) that ties them together
"""
