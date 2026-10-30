# KV transfer codecs on unsloth/Llama-3.2-3B

Captures: 7 documents x 8192 tokens; 28 layers, 8 KV heads x 128; torch 2.11.0+cu128, transformers 4.57.6.



## Links measured on this machine (scripts/pd_link_probe.py)

1 GPU(s): NVIDIA RTX A4500; 896 MiB per copy (one 8K Llama-3.2-3B document).

| GPU | to pinned host | from pinned host | to pageable host | from pageable host |
| --- | --- | --- | --- | --- |
| 0 | 26.4 GB/s (211 Gb/s) | 24.0 GB/s (192 Gb/s) | 14.2 GB/s (114 Gb/s) | 15.0 GB/s (120 Gb/s) |

One GPU allocated: no GPU-to-GPU pair measured. A transfer to another machine passes through the host legs above on both sides, so the slower pinned leg bounds any network link faster than it.

