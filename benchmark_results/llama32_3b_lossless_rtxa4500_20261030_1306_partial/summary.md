# KV transfer codecs on unsloth/Llama-3.2-3B

Captures: 7 documents x 8192 tokens; 28 layers, 8 KV heads x 128; torch 2.11.0+cu128, transformers 4.57.6.


## Lossless on SplitZip's datasets (our predictor calibrated on the wikitext set)

Per dataset: up to 4 windows of 8,192 consecutive tokens of its text (scripts/fetch_eval_texts.py; the windows of each dataset are in the code-book table), coded as one case, one stream per window; every entry a real stream decoded bit-exact on this GPU. SplitZip's code book (and the authors' codec) calibrated on the KV of wtrain0 (WikiText-2 train, the paper's protocol); our predictor on wtrain0, wtrain1, wtrain2, wtrain3, wtrain4, wtrain5.

**Bits per value**

| Method | WikiText-2 test |
| --- | --- |
| ours, one-step predictor (exact cost) | 8.869 |
| ours, strong predictor (exact cost) | 7.793 |
| ours, per layer, dense | 8.972 |
| ours, per layer, keys then values | 8.633 |
| ours, per layer, 4 head groups | 8.408 |
| ours, per layer, head by head | 8.279 |
| ours, per layer, rank 256 | 9.339 |
| ours, SplitZip format, rank 256 | 10.122 |
| SplitZip (authors' codec) | 12.050 |
| ExANS (authors' codec) | 10.872 |
| nvCOMP LZ4 | 16.013 |
| nvCOMP Cascaded | 16.006 |
| nvCOMP ANS | 13.078 |
| nvCOMP Zstd | 12.898 |
| AdaCode (authors' codec) | 11.370 |
| AdaCode, code from the coded KV (authors' codec) | 11.370 |
| exactkv (authors' codec) | 12.575 |
| exponent coders' floor: 8 + Huffman(exponent), DFloat11 / ZipNN format | 10.729 (computed) |
| exponent coders' floor: 8 + exponent entropy | 10.691 (computed) |
| nvCOMP Bitcomp | not measured |

**Decode time** (encode in brackets; [graph]: our decode as one CUDA graph replay)

| Method | WikiText-2 test |
| --- | --- |
| ours, per layer, dense | 124.30 ms (284) [graph 124.18] |
| ours, per layer, keys then values | 167.53 ms (417) [graph 167.88] |
| ours, per layer, 4 head groups | 211.68 ms (420) [graph 209.77] |
| ours, per layer, head by head | 452.52 ms (422) [graph 331.74] |
| ours, per layer, rank 256 | 91.36 ms (161) [graph 91.16] |
| ours, SplitZip format, rank 256 | 70.76 ms (4032) |
| SplitZip (authors' codec) | 12.17 ms (32) |
| ExANS (authors' codec) | 70.55 ms (703) |
| nvCOMP LZ4 | 32.65 ms (1514) |
| nvCOMP Cascaded | 26.71 ms (171) |
| nvCOMP ANS | 20.49 ms (47) |
| nvCOMP Zstd | 258.93 ms (2465) |
| AdaCode (authors' codec) | 12.15 ms (84) |
| AdaCode, code from the coded KV (authors' codec) | 12.16 ms (84) |
| exactkv (authors' codec) | 83.20 ms (22) |
| exponent coders' floor: 8 + Huffman(exponent), DFloat11 / ZipNN format | - |
| exponent coders' floor: 8 + exponent entropy | - |
| nvCOMP Bitcomp | not measured |

**SplitZip's code book on each dataset** (its Table 4 reports the top-16 coverage A->B)

| | WikiText-2 test |
| --- | --- |
| windows x tokens | 4 x 8,192 |
| top-16 coverage, WikiText-2 train book (A->B) | 99.91% |
| top-16 coverage, own book (B->B) | 99.91% |
| top-8 coverage, WikiText-2 train book | 94.56% |
| exponent entropy (bits) | 3.11 |
| SplitZip paper, top-16 A->B, Llama-3-8B (same family; not this model) | 99.85% |

**Transfer time per link** (all windows of a dataset; the longer of the wire time and the decode; ours = our fastest exact mode at that link, the speed-up against SplitZip)

| Link | WikiText-2 test |
| --- | --- |
| 1 Gb/s | 15556.7 ms (1.46x), per layer, head by head; 1.31x the best other, ExANS |
| 10 Gb/s | 1555.7 ms (1.46x), per layer, head by head; 1.31x the best other, ExANS |
| 25 Gb/s | 622.3 ms (1.46x), per layer, head by head; 1.31x the best other, ExANS |
| 50 Gb/s | 316.0 ms (1.43x), per layer, 4 head groups; 1.29x the best other, ExANS |
| 100 Gb/s | 167.5 ms (1.35x), per layer, keys then values; 1.22x the best other, ExANS |
| 200 Gb/s | 91.2 ms (1.24x), per layer, rank 256; 1.12x the best other, ExANS |
| 400 Gb/s | 70.8 ms (0.80x), SplitZip format, rank 256; 0.75x the best other, AdaCode, code from the coded KV |


## Links measured on this machine (scripts/pd_link_probe.py)

1 GPU(s): NVIDIA RTX A4500; 896 MiB per copy (one 8K Llama-3.2-3B document).

| GPU | to pinned host | from pinned host | to pageable host | from pageable host |
| --- | --- | --- | --- | --- |
| 0 | 26.4 GB/s (211 Gb/s) | 24.0 GB/s (192 Gb/s) | 14.2 GB/s (113 Gb/s) | 14.9 GB/s (119 Gb/s) |

One GPU allocated: no GPU-to-GPU pair measured. A transfer to another machine passes through the host legs above on both sides, so the slower pinned leg bounds any network link faster than it.

