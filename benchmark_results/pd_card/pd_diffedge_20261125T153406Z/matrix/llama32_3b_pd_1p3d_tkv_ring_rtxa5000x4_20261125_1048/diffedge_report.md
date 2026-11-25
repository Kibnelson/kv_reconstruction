# Diff edges: pages coded against their earlier versions (plain), then with T's prediction

Pages of 16 tokens x 28 layers. Every value coded with the value-slice coder (exact mode) against a prediction made from the page's earlier versions by plain arithmetic (keys turned by RoPE to the page's place), decoded and checked. Weights and spreads fitted on the even conversations; held out: the odd ones. Bits per value (raw 16), all layers coded. Agreement bucket: the median over the page's tokens of how many tokens up to them its text has agreed with its latest version's.

## mtrag_raw_interleaved_conversation

7,985 blocks in 4 conversations: 6,150 references (bytes P stored before), 1,835 distinct pages: 0 twins with D's bytes, 206 twins with other bytes, 545 moved (1 / 2 / 3+ earlier versions: 194 / 89 / 262), 1,084 new.

Held out: 130 moved pages, bits per value by agreement bucket (pages)

| Edge | All | 0-15 (6) | 16-63 (35) | 64-255 (55) | 256-1023 (34) | 1024+ (0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 9.27 | 9.23 | 9.26 | 9.24 | 9.33 | - |
| Its latest version (chain) | 9.02 | 9.21 | 9.08 | 9.06 | 8.86 | - |
| Latest, scale + offset per channel | 8.91 | 9.10 | 8.98 | 8.93 | 8.76 | - |
| Second order: latest + l (latest - previous) | 8.97 | 9.16 | 9.02 | 9.00 | 8.82 | - |
| a latest + b previous + c first | 8.97 | 9.16 | 9.02 | 9.01 | 8.83 | - |
| Composite: a latest + b drift + c first + d, per channel | 8.88 | 9.09 | 8.92 | 8.89 | 8.77 | - |
| T alone, its own spread (as now) | 8.32 | 8.36 | 8.33 | 8.30 | 8.33 | - |
| T alone, fitted spread | 8.30 | 8.36 | 8.32 | 8.28 | 8.31 | - |
| Blend: T + w (latest - T) | 8.12 | 8.20 | 8.13 | 8.10 | 8.12 | - |
| Best plain edge per page (or raw) | 8.86 | 9.08 | 8.92 | 8.87 | 8.74 | - |
| Best edge per page (or raw) | 8.11 | 8.20 | 8.12 | 8.09 | 8.11 | - |
| Rule by bucket, from training (t_blend / t_blend / t_blend / t_blend / latest) | 8.12 | 8.20 | 8.13 | 8.10 | 8.12 | - |

Values equal to the latest version's (V; keys are turned): 4.2%. Best edge per page: t_blend 118, t_fit 12.

Held out, by earlier versions (pages): diff of a diff needs two

| Edge | 1 (94) | 2 (17) | 3+ (19) |
| --- | ---: | ---: | ---: |
| Its first version (star) | 9.20 | 9.44 | 9.46 |
| Its latest version (chain) | 9.19 | 8.64 | 8.50 |
| Latest, scale + offset per channel | 9.07 | 8.52 | 8.49 |
| Second order: latest + l (latest - previous) | 9.13 | 8.63 | 8.48 |
| a latest + b previous + c first | 9.13 | 8.64 | 8.48 |
| Composite: a latest + b drift + c first + d, per channel | 9.01 | 8.61 | 8.47 |
| T alone, its own spread (as now) | 8.32 | 8.33 | 8.29 |
| T alone, fitted spread | 8.31 | 8.31 | 8.27 |
| Blend: T + w (latest - T) | 8.11 | 8.24 | 8.03 |

Held out, by what the page holds (pages)

| Edge | answer (2) | doc (128) |
| --- | ---: | ---: |
| Its first version (star) | 9.00 | 9.27 |
| Its latest version (chain) | 8.45 | 9.03 |
| Latest, scale + offset per channel | 8.42 | 8.92 |
| Second order: latest + l (latest - previous) | 8.42 | 8.98 |
| a latest + b previous + c first | 8.42 | 8.98 |
| Composite: a latest + b drift + c first + d, per channel | 8.42 | 8.89 |
| T alone, its own spread (as now) | 8.42 | 8.32 |
| T alone, fitted spread | 8.39 | 8.30 |
| Blend: T + w (latest - T) | 7.92 | 8.12 |

Held out, by layer group and K/V

| Edge | K 0-0 | V 0-0 | K 1-4 | V 1-4 | K 5-13 | V 5-13 | K 14-27 | V 14-27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 2.50 | 0.27 | 7.25 | 7.91 | 9.47 | 10.21 | 9.56 | 10.34 |
| Its latest version (chain) | 2.49 | 0.37 | 7.09 | 7.77 | 9.25 | 10.02 | 9.26 | 9.99 |
| Latest, scale + offset per channel | 2.50 | 0.37 | 6.96 | 7.70 | 9.17 | 9.88 | 9.16 | 9.84 |
| Second order: latest + l (latest - previous) | 2.50 | 0.37 | 7.11 | 7.76 | 9.25 | 9.91 | 9.26 | 9.84 |
| a latest + b previous + c first | 2.50 | 0.37 | 7.11 | 7.76 | 9.25 | 9.91 | 9.26 | 9.84 |
| Composite: a latest + b drift + c first + d, per channel | 2.48 | 0.37 | 7.02 | 7.70 | 9.16 | 9.84 | 9.13 | 9.75 |
| T alone, its own spread (as now) | 2.49 | 0.37 | 8.02 | 9.14 | 8.35 | 9.07 | 7.92 | 9.04 |
| T alone, fitted spread | 2.49 | 0.37 | 8.01 | 9.14 | 8.34 | 9.06 | 7.91 | 9.02 |
| Blend: T + w (latest - T) | 2.49 | 0.37 | 7.04 | 7.74 | 8.34 | 9.03 | 7.94 | 8.93 |
| Twin: P's page against D's copy | 3.08 | 5.09 | 3.10 | 3.74 | 4.23 | 4.99 | 4.18 | 4.89 |

Held out: 102 twins (answer 91, question 4, other 4, doc 3): 4.38 bits per value; 24.8% of their words equal to D's.

Training (fitted on these): 415 moved pages, bits per value by agreement bucket (pages)

| Edge | All | 0-15 (14) | 16-63 (68) | 64-255 (166) | 256-1023 (167) | 1024+ (0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 9.27 | 9.29 | 9.33 | 9.26 | 9.24 | - |
| Its latest version (chain) | 7.98 | 8.61 | 8.45 | 8.05 | 7.66 | - |
| Latest, scale + offset per channel | 8.06 | 8.59 | 8.43 | 8.10 | 7.83 | - |
| Second order: latest + l (latest - previous) | 7.95 | 8.56 | 8.41 | 8.01 | 7.64 | - |
| a latest + b previous + c first | 7.95 | 8.56 | 8.41 | 8.01 | 7.64 | - |
| Composite: a latest + b drift + c first + d, per channel | 7.88 | 8.47 | 8.32 | 7.93 | 7.59 | - |
| T alone, its own spread (as now) | 8.25 | 8.31 | 8.25 | 8.26 | 8.25 | - |
| T alone, fitted spread | 8.23 | 8.30 | 8.23 | 8.23 | 8.21 | - |
| Blend: T + w (latest - T) | 7.62 | 7.98 | 7.80 | 7.62 | 7.52 | - |
| Best plain edge per page (or raw) | 7.87 | 8.47 | 8.32 | 7.93 | 7.59 | - |
| Best edge per page (or raw) | 7.60 | 7.98 | 7.80 | 7.60 | 7.49 | - |
| Rule by bucket, from training (t_blend / t_blend / t_blend / t_blend / latest) | 7.62 | 7.98 | 7.80 | 7.62 | 7.52 | - |

Training (fitted on these): 104 twins (answer 91, question 8, other 4, doc 1): 3.86 bits per value; 28.0% of their words equal to D's.

The graph (MB): plain edges, then with T's

| | Held out | Training | All |
| --- | ---: | ---: | ---: |
| Raw, every block | 6,580.3 | 8,072.2 | 14,652.5 |
| Raw, distinct pages (references free) | 1,471.7 | 1,895.6 | 3,367.2 |
| Twins (D's copy) as diffs, or raw | 51.3 | 46.1 | 97.3 |
| Moved pages, best plain edge or raw | 132.1 | 374.7 | 506.8 |
| Moved pages, best edge with T's, or raw | 120.9 | 361.8 | 482.7 |
| Moved pages, latest edge or raw | 134.5 | 379.6 | 514.1 |
| New pages raw | 1,046.0 | 943.2 | 1,989.1 |
| **The plain graph** | 1,229.3 | 1,364.0 | 2,593.3 |
| The plain graph, latest edge only | 1,231.7 | 1,368.8 | 2,600.6 |
| **The graph with T's edges** | 1,218.2 | 1,351.0 | 2,569.2 |
| Plain graph under raw distinct pages | 16.5% | 28.0% | 23.0% |
| With T's edges, under raw distinct pages | 17.2% | 28.7% | 23.7% |

Chains and stars (moved pages; a twin is depth 1)

| Coded against | depth: pages | pages to decode, mean / p90 / max |
| --- | --- | --- |
| latest | 1: 189, 2: 100, 3: 77, 4: 55, 5: 42, 6: 40, 7: 37, 8: 5 | 6.9 / 20 / 28 |
| first | 1: 526, 2: 18, 3: 1 | 1.0 / 1 / 4 |

Rebuild: turning a version 0.13 ms per page, decoding 0.213 ms per page (batches of 32); recomputing: 0.55 ms per page (P's prefill in this run, 30 requests).

Fitted weights, pages with two versions or more, by agreement bucket (mean over layers): second order's l: K -0.22 / -0.09 / -0.03 / -0.02 / 0.00; V -0.21 / -0.12 / -0.06 / -0.05 / 0.00.
a latest + b previous + c first, (a, b, c): K (0.81, 0.17, 0.05) / (0.92, 0.07, 0.02) / (0.98, 0.01, 0.02) / (0.98, 0.01, 0.01) / (1.00, 0.00, 0.00); V (0.72, 0.20, 0.02) / (0.83, 0.13, -0.01) / (0.91, 0.07, -0.00) / (0.93, 0.06, -0.01) / (1.00, 0.00, 0.00).
Composite (a, b, c, d), mean over channels too: K (0.93, -0.19, 0.05, -0.00) / (0.96, -0.10, 0.02, -0.00) / (0.97, -0.04, 0.01, -0.00) / (0.98, -0.04, 0.01, -0.00) / (1.00, 0.00, 0.00, 0.00); V (0.87, -0.19, 0.03, 0.00) / (0.92, -0.13, 0.01, 0.00) / (0.95, -0.07, 0.01, 0.00) / (0.96, -0.06, 0.00, 0.00) / (1.00, 0.00, 0.00, 0.00).

Latest's scale and offset (mean over layers and channels): K 0.942, -0.0034; V 0.866, -0.0001.

T's edges: t2-hbh, set wikitext (smallest on the first training pages); layers 1.. with T's coder (uniform mode), layer 0 as the latest edge codes it.
Blend w (0: T alone, 1: the latest version), pages with two versions or more, by agreement bucket: K 1-4 0.85 / 0.92 / 0.97 / 0.99 / -; V 1-4 0.80 / 0.87 / 0.95 / 0.97 / -; K 5-13 0.32 / 0.49 / 0.67 / 0.75 / -; V 5-13 0.25 / 0.39 / 0.57 / 0.66 / -; K 14-27 0.25 / 0.35 / 0.50 / 0.58 / -; V 14-27 0.40 / 0.48 / 0.64 / 0.70 / -.
With one version: K 1-4 0.56 / 0.59 / 0.67 / 0.84 / -; V 1-4 0.53 / 0.53 / 0.60 / 0.76 / -; K 5-13 0.15 / 0.12 / 0.12 / 0.22 / -; V 5-13 0.14 / 0.11 / 0.11 / 0.18 / -; K 14-27 0.09 / 0.06 / 0.05 / 0.07 / -; V 14-27 0.15 / 0.10 / 0.09 / 0.11 / -.

Classes with fewer than 64 training tokens fitted as the nearest: 1024+ 1 as 256-1023 1, 1024+ 2+ as 256-1023 2+.

Spread multipliers chosen (by class: buckets with one version, then two or more): first 1/1/1/1/1/1/1/1/1.4/1; latest 1/1/1/1/1/1/1.4/1.4/2/1; latest_affine 1/1/1/1/1/1/1.4/1.4/2/1; second_order 1/1/1/1/1/1/1.4/1.4/2/1; multi_ref 1/1/1/1/1/1/1.4/1.4/2/1; composite 1/1/1/1/1/1/1.4/1.4/2/1; twin 1/1/1/1/1/1/1/1/1/1; t_fit 1/1/1/1/1/1/1/1/1/1; t_blend 1/1/1/1/1/1/1/1.4/2/1.

Coded a page and a slice at a time (the tables) against whole batches of 32 pages, all layers (bits per value; the difference is the coder's streams' overhead): latest 8.22 vs 8.20; second_order 8.19 vs 8.17; composite 8.11 vs 8.09; twin 4.12 vs 4.10.

Decoded exactly: latest 4,378 of 4,378, second_order 4,378 of 4,378, composite 4,378 of 4,378, first 4,360 of 4,360, latest_affine 4,360 of 4,360, multi_ref 4,360 of 4,360, t 3,270 of 3,270, t_fit 3,270 of 3,270, t_blend 3,270 of 3,270, twin 1,655 of 1,655 (ALL).

## Times (CUDA-synchronised, one GPU)

| Step | calls | ms per page | mean ms per call |
| --- | ---: | ---: | ---: |
| batch composite decode | 18 | 0.20 | 6.16 |
| batch composite encode | 18 | 0.11 | 3.33 |
| batch latest decode | 18 | 0.21 | 6.44 |
| batch latest encode | 18 | 0.11 | 3.35 |
| batch second_order decode | 18 | 0.21 | 6.35 |
| batch second_order encode | 18 | 0.11 | 3.38 |
| batch twin decode | 7 | 0.14 | 4.21 |
| batch twin encode | 7 | 0.11 | 3.12 |
| predict composite | 18 | 0.24 | 7.18 |
| predict first | 18 | 0.07 | 2.10 |
| predict latest | 18 | 0.06 | 1.95 |
| predict latest_affine | 18 | 0.10 | 3.18 |
| predict multi_ref | 18 | 0.15 | 4.49 |
| predict second_order | 18 | 0.12 | 3.67 |
| predict t | 58 | 0.50 | 15.64 |
| turn versions | 58 | 0.13 | 12.46 |

Sample for analysis: `diffedge_sample.npz` (27.2 MB): 96 moved pages with their latest, previous and first versions, 64 twins with D's copy; layers 0, 1, 3, 4, 12, 13, 26, 27, head 0's keys and values; per-page sizes in `diffedge_pages.json`.
