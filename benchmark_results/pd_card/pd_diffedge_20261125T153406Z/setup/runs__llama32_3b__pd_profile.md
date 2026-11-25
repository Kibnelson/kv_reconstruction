# Where the codecs spend their time (encode / decode on one GPU, no vLLM)

wall: the call with the GPU finished; issue: the CPU's time to issue it; GPU: its kernels' summed time; launches: its kernels. Issue near wall and GPU far below it: bound by issuing launches.

| Codec | Tokens | Encode wall / issue / GPU ms | launches | Decode wall / issue / GPU ms | launches | Bits |
| --- | --- | --- | --- | --- | --- | --- |
| ours-hbh | 512 | 4.8 / 4.8 / 4.2 | 222 | 23.8 / 0.6 / 21.4 | 945 | 8.1 |
| t2-hbh | 512 | 9.5 / 9.4 / 8.5 | 484 | 21.7 / 1.1 / 21.4 | 945 | 8.1 |
| t2-rank256 | 512 | 9.1 / 9.0 / 8.1 | 454 | 3.8 / 2.9 / 3.4 | 131 | 9.3 |
| raw | 512 | 0.0 / 0.0 / 0.0 | 0 | 0.2 / 0.2 / 0.2 | 28 | 16.0 |
| splitzip | 512 | 3.1 / 3.1 / 0.7 | 196 | 1.2 / 1.2 / 0.3 | 84 | 12.0 |
| adacode | 512 | 14.3 / 14.3 / 2.2 | 952 | 2.5 / 2.5 / 0.4 | 112 | 11.4 |
| exactkv | 512 | 11.8 / 11.8 / 1.7 | 896 | 2.5 / 2.5 / 1.5 | 152 | 12.6 |
| exans | 512 | 13.5 / 13.3 / 12.8 | 83 | 3.7 / 3.7 / 3.1 | 155 | 11.0 |
| aquakv-2 | 512 | 39.8 / 39.8 / 36.6 | 3286 | 16.0 / 15.9 / 10.7 | 1397 | 2.2 |
| ours-hbh | 2048 | 14.8 / 14.4 / 13.6 | 222 | 26.6 / 1.1 / 27.3 | 945 | 8.1 |
| t2-hbh | 2048 | 24.1 / 23.7 / 22.7 | 484 | 26.6 / 1.1 / 26.3 | 945 | 8.1 |
| t2-rank256 | 2048 | 17.7 / 17.2 / 17.2 | 454 | 6.3 / 2.9 / 5.9 | 131 | 9.3 |
| raw | 2048 | 0.0 / 0.0 / 0.0 | 0 | 0.8 / 0.2 / 0.7 | 28 | 16.0 |
| splitzip | 2048 | 3.1 / 3.0 / 1.9 | 196 | 1.5 / 1.1 / 1.4 | 84 | 12.0 |
| adacode | 2048 | 14.9 / 14.9 / 4.8 | 952 | 2.5 / 2.5 / 2.2 | 112 | 11.4 |
| exactkv | 2048 | 13.0 / 13.0 / 3.8 | 896 | 6.9 / 2.6 / 6.3 | 152 | 12.6 |
| exans | 2048 | 47.1 / 46.6 / 44.8 | 261 | 7.8 / 7.2 / 6.0 | 509 | 10.9 |
| aquakv-2 | 2048 | 144.3 / 144.3 / 143.0 | 3294 | 41.2 / 15.4 / 39.5 | 1341 | 2.1 |
| ours-hbh | 8192 | 60.2 / 59.5 / 58.0 | 446 | 73.2 / 36.6 / 73.3 | 1918 | 8.0 |
| t2-hbh | 8192 | 78.4 / 77.7 / 76.4 | 970 | 73.2 / 36.6 / 71.6 | 1918 | 8.0 |
| t2-rank256 | 8192 | 52.5 / 51.6 / 51.3 | 910 | 25.9 / 14.7 / 24.0 | 290 | 9.2 |
| raw | 8192 | 0.0 / 0.0 / 0.0 | 0 | 2.8 / 0.2 / 2.8 | 28 | 16.0 |
| splitzip | 8192 | 7.8 / 7.7 / 6.9 | 196 | 5.5 / 1.2 / 5.4 | 84 | 12.0 |
| adacode | 8192 | 22.3 / 22.3 / 20.4 | 952 | 8.3 / 2.5 / 8.2 | 112 | 11.4 |
| exactkv | 8192 | 18.4 / 18.4 / 11.9 | 896 | 25.0 / 2.7 / 24.2 | 152 | 12.6 |
| exans | 8192 | 188.7 / 186.7 / 179.6 | 1041 | 30.9 / 28.3 / 23.8 | 1949 | 10.9 |
| aquakv-2 | 8192 | 561.8 / 561.6 / 562.4 | 4746 | 161.5 / 38.9 / 159.5 | 1341 | 2.1 |
| ours-hbh | 16384 | 121.9 / 120.5 / 120.2 | 892 | 153.4 / 135.9 / 152.0 | 3806 | 10.9 |
| t2-hbh | 16384 | 160.2 / 158.9 / 156.2 | 1940 | 154.3 / 136.3 / 151.5 | 3806 | 10.9 |
| t2-rank256 | 16384 | 106.0 / 104.8 / 104.1 | 1820 | 54.7 / 39.1 / 51.8 | 552 | 10.4 |
| raw | 16384 | 0.0 / 0.0 / 0.0 | 0 | 5.6 / 0.2 / 5.6 | 28 | 16.0 |
| splitzip | 16384 | 14.7 / 14.4 / 13.6 | 196 | 10.7 / 1.2 / 10.6 | 84 | 12.0 |
| adacode | 16384 | 40.7 / 40.5 / 39.1 | 952 | 16.2 / 2.6 / 16.1 | 112 | 11.4 |
| exactkv | 16384 | 26.3 / 26.2 / 22.6 | 896 | 49.2 / 2.9 / 48.4 | 152 | 12.6 |
| exans | 16384 | 377.9 / 374.1 / 359.4 | 2080 | 61.7 / 56.4 / 47.6 | 3869 | 10.9 |
| aquakv-2 | 16384 | 1122.8 / 1122.7 / 1127.6 | 6682 | 321.6 / 76.9 / 320.2 | 1341 | 2.1 |

## T and T2 against their calls as of job 30436 (wall ms)

Encode: each layer's sums once / twice (TKV_ENCODE_ONCE). Decode: graphs / launches one by one with the fused group product / one by one with the column copy, product and add (TKV_DECODE_GRAPHS, TKV_GROUP_PRODUCT).

| Codec | Tokens | Encode once | Encode twice | Decode (graphs) | Decode eager | Decode as before |
| --- | --- | --- | --- | --- | --- | --- |
| ours-hbh | 512 | 4.8 | 7.1 | 23.8 | 22.3 | 30.8 |
| t2-hbh | 512 | 9.5 | 11.9 | 21.7 | 26.6 | 40.1 |
| t2-rank256 | 512 | 9.1 | 10.5 | 3.8 | 3.8 | 3.8 |
| ours-hbh | 2048 | 14.8 | 22.9 | 26.6 | 27.3 | 40.6 |
| t2-hbh | 2048 | 24.1 | 32.1 | 26.6 | 27.2 | 40.2 |
| t2-rank256 | 2048 | 17.7 | 19.6 | 6.3 | 6.3 | 6.4 |
| ours-hbh | 8192 | 60.2 | 91.8 | 73.2 | 73.2 | 97.0 |
| t2-hbh | 8192 | 78.4 | 111.0 | 73.2 | 73.5 | 96.9 |
| t2-rank256 | 8192 | 52.5 | 58.6 | 25.9 | 25.9 | 25.9 |
| ours-hbh | 16384 | 121.9 | 186.3 | 153.4 | 153.4 | 200.7 |
| t2-hbh | 16384 | 160.2 | 223.8 | 154.3 | 154.5 | 202.5 |
| t2-rank256 | 16384 | 106.0 | 117.9 | 54.7 | 54.7 | 54.6 |

## By kernel kind (count, GPU ms)

- ours-hbh, 512 tokens, encode: products (int8 GEMM) 55 x, 1.7 ms; our encoder 28 x, 1.1 ms; our features 82 x, 0.7 ms; copies and fills 56 x, 0.6 ms; packing 1 x, 0.1 ms
- ours-hbh, 512 tokens, decode: our decoder 448 x, 17.0 ms; other 420 x, 3.3 ms; products (int8 GEMM) 27 x, 0.8 ms; copies and fills 50 x, 0.3 ms
- t2-hbh, 512 tokens, encode: copies and fills 308 x, 4.6 ms; products (int8 GEMM) 65 x, 2.0 ms; our encoder 28 x, 1.1 ms; our features 82 x, 0.7 ms; packing 1 x, 0.1 ms
- t2-hbh, 512 tokens, decode: our decoder 448 x, 17.0 ms; other 420 x, 3.3 ms; products (int8 GEMM) 27 x, 0.8 ms; copies and fills 50 x, 0.3 ms
- t2-rank256, 512 tokens, encode: copies and fills 280 x, 4.1 ms; our encoder 28 x, 1.8 ms; products (int8 GEMM) 64 x, 1.3 ms; our features 81 x, 0.8 ms; packing 1 x, 0.1 ms
- t2-rank256, 512 tokens, decode: our decoder 28 x, 2.2 ms; products (int8 GEMM) 54 x, 1.0 ms; copies and fills 49 x, 0.2 ms
- raw, 512 tokens, decode: copies and fills 28 x, 0.2 ms
- splitzip, 512 tokens, encode: other 56 x, 0.5 ms; copies and fills 140 x, 0.3 ms
- splitzip, 512 tokens, decode: other 56 x, 0.2 ms; copies and fills 28 x, 0.1 ms
- adacode, 512 tokens, encode: copies and fills 784 x, 1.9 ms; other 112 x, 0.3 ms; packing 56 x, 0.1 ms
- adacode, 512 tokens, decode: other 56 x, 0.3 ms; copies and fills 56 x, 0.2 ms
- exactkv, 512 tokens, encode: copies and fills 840 x, 1.2 ms; exactkv 56 x, 0.6 ms
- exactkv, 512 tokens, decode: exactkv 56 x, 1.3 ms; copies and fills 96 x, 0.2 ms
- exans, 512 tokens, encode: other 10 x, 11.0 ms; copies and fills 73 x, 1.8 ms
- exans, 512 tokens, decode: copies and fills 152 x, 1.9 ms; other 3 x, 1.2 ms
- aquakv-2, 512 tokens, encode: copies and fills 3058 x, 24.3 ms; products (int8 GEMM) 228 x, 12.3 ms
- aquakv-2, 512 tokens, decode: products (int8 GEMM) 110 x, 5.7 ms; copies and fills 1287 x, 5.1 ms
- ours-hbh, 2048 tokens, encode: products (int8 GEMM) 55 x, 5.6 ms; our encoder 28 x, 3.0 ms; copies and fills 56 x, 2.4 ms; our features 82 x, 2.3 ms; packing 1 x, 0.4 ms
- ours-hbh, 2048 tokens, decode: our decoder 448 x, 18.9 ms; other 420 x, 4.5 ms; products (int8 GEMM) 27 x, 2.7 ms; copies and fills 50 x, 1.1 ms
- t2-hbh, 2048 tokens, encode: copies and fills 308 x, 10.9 ms; products (int8 GEMM) 65 x, 6.2 ms; our encoder 28 x, 3.0 ms; our features 82 x, 2.3 ms; packing 1 x, 0.4 ms
- t2-hbh, 2048 tokens, decode: our decoder 448 x, 18.3 ms; other 420 x, 4.4 ms; products (int8 GEMM) 27 x, 2.6 ms; copies and fills 50 x, 1.1 ms
- t2-rank256, 2048 tokens, encode: copies and fills 280 x, 8.8 ms; our encoder 28 x, 3.3 ms; products (int8 GEMM) 64 x, 2.4 ms; our features 81 x, 2.2 ms; packing 1 x, 0.5 ms
- t2-rank256, 2048 tokens, decode: our decoder 28 x, 3.4 ms; products (int8 GEMM) 54 x, 1.7 ms; copies and fills 49 x, 0.8 ms
- raw, 2048 tokens, decode: copies and fills 28 x, 0.7 ms
- splitzip, 2048 tokens, encode: other 56 x, 1.6 ms; copies and fills 140 x, 0.3 ms
- splitzip, 2048 tokens, decode: copies and fills 28 x, 0.7 ms; other 56 x, 0.7 ms
- adacode, 2048 tokens, encode: copies and fills 784 x, 3.9 ms; other 112 x, 0.9 ms; packing 56 x, 0.1 ms
- adacode, 2048 tokens, decode: copies and fills 56 x, 1.5 ms; other 56 x, 0.7 ms
- exactkv, 2048 tokens, encode: exactkv 56 x, 1.9 ms; copies and fills 840 x, 1.9 ms
- exactkv, 2048 tokens, decode: exactkv 56 x, 4.7 ms; copies and fills 96 x, 1.6 ms
- exans, 2048 tokens, encode: other 10 x, 41.2 ms; copies and fills 251 x, 3.6 ms
- exans, 2048 tokens, decode: copies and fills 506 x, 3.9 ms; other 3 x, 2.1 ms
- aquakv-2, 2048 tokens, encode: copies and fills 3042 x, 103.9 ms; products (int8 GEMM) 252 x, 39.1 ms
- aquakv-2, 2048 tokens, decode: copies and fills 1231 x, 24.8 ms; products (int8 GEMM) 110 x, 14.7 ms
- ours-hbh, 8192 tokens, encode: products (int8 GEMM) 110 x, 23.0 ms; our encoder 56 x, 12.8 ms; copies and fills 114 x, 12.1 ms; our features 164 x, 8.5 ms; packing 2 x, 1.5 ms
- ours-hbh, 8192 tokens, decode: our decoder 896 x, 40.7 ms; other 840 x, 14.4 ms; products (int8 GEMM) 54 x, 11.0 ms; copies and fills 128 x, 7.2 ms
- t2-hbh, 8192 tokens, encode: copies and fills 618 x, 29.3 ms; products (int8 GEMM) 130 x, 24.3 ms; our encoder 56 x, 12.8 ms; our features 164 x, 8.5 ms; packing 2 x, 1.5 ms
- t2-hbh, 8192 tokens, decode: our decoder 896 x, 39.5 ms; other 840 x, 14.2 ms; products (int8 GEMM) 54 x, 10.7 ms; copies and fills 128 x, 7.2 ms
- t2-rank256, 8192 tokens, encode: copies and fills 562 x, 21.0 ms; our encoder 56 x, 13.1 ms; our features 162 x, 8.4 ms; products (int8 GEMM) 128 x, 7.0 ms; packing 2 x, 1.9 ms
- t2-rank256, 8192 tokens, decode: our decoder 56 x, 12.6 ms; copies and fills 126 x, 5.9 ms; products (int8 GEMM) 108 x, 5.5 ms
- raw, 8192 tokens, decode: copies and fills 28 x, 2.8 ms
- splitzip, 8192 tokens, encode: other 56 x, 6.6 ms; copies and fills 140 x, 0.3 ms
- splitzip, 8192 tokens, decode: copies and fills 28 x, 2.8 ms; other 56 x, 2.6 ms
- adacode, 8192 tokens, encode: copies and fills 784 x, 15.7 ms; other 112 x, 4.6 ms; packing 56 x, 0.1 ms
- adacode, 8192 tokens, decode: copies and fills 56 x, 5.7 ms; other 56 x, 2.5 ms
- exactkv, 8192 tokens, encode: exactkv 56 x, 7.0 ms; copies and fills 840 x, 4.9 ms
- exactkv, 8192 tokens, decode: exactkv 56 x, 18.4 ms; copies and fills 96 x, 5.8 ms
- exans, 8192 tokens, encode: other 40 x, 165.0 ms; copies and fills 1001 x, 14.5 ms
- exans, 8192 tokens, decode: copies and fills 1937 x, 15.4 ms; other 12 x, 8.4 ms
- aquakv-2, 8192 tokens, encode: copies and fills 4074 x, 404.8 ms; products (int8 GEMM) 672 x, 157.6 ms
- aquakv-2, 8192 tokens, decode: copies and fills 1231 x, 98.2 ms; products (int8 GEMM) 110 x, 61.3 ms
- ours-hbh, 16384 tokens, encode: products (int8 GEMM) 220 x, 47.6 ms; our encoder 112 x, 26.8 ms; copies and fills 228 x, 24.4 ms; our features 328 x, 17.2 ms; packing 4 x, 4.3 ms
- ours-hbh, 16384 tokens, decode: our decoder 1792 x, 88.1 ms; other 1680 x, 29.0 ms; products (int8 GEMM) 108 x, 21.8 ms; copies and fills 226 x, 13.0 ms
- t2-hbh, 16384 tokens, encode: copies and fills 1236 x, 58.6 ms; products (int8 GEMM) 260 x, 49.7 ms; our encoder 112 x, 26.4 ms; our features 328 x, 17.2 ms; packing 4 x, 4.3 ms
- t2-hbh, 16384 tokens, decode: our decoder 1792 x, 87.7 ms; other 1680 x, 28.9 ms; products (int8 GEMM) 108 x, 21.8 ms; copies and fills 226 x, 13.0 ms
- t2-rank256, 16384 tokens, encode: copies and fills 1124 x, 41.9 ms; our encoder 112 x, 26.8 ms; our features 324 x, 16.9 ms; products (int8 GEMM) 256 x, 14.2 ms; packing 4 x, 4.2 ms
- t2-rank256, 16384 tokens, decode: our decoder 112 x, 28.8 ms; copies and fills 224 x, 11.8 ms; products (int8 GEMM) 216 x, 11.2 ms
- raw, 16384 tokens, decode: copies and fills 28 x, 5.6 ms
- splitzip, 16384 tokens, encode: other 56 x, 13.3 ms; copies and fills 140 x, 0.4 ms
- splitzip, 16384 tokens, decode: copies and fills 28 x, 5.5 ms; other 56 x, 5.1 ms
- adacode, 16384 tokens, encode: copies and fills 784 x, 29.9 ms; other 112 x, 9.1 ms; packing 56 x, 0.1 ms
- adacode, 16384 tokens, decode: copies and fills 56 x, 11.2 ms; other 56 x, 4.9 ms
- exactkv, 16384 tokens, encode: exactkv 56 x, 13.7 ms; copies and fills 840 x, 8.9 ms
- exactkv, 16384 tokens, decode: exactkv 56 x, 37.0 ms; copies and fills 96 x, 11.4 ms
- exans, 16384 tokens, encode: other 80 x, 330.3 ms; copies and fills 2000 x, 29.1 ms
- exans, 16384 tokens, decode: copies and fills 3845 x, 30.9 ms; other 24 x, 16.8 ms
- aquakv-2, 16384 tokens, encode: copies and fills 5450 x, 807.1 ms; products (int8 GEMM) 1232 x, 320.5 ms
- aquakv-2, 16384 tokens, decode: copies and fills 1231 x, 193.7 ms; products (int8 GEMM) 110 x, 126.5 ms
