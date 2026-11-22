# Where the codecs spend their time (encode / decode on one GPU, no vLLM)

wall: the call with the GPU finished; issue: the CPU's time to issue it; GPU: its kernels' summed time; launches: its kernels. Issue near wall and GPU far below it: bound by issuing launches.

| Codec | Tokens | Encode wall / issue / GPU ms | launches | Decode wall / issue / GPU ms | launches | Bits |
| --- | --- | --- | --- | --- | --- | --- |
| ours-hbh | 512 | 4.8 / 4.7 / 4.2 | 222 | 23.7 / 0.6 / 21.3 | 945 | 8.1 |
| t2-hbh | 512 | 9.5 / 9.4 / 8.5 | 484 | 21.7 / 1.1 / 21.3 | 945 | 8.1 |
| t2-rank256 | 512 | 9.1 / 9.0 / 8.1 | 454 | 3.8 / 2.9 / 3.4 | 131 | 9.3 |
| raw | 512 | 0.0 / 0.0 / 0.0 | 0 | 0.2 / 0.2 / 0.2 | 28 | 16.0 |
| splitzip | 512 | 3.1 / 3.0 / 0.7 | 196 | 1.1 / 1.1 / 0.3 | 84 | 12.0 |
| adacode | 512 | 14.3 / 14.3 / 2.2 | 952 | 2.5 / 2.5 / 0.4 | 112 | 11.4 |
| exactkv | 512 | 11.9 / 11.8 / 1.7 | 896 | 2.5 / 2.5 / 1.5 | 152 | 12.6 |
| exans | 512 | 13.4 / 13.2 / 12.7 | 83 | 3.7 / 3.7 / 3.1 | 155 | 11.0 |
| aquakv-2 | 512 | 39.7 / 39.7 / 36.7 | 3286 | 16.0 / 16.0 / 10.8 | 1397 | 2.2 |
| ours-hbh | 2048 | 14.6 / 14.2 / 13.6 | 222 | 26.7 / 1.1 / 26.5 | 945 | 8.1 |
| t2-hbh | 2048 | 24.0 / 23.6 / 22.7 | 484 | 26.8 / 1.1 / 26.4 | 945 | 8.1 |
| t2-rank256 | 2048 | 17.7 / 17.2 / 17.3 | 454 | 6.3 / 2.9 / 5.9 | 131 | 9.3 |
| raw | 2048 | 0.0 / 0.0 / 0.0 | 0 | 0.8 / 0.2 / 0.7 | 28 | 16.0 |
| splitzip | 2048 | 3.0 / 3.0 / 1.9 | 196 | 1.5 / 1.1 / 1.4 | 84 | 12.0 |
| adacode | 2048 | 14.9 / 14.9 / 4.8 | 952 | 2.5 / 2.4 / 2.2 | 112 | 11.4 |
| exactkv | 2048 | 13.1 / 13.1 / 3.8 | 896 | 7.0 / 2.5 / 6.3 | 152 | 12.6 |
| exans | 2048 | 47.1 / 46.6 / 44.8 | 261 | 7.8 / 7.3 / 6.0 | 509 | 10.9 |
| aquakv-2 | 2048 | 143.9 / 143.9 / 143.0 | 3294 | 41.1 / 15.6 / 39.2 | 1341 | 2.1 |
| ours-hbh | 8192 | 60.3 / 59.5 / 57.4 | 446 | 72.8 / 36.4 / 72.9 | 1918 | 8.0 |
| t2-hbh | 8192 | 78.5 / 77.7 / 76.7 | 970 | 73.2 / 36.5 / 72.2 | 1918 | 8.0 |
| t2-rank256 | 8192 | 52.7 / 51.8 / 51.5 | 910 | 25.9 / 14.7 / 24.1 | 290 | 9.2 |
| raw | 8192 | 0.0 / 0.0 / 0.0 | 0 | 2.8 / 0.2 / 2.8 | 28 | 16.0 |
| splitzip | 8192 | 7.8 / 7.7 / 7.0 | 196 | 5.5 / 1.2 / 5.4 | 84 | 12.0 |
| adacode | 8192 | 22.4 / 22.3 / 20.4 | 952 | 8.3 / 2.5 / 8.2 | 112 | 11.4 |
| exactkv | 8192 | 18.4 / 18.4 / 11.9 | 896 | 25.0 / 2.7 / 24.3 | 152 | 12.6 |
| exans | 8192 | 188.7 / 186.7 / 179.6 | 1041 | 31.0 / 28.4 / 23.8 | 1949 | 10.9 |
| aquakv-2 | 8192 | 565.2 / 565.1 / 567.3 | 4746 | 162.6 / 39.1 / 160.7 | 1341 | 2.1 |
| ours-hbh | 16384 | 122.4 / 121.0 / 119.9 | 892 | 153.7 / 136.0 / 151.9 | 3806 | 10.9 |
| t2-hbh | 16384 | 159.7 / 158.4 / 155.9 | 1940 | 154.1 / 136.1 / 151.6 | 3806 | 10.9 |
| t2-rank256 | 16384 | 106.0 / 104.9 / 104.0 | 1820 | 54.6 / 39.1 / 52.0 | 552 | 10.4 |
| raw | 16384 | 0.0 / 0.0 / 0.0 | 0 | 5.6 / 0.2 / 5.6 | 28 | 16.0 |
| splitzip | 16384 | 14.8 / 14.6 / 13.7 | 196 | 10.7 / 1.2 / 10.6 | 84 | 12.0 |
| adacode | 16384 | 40.9 / 40.7 / 39.2 | 952 | 16.2 / 2.6 / 16.1 | 112 | 11.4 |
| exactkv | 16384 | 26.2 / 26.1 / 22.5 | 896 | 49.1 / 2.9 / 48.4 | 152 | 12.6 |
| exans | 16384 | 377.6 / 373.9 / 359.4 | 2080 | 61.9 / 56.5 / 47.6 | 3869 | 10.9 |
| aquakv-2 | 16384 | 1123.2 / 1123.1 / 1127.0 | 6682 | 321.5 / 77.1 / 321.2 | 1341 | 2.1 |

## T and T2 against their calls as of job 30436 (wall ms)

Encode: each layer's sums once / twice (TKV_ENCODE_ONCE). Decode: graphs / launches one by one with the fused group product / one by one with the column copy, product and add (TKV_DECODE_GRAPHS, TKV_GROUP_PRODUCT).

| Codec | Tokens | Encode once | Encode twice | Decode (graphs) | Decode eager | Decode as before |
| --- | --- | --- | --- | --- | --- | --- |
| ours-hbh | 512 | 4.8 | 7.1 | 23.7 | 22.2 | 30.7 |
| t2-hbh | 512 | 9.5 | 11.9 | 21.7 | 26.4 | 40.1 |
| t2-rank256 | 512 | 9.1 | 10.5 | 3.8 | 3.8 | 3.7 |
| ours-hbh | 2048 | 14.6 | 22.6 | 26.7 | 27.2 | 40.2 |
| t2-hbh | 2048 | 24.0 | 32.0 | 26.8 | 27.2 | 40.2 |
| t2-rank256 | 2048 | 17.7 | 19.7 | 6.3 | 6.4 | 6.4 |
| ours-hbh | 8192 | 60.3 | 91.9 | 72.8 | 73.2 | 97.0 |
| t2-hbh | 8192 | 78.5 | 110.6 | 73.2 | 73.3 | 96.9 |
| t2-rank256 | 8192 | 52.7 | 58.8 | 25.9 | 25.9 | 25.9 |
| ours-hbh | 16384 | 122.4 | 186.3 | 153.7 | 153.6 | 201.2 |
| t2-hbh | 16384 | 159.7 | 223.8 | 154.1 | 154.1 | 201.5 |
| t2-rank256 | 16384 | 106.0 | 118.1 | 54.6 | 54.7 | 54.9 |

## By kernel kind (count, GPU ms)

- ours-hbh, 512 tokens, encode: products (int8 GEMM) 55 x, 1.7 ms; our encoder 28 x, 1.1 ms; our features 82 x, 0.7 ms; copies and fills 56 x, 0.6 ms; packing 1 x, 0.1 ms
- ours-hbh, 512 tokens, decode: our decoder 448 x, 16.9 ms; other 420 x, 3.3 ms; products (int8 GEMM) 27 x, 0.8 ms; copies and fills 50 x, 0.3 ms
- t2-hbh, 512 tokens, encode: copies and fills 308 x, 4.6 ms; products (int8 GEMM) 65 x, 2.0 ms; our encoder 28 x, 1.1 ms; our features 82 x, 0.7 ms; packing 1 x, 0.1 ms
- t2-hbh, 512 tokens, decode: our decoder 448 x, 16.9 ms; other 420 x, 3.3 ms; products (int8 GEMM) 27 x, 0.8 ms; copies and fills 50 x, 0.3 ms
- t2-rank256, 512 tokens, encode: copies and fills 280 x, 4.1 ms; our encoder 28 x, 1.8 ms; products (int8 GEMM) 64 x, 1.3 ms; our features 81 x, 0.8 ms; packing 1 x, 0.1 ms
- t2-rank256, 512 tokens, decode: our decoder 28 x, 2.2 ms; products (int8 GEMM) 54 x, 1.0 ms; copies and fills 49 x, 0.2 ms
- raw, 512 tokens, decode: copies and fills 28 x, 0.2 ms
- splitzip, 512 tokens, encode: other 56 x, 0.5 ms; copies and fills 140 x, 0.3 ms
- splitzip, 512 tokens, decode: other 56 x, 0.2 ms; copies and fills 28 x, 0.1 ms
- adacode, 512 tokens, encode: copies and fills 784 x, 1.9 ms; other 112 x, 0.3 ms; packing 56 x, 0.1 ms
- adacode, 512 tokens, decode: other 56 x, 0.3 ms; copies and fills 56 x, 0.2 ms
- exactkv, 512 tokens, encode: copies and fills 840 x, 1.2 ms; exactkv 56 x, 0.6 ms
- exactkv, 512 tokens, decode: exactkv 56 x, 1.3 ms; copies and fills 96 x, 0.2 ms
- exans, 512 tokens, encode: other 10 x, 10.9 ms; copies and fills 73 x, 1.8 ms
- exans, 512 tokens, decode: copies and fills 152 x, 1.9 ms; other 3 x, 1.2 ms
- aquakv-2, 512 tokens, encode: copies and fills 3058 x, 24.4 ms; products (int8 GEMM) 228 x, 12.3 ms
- aquakv-2, 512 tokens, decode: products (int8 GEMM) 110 x, 5.7 ms; copies and fills 1287 x, 5.1 ms
- ours-hbh, 2048 tokens, encode: products (int8 GEMM) 55 x, 5.6 ms; our encoder 28 x, 3.0 ms; copies and fills 56 x, 2.4 ms; our features 82 x, 2.3 ms; packing 1 x, 0.4 ms
- ours-hbh, 2048 tokens, decode: our decoder 448 x, 18.4 ms; other 420 x, 4.4 ms; products (int8 GEMM) 27 x, 2.6 ms; copies and fills 50 x, 1.1 ms
- t2-hbh, 2048 tokens, encode: copies and fills 308 x, 10.8 ms; products (int8 GEMM) 65 x, 6.2 ms; our encoder 28 x, 3.0 ms; our features 82 x, 2.3 ms; packing 1 x, 0.4 ms
- t2-hbh, 2048 tokens, decode: our decoder 448 x, 18.4 ms; other 420 x, 4.4 ms; products (int8 GEMM) 27 x, 2.6 ms; copies and fills 50 x, 1.1 ms
- t2-rank256, 2048 tokens, encode: copies and fills 280 x, 8.8 ms; our encoder 28 x, 3.3 ms; products (int8 GEMM) 64 x, 2.4 ms; our features 81 x, 2.2 ms; packing 1 x, 0.4 ms
- t2-rank256, 2048 tokens, decode: our decoder 28 x, 3.4 ms; products (int8 GEMM) 54 x, 1.7 ms; copies and fills 49 x, 0.8 ms
- raw, 2048 tokens, decode: copies and fills 28 x, 0.7 ms
- splitzip, 2048 tokens, encode: other 56 x, 1.6 ms; copies and fills 140 x, 0.3 ms
- splitzip, 2048 tokens, decode: other 56 x, 0.7 ms; copies and fills 28 x, 0.7 ms
- adacode, 2048 tokens, encode: copies and fills 784 x, 3.9 ms; other 112 x, 0.9 ms; packing 56 x, 0.1 ms
- adacode, 2048 tokens, decode: copies and fills 56 x, 1.5 ms; other 56 x, 0.7 ms
- exactkv, 2048 tokens, encode: exactkv 56 x, 1.9 ms; copies and fills 840 x, 1.9 ms
- exactkv, 2048 tokens, decode: exactkv 56 x, 4.7 ms; copies and fills 96 x, 1.6 ms
- exans, 2048 tokens, encode: other 10 x, 41.2 ms; copies and fills 251 x, 3.6 ms
- exans, 2048 tokens, decode: copies and fills 506 x, 3.9 ms; other 3 x, 2.1 ms
- aquakv-2, 2048 tokens, encode: copies and fills 3042 x, 103.9 ms; products (int8 GEMM) 252 x, 39.1 ms
- aquakv-2, 2048 tokens, decode: copies and fills 1231 x, 24.7 ms; products (int8 GEMM) 110 x, 14.4 ms
- ours-hbh, 8192 tokens, encode: products (int8 GEMM) 110 x, 22.7 ms; our encoder 56 x, 12.6 ms; copies and fills 114 x, 12.1 ms; our features 164 x, 8.4 ms; packing 2 x, 1.5 ms
- ours-hbh, 8192 tokens, decode: our decoder 896 x, 40.4 ms; other 840 x, 14.4 ms; products (int8 GEMM) 54 x, 10.9 ms; copies and fills 128 x, 7.2 ms
- t2-hbh, 8192 tokens, encode: copies and fills 618 x, 29.3 ms; products (int8 GEMM) 130 x, 24.4 ms; our encoder 56 x, 12.9 ms; our features 164 x, 8.5 ms; packing 2 x, 1.5 ms
- t2-hbh, 8192 tokens, decode: our decoder 896 x, 39.9 ms; other 840 x, 14.3 ms; products (int8 GEMM) 54 x, 10.8 ms; copies and fills 128 x, 7.2 ms
- t2-rank256, 8192 tokens, encode: copies and fills 562 x, 20.9 ms; our encoder 56 x, 13.2 ms; our features 162 x, 8.4 ms; products (int8 GEMM) 128 x, 7.0 ms; packing 2 x, 1.9 ms
- t2-rank256, 8192 tokens, decode: our decoder 56 x, 12.7 ms; copies and fills 126 x, 5.9 ms; products (int8 GEMM) 108 x, 5.5 ms
- raw, 8192 tokens, decode: copies and fills 28 x, 2.8 ms
- splitzip, 8192 tokens, encode: other 56 x, 6.7 ms; copies and fills 140 x, 0.3 ms
- splitzip, 8192 tokens, decode: copies and fills 28 x, 2.8 ms; other 56 x, 2.6 ms
- adacode, 8192 tokens, encode: copies and fills 784 x, 15.7 ms; other 112 x, 4.6 ms; packing 56 x, 0.1 ms
- adacode, 8192 tokens, decode: copies and fills 56 x, 5.7 ms; other 56 x, 2.5 ms
- exactkv, 8192 tokens, encode: exactkv 56 x, 7.0 ms; copies and fills 840 x, 4.9 ms
- exactkv, 8192 tokens, decode: exactkv 56 x, 18.5 ms; copies and fills 96 x, 5.8 ms
- exans, 8192 tokens, encode: other 40 x, 165.1 ms; copies and fills 1001 x, 14.5 ms
- exans, 8192 tokens, decode: copies and fills 1937 x, 15.4 ms; other 12 x, 8.4 ms
- aquakv-2, 8192 tokens, encode: copies and fills 4074 x, 406.9 ms; products (int8 GEMM) 672 x, 160.4 ms
- aquakv-2, 8192 tokens, decode: copies and fills 1231 x, 98.2 ms; products (int8 GEMM) 110 x, 62.5 ms
- ours-hbh, 16384 tokens, encode: products (int8 GEMM) 220 x, 47.4 ms; our encoder 112 x, 26.6 ms; copies and fills 228 x, 24.3 ms; our features 328 x, 17.2 ms; packing 4 x, 4.3 ms
- ours-hbh, 16384 tokens, decode: our decoder 1792 x, 88.1 ms; other 1680 x, 29.0 ms; products (int8 GEMM) 108 x, 21.8 ms; copies and fills 226 x, 13.0 ms
- t2-hbh, 16384 tokens, encode: copies and fills 1236 x, 58.6 ms; products (int8 GEMM) 260 x, 49.6 ms; our encoder 112 x, 26.3 ms; our features 328 x, 17.1 ms; packing 4 x, 4.3 ms
- t2-hbh, 16384 tokens, decode: our decoder 1792 x, 87.7 ms; other 1680 x, 28.9 ms; products (int8 GEMM) 108 x, 21.8 ms; copies and fills 226 x, 13.0 ms
- t2-rank256, 16384 tokens, encode: copies and fills 1124 x, 41.9 ms; our encoder 112 x, 26.8 ms; our features 324 x, 16.9 ms; products (int8 GEMM) 256 x, 14.2 ms; packing 4 x, 4.2 ms
- t2-rank256, 16384 tokens, decode: our decoder 112 x, 29.0 ms; copies and fills 224 x, 11.8 ms; products (int8 GEMM) 216 x, 11.2 ms
- raw, 16384 tokens, decode: copies and fills 28 x, 5.6 ms
- splitzip, 16384 tokens, encode: other 56 x, 13.3 ms; copies and fills 140 x, 0.4 ms
- splitzip, 16384 tokens, decode: copies and fills 28 x, 5.5 ms; other 56 x, 5.1 ms
- adacode, 16384 tokens, encode: copies and fills 784 x, 30.0 ms; other 112 x, 9.1 ms; packing 56 x, 0.1 ms
- adacode, 16384 tokens, decode: copies and fills 56 x, 11.2 ms; other 56 x, 4.9 ms
- exactkv, 16384 tokens, encode: exactkv 56 x, 13.6 ms; copies and fills 840 x, 8.9 ms
- exactkv, 16384 tokens, decode: exactkv 56 x, 36.9 ms; copies and fills 96 x, 11.4 ms
- exans, 16384 tokens, encode: other 80 x, 330.4 ms; copies and fills 2000 x, 29.1 ms
- exans, 16384 tokens, decode: copies and fills 3845 x, 30.9 ms; other 24 x, 16.8 ms
- aquakv-2, 16384 tokens, encode: copies and fills 5450 x, 806.8 ms; products (int8 GEMM) 1232 x, 320.2 ms
- aquakv-2, 16384 tokens, decode: copies and fills 1231 x, 193.8 ms; products (int8 GEMM) 110 x, 127.4 ms
