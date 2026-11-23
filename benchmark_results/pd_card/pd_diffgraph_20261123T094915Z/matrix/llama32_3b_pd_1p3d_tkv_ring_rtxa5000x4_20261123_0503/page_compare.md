# Their page measures on our captured pages

Pages of 16 tokens x 28 layers (1,835,008 B raw; rows of 2,048 B); zlib level 1 throughout; deltas against an earlier turn of the same conversation, depth one. RoPE turning on. Their run (job 30547, 91 pages): zlib 18.8%, best independent (exponent and sign-fraction streams) 27.5%, graph 28.9% of raw saved: 1.9% beyond independent.

## mtrag_raw_docs_first_conversation

47 requests; 3,267 distinct complete pages (3,267 measured), 5,995.0 MB raw. Pages with a base: same position 2,616, same text 1,361.

| Representation | MB | Saving vs raw |
| --- | ---: | ---: |
| Raw | 5,995.0 | - |
| zlib 1 | 4,853.6 | 19.0% |
| byte shuffle + zlib | 4,362.6 | 27.2% |
| exponent / sign-fraction streams + zlib (their best) | 4,326.4 | 27.8% |
| best of the three per page | 4,326.4 | 27.8% |
| graph, their bases (position, text): 820 deltas on 668 bases | 4,220.6 | 29.6% (beyond independent: 2.4%) |
| graph, + keys turned by RoPE: 810 deltas on 646 bases | 4,153.4 | 30.7% (beyond independent: 4.0%) |

Per base rule (alone): beyond independent / pages with a candidate / per-page best saving median, p90, max:

- position: 0.6% / 2,616 / -3.3%, -2.8%, 86.1%
- text: 2.0% / 1,361 / 7.5%, 14.3%, 28.1%
- turned: 3.5% / 1,335 / 14.6%, 23.9%, 37.0%

By the kind of text a page starts in (graph with all rules): beyond independent (deltas):

- answer: 6.9% (559 of 1156 with a candidate)
- doc: 1.6% (205 of 1485 with a candidate)
- other: 19.6% (30 of 48 with a candidate)
- question: 7.5% (60 of 109 with a candidate)

High / low byte matches with the base (their 10 aligned pairs: K 76% / 8%, V 67% / 5%):

| Base | K L0 | K L1+ | V L0 | V L1+ |
| --- | --- | --- | --- | --- |
| aligned | 100.0% / 100.0% | 81.1% / 3.6% | 100.0% / 100.0% | 72.3% / 1.8% |
| position | 51.1% / 4.8% | 37.9% / 1.1% | 20.2% / 4.7% | 23.0% / 1.0% |
| text | 65.6% / 37.7% | 51.5% / 1.5% | 100.0% / 100.0% | 67.1% / 1.5% |
| turned | 99.7% / 71.2% | 75.7% / 2.6% | 100.0% / 100.0% | 66.9% / 1.5% |

Rows byte-identical to an earlier page's row of the same conversation (their: slot remapping added 570 matches, all layer-0 V; 150 from earlier turns):

| | rows | same slot | any slot | any slot, earlier turns |
| --- | ---: | ---: | ---: | ---: |
| L0 K | 52,272 | 1,800 of 41,856 | 2,821 | 2,821 |
| L0 V | 52,272 | 1,800 of 41,856 | 44,646 | 39,396 |
| L1+ K | 1,411,344 | 4,428 of 1,130,112 | 6,534 | 6,534 |
| L1+ V | 1,411,344 | 4,428 of 1,130,112 | 6,534 | 6,534 |

Each layer as an XOR against the previous one (byte-shuffled zlib per layer): 4,431.6 MB against 4,243.1 MB alone (4.4%; theirs +4.4%).

## mtrag_raw_history_first_conversation

47 requests; 2,278 distinct complete pages (2,278 measured), 4,180.1 MB raw. Pages with a base: same position 1,865, same text 629.

| Representation | MB | Saving vs raw |
| --- | ---: | ---: |
| Raw | 4,180.1 | - |
| zlib 1 | 3,386.0 | 19.0% |
| byte shuffle + zlib | 3,044.7 | 27.2% |
| exponent / sign-fraction streams + zlib (their best) | 3,018.9 | 27.8% |
| best of the three per page | 3,018.9 | 27.8% |
| graph, their bases (position, text): 466 deltas on 332 bases | 2,943.9 | 29.6% (beyond independent: 2.5%) |
| graph, + keys turned by RoPE: 466 deltas on 332 bases | 2,897.5 | 30.7% (beyond independent: 4.0%) |

Per base rule (alone): beyond independent / pages with a candidate / per-page best saving median, p90, max:

- position: 0.6% / 1,865 / -3.3%, -3.0%, 73.5%
- text: 1.8% / 629 / 10.7%, 18.5%, 23.2%
- turned: 3.4% / 629 / 19.5%, 31.6%, 38.7%

By the kind of text a page starts in (graph with all rules): beyond independent (deltas):

- answer: 6.3% (134 of 362 with a candidate)
- doc: 3.5% (334 of 1475 with a candidate)
- other: 32.9% (7 of 10 with a candidate)
- question: 0.5% (2 of 36 with a candidate)

High / low byte matches with the base (their 10 aligned pairs: K 76% / 8%, V 67% / 5%):

| Base | K L0 | K L1+ | V L0 | V L1+ |
| --- | --- | --- | --- | --- |
| aligned | - | - | - | - |
| position | 50.1% / 2.9% | 36.8% / 1.2% | 18.8% / 2.7% | 21.9% / 1.0% |
| text | 64.5% / 36.1% | 53.6% / 2.1% | 99.9% / 95.7% | 73.9% / 2.4% |
| turned | 99.7% / 70.0% | 80.8% / 4.1% | 99.9% / 95.7% | 73.9% / 2.4% |

Rows byte-identical to an earlier page's row of the same conversation (their: slot remapping added 570 matches, all layer-0 V; 150 from earlier turns):

| | rows | same slot | any slot | any slot, earlier turns |
| --- | ---: | ---: | ---: | ---: |
| L0 K | 36,448 | 632 of 29,840 | 903 | 903 |
| L0 V | 36,448 | 632 of 29,840 | 28,351 | 21,766 |
| L1+ K | 984,096 | 3,618 of 805,680 | 2,646 | 2,646 |
| L1+ V | 984,096 | 3,618 of 805,680 | 2,646 | 2,646 |

Each layer as an XOR against the previous one (byte-shuffled zlib per layer): 3,091.6 MB against 2,960.0 MB alone (4.4%; theirs +4.4%).

