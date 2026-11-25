# The conversations, one by one: memory mode

Each page follows one MTRAG conversation turn by turn: the prompt's pieces, the 256-token chunks LMCache keeps them in, each chunk's 16 vLLM blocks, and what happened to every chunk on P, in LMCache and on the conversation's D engine. Built by scripts/lmc_lifecycle.py from the run's trace.

| conversation | domain | turns | last prompt | D engine | chunks stored | pulled | held by D | page |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 0 | mt-rag-clapnq-elser-512-100-20240503 | 8 | 9,097 | D1 | 35 | 32 | 123 | [conversation_0.md](conversation_0.md) |
| 1 | mt-rag-govt-elser-512-100-20240611 | 8 | 9,765 | D1 | 38 | 35 | 139 | [conversation_1.md](conversation_1.md) |
| 2 | mt-rag-govt-elser-512-100-20240611 | 7 | 7,503 | D2 | 29 | 25 | 88 | [conversation_2.md](conversation_2.md) |
| 3 | mt-rag-fiqa-beir-elser-512-100-20240501 | 7 | 3,102 | D3 | 12 | 10 | 35 | [conversation_3.md](conversation_3.md) |
