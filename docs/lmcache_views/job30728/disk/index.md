# The conversations, one by one: disk mode

Each page follows one MTRAG conversation turn by turn: the prompt's pieces, the 256-token chunks LMCache keeps them in, each chunk's 16 vLLM blocks, and what happened to every chunk on P, in LMCache and on the conversation's D engine. Built by scripts/lmc_lifecycle.py from the run's trace. The prompts' text is rebuilt from the MTRAG file the run read, by the driver's own code (scripts/pd_mtrag.py build()), and matches the run's record piece for piece; token counts and places are the run's. D's answers were not kept by this run's driver, only their token ids. Lines over 100 characters are wrapped here for reading.

| conversation | domain | first question | turns | last prompt | D engine | chunks stored | pulled | held by D | page |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 0 | mt-rag-clapnq-elser-512-100-20240503 | where do the arizona cardinals play this week | 8 | 9,097 | D1 | 35 | 32 | 123 | [conversation_0.md](conversation_0.md) |
| 1 | mt-rag-govt-elser-512-100-20240611 | "What are the sheltered rooms designated for use? | 8 | 9,765 | D1 | 38 | 35 | 139 | [conversation_1.md](conversation_1.md) |
| 2 | mt-rag-govt-elser-512-100-20240611 | What is ground water contamination? | 7 | 7,503 | D2 | 29 | 25 | 88 | [conversation_2.md](conversation_2.md) |
| 3 | mt-rag-fiqa-beir-elser-512-100-20240501 | What's the difference between Market Cap and NAV? | 7 | 3,102 | D3 | 12 | 10 | 35 | [conversation_3.md](conversation_3.md) |
