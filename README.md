# rag-from-scratch

A retrieval-augmented generation pipeline written without a RAG framework, then wrapped as an agent tool.

The point of this repo is the _decisions_, not the code volume. Chunking, embedding, indexing and retrieval are implemented directly so that every parameter is a choice rather than a library default.

## What it does

Answers questions about the 2026 FIFA World Cup using five Wikipedia articles as its corpus. The underlying model's training data predates the tournament, so every correct answer has to come from retrieval.

```
$ python main.py
Spain won the 2026 FIFA World Cup. They defeated the defending champions
Argentina 1–0 (after extra time) in the final on July 19, 2026, at MetLife
Stadium in East Rutherford, New Jersey.
```

## Pipeline

| Stage    | Implementation                                                        |
| -------- | --------------------------------------------------------------------- |
| Load     | Plain text files from `data/`, tagged with their source filename      |
| Chunk    | Recursive splitting, max 1000 characters                              |
| Embed    | `all-MiniLM-L6-v2` via `sentence-transformers`, 384 dimensions        |
| Index    | FAISS `IndexFlatL2`, exact search                                     |
| Retrieve | Embed the query, take top-k, map row numbers back to chunks           |
| Generate | Retriever exposed as a LangChain tool; Gemini decides when to call it |

Corpus is roughly 565,000 characters, producing 745 chunks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with a Google AI Studio key:

```
GOOGLE_API_KEY=your-key-here
```

Then:

```bash
python main.py
```

## Design notes

### Recursive chunking, not fixed-size with overlap

Recursive splitting tries separators in priority order — paragraph breaks, line breaks, sentence ends, then spaces — so cuts land at natural boundaries instead of mid-sentence.

The trade-off is real. Fixed-size chunking with overlap _repairs_ bad boundaries by repeating text between chunks; recursive splitting _avoids creating_ them. Neither does the other's job:

- No overlap means connective context between adjacent chunks is lost.
- The character-slice fallback still cuts blindly when no separator exists.
- Chunk sizes vary, so the per-query token budget is unpredictable.

Production systems often run both together.

### Exact search

`IndexFlatL2` compares the query against every stored vector. No approximation. At 745 vectors this is instant, and approximate indexes only earn their complexity in the millions.

### One embedding model for both sides

The query and the chunks must be embedded by the same model. Two models produce two unrelated vector spaces, and distances measured across them are meaningless.

### A closure, not globals

The agent supplies only `query` when it calls the tool. `index` and `all_chunks` reach the tool through the enclosing scope of `make_tool` rather than through module-level globals — explicit, no load-order dependency, and two corpora could coexist.

### Cached embedding model

Loading `all-MiniLM-L6-v2` reads ~90MB from disk. A fixed pipeline calls `embed` a known number of times; an agent calls it an unpredictable number of times, so per-call loading goes from wasteful to slow.

## Two failures worth reading about

These are the reason the repo exists.

### 1. Semantic similarity is not factual specificity

Asked _"who won the world cup"_, the top-ranked chunk was about **Argentina winning in Qatar 2022** — a passage from the 2026 corpus, dense with final-related vocabulary, topically perfect and factually wrong. The correct answer ranked second.

Embeddings match on meaning, and they are weakest exactly where precision matters most: dates, version numbers, names, IDs.

Mitigations, none of which are implemented here:

- A reranker as a second pass over the top-k
- Metadata filtering (by year) before the vector search runs
- Hybrid keyword + vector search
- A higher `k`, so the correct answer appears somewhere in the set and generation resolves it

The last one is what this repo does. At `k=5`, three of five hits corroborated the correct answer across two different source files, and the model resolved the contradiction. That is the real argument for `k > 1`: not more chances to get lucky, but redundancy that lets generation settle what ranking could not.

### 2. The tool description is the interface

On one run the model **never called the tool at all**. It answered from training data and reported that the 2026 World Cup was upcoming.

Nothing was wrong with the retrieval code. The docstring was weak, and the question was ambiguous. A model reaches for a tool only when it believes the tool knows something it does not.

This also crashed the first version of the agent loop, which assumed a tool call would always happen. That crash is the practical difference between a pipeline and an agent: **you do not control whether the tool gets called.**

### 3. Grounded and ungrounded claims blend invisibly

Asked how Sergio Ramos performed in the tournament, the system answered that he
was not in Spain's 26-man squad — correct, and supported by `squads.txt` — and
then added that he had retired from international football in February 2023.

That second claim almost certainly is not in the corpus. It predates the
tournament and has no reason to appear in a 2026 World Cup article. The model
retrieved correctly, then extended the answer from training data, with nothing
in the output marking where one ended and the other began.

This is harder to catch than failure 2. A skipped tool call shows up in logs.
This looks like a perfect answer.

It is also the clearest argument for citations: if every claim had to point at a
retrieved passage, the retirement line would have had nothing to point to.

## Known limitations

- The corpus is unprocessed Wikipedia text, navigation junk included. Deliberate — clean input hides the extraction problems real corpora have.
- No evaluation set. Retrieval quality is judged by eye, which is exactly the thing a production system cannot do.
- The agent loop has no iteration cap.
- `k` and the chunk size are hardcoded rather than tuned against anything.
- The index is rebuilt on every run instead of being persisted.

## Stack

Python, `sentence-transformers`, `faiss-cpu`, `numpy`, `langchain-core`, `langchain-google-genai`, `python-dotenv`.
