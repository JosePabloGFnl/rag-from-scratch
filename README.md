# rag-from-scratch

A retrieval-augmented generation pipeline written without a RAG framework, then wrapped as an agent tool.

The point of this repo is the _decisions_, not the code volume. Chunking, embedding, indexing and retrieval are implemented directly so that every parameter is a choice rather than a library default.

## What it does

Answers questions about the 2026 FIFA World Cup using five Wikipedia articles as its corpus. The underlying model's training data predates the tournament, so every correct answer has to come from retrieval.

```
$ python main.py
Indexed 745 chunks from 5 documents.
Ask a question, or press Ctrl-D to quit.

> who won the 2026 world cup?
Spain won the 2026 FIFA World Cup, defeating defending champions Argentina 1–0
after extra time in the final on July 19, 2026, at MetLife Stadium in East
Rutherford, New Jersey. This was Spain's second World Cup title.
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

Indexing takes roughly fifteen seconds and happens once at startup; questions after that are near-instant. That split is why this is a REPL rather than a one-shot script.

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

### A bounded agent loop

`MAX_TOOL_ROUNDS` caps how many search-and-reconsider cycles the agent gets before the attempt is abandoned. Nothing else bounds it: the model decides when to stop calling tools, and a model that never stops would run until the API quota did. See failure 4 for a question that actually hits the cap.

## Four failures worth reading about

These are the reason the repo exists. All four came from ordinary questions, not contrived tests.

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

Asked how Sergio Ramos performed in the tournament, the system answered that he was not in Spain's 26-man squad — correct, and supported by `squads.txt` — and then added that he had retired from international football in February 2023.

That second claim almost certainly is not in the corpus. It predates the tournament and has no reason to appear in a 2026 World Cup article. The model retrieved correctly, then extended the answer from training data, with nothing in the output marking where one ended and the other began.

This is harder to catch than failure 2. A skipped tool call shows up in logs. This looks like a perfect answer.

It is also the clearest argument for citations: if every claim had to point at a retrieved passage, the retirement line would have had nothing to point to.

### 4. The same question fails and succeeds on different runs

Asked _"how many goals did Messi score during the 2026 World Cup?"_, the agent exhausted all five tool rounds without producing an answer. Asked again, unchanged, it answered correctly: 8 goals, second behind Mbappé's 10.

The answer was in the corpus the whole time, in the Golden Boot table in `world-cup-2026.txt`:

```
Mbappé 10 goals, 4 assists, 769 minutes played | Messi 8 goals, 4 assists,
853 minutes played | Bellingham 7 goals, 1 assist, 698 minutes played
```

Two things make this hard to retrieve. A per-player goal tally is a _number_, and numbers are where embeddings are weakest. And the query matches hundreds of chunks about goals and scoring, while the answer sits in one stats row whose neighbouring text is mostly other players' names and numbers. Retrieval has to land on that specific row among many plausible ones. Sometimes it does.

There is a second lesson buried in this one. After failure 3, the correct answer _looked_ suspicious — a tidy top-scorer ranking is exactly the shape a model's parametric knowledge produces. It took a `grep` over the corpus to confirm it was real.

Once a system can blend retrieved and unretrieved content invisibly, **"it sounded right" and "it sounded suspicious" are both useless signals.** That is the practical case for citations, restated from the other direction.

## Known limitations

- The corpus is unprocessed Wikipedia text, navigation junk included. Deliberate — clean input hides the extraction problems real corpora have.
- Table structure is flattened into plain text during chunking. The Golden Boot row survived as a readable line, but that was luck rather than design.
- No evaluation set. Retrieval quality is judged by eye, which is exactly the thing a production system cannot do.
- No citations, so grounded and ungrounded claims are indistinguishable in the output (failures 3 and 4).
- `k` and the chunk size are constants, chosen by inspection rather than tuned against anything.
- The index is rebuilt on every run instead of being persisted.

## Stack

Python, `sentence-transformers`, `faiss-cpu`, `numpy`, `langchain-core`, `langchain-google-genai`, `python-dotenv`.
