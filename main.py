import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError
from langchain_core.messages import HumanMessage, ToolMessage


# Agents decide their own control flow, so nothing bounds the number of
# tool rounds except this. Without it a model that keeps requesting tools
# spins until the quota runs out.
MAX_TOOL_ROUNDS = 5


def file_loader():
    p = r"data"

    articles = []
    with os.scandir(p) as entries:
        for e in entries:
            if e.is_file() and e.name.endswith(".txt"):
                with open(e.path, "r", encoding="utf-8") as f:
                    text = f.read()

                articles.append({
                    "source": e.name,
                    "text": text
                })

    # Fail here, where we know what went wrong. Otherwise this surfaces as an
    # array-shape error deep inside FAISS.
    if not articles:
        raise FileNotFoundError(
            f"No .txt files found in {p!r}. See the README for corpus setup."
        )
    return articles


def recursive_chunking(text: str, max_chunk_size: int = 1000):
    # Base case: if text is small enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []

    # Ordered widest to narrowest. Splitting on paragraph breaks first keeps
    # sentences intact, so this needs no overlap to protect boundaries.
    separators = ["\n\n", "\n", ". ", " "]

    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""

            for part in parts:
                # Check if adding this part would exceed the limit
                test_chunk = current_chunk + separator + part if current_chunk else part

                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part

            # Add the final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Recursively process any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)

            return [chunk for chunk in final_chunks if chunk]

    # Fallback: split by character limit if no separators work.
    # This is the one path that can cut mid-sentence.
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


# Cached: loading the model is ~90MB from disk, and an agent may call embed
# an unpredictable number of times per question.
_model = None


def embed(list_of_sentences: list):
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model.encode(list_of_sentences)


def build_index(embeddings: np.ndarray):
    # "Flat" compares against every stored vector: no approximation, exact
    # results. Fine at this scale; approximate indexes earn their keep in the
    # millions. "L2" is straight-line distance.
    # Query and chunks must share an embedding model. Two different models
    # produce two unrelated spaces, and distances across them are meaningless.
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# Closure, not globals: the model only supplies `query`, so `index` and
# `all_chunks` have to reach the tool through the enclosing scope.
def make_tool(index, all_chunks):
    @tool
    def search_corpus(query: str) -> str:
        # The docstring IS the interface. The model decides whether to call
        # this based on it alone.
        """This searcher is intended for FIFA World Cup 2026 questions."""
        query_vector = embed([query])
        _, indices = index.search(query_vector, 5)

        # FAISS returns row numbers. all_chunks is the lookup table that turns
        # them back into readable text; position alignment between the two is
        # an invariant nothing enforces.
        passages = []
        for row in indices[0]:
            match = all_chunks[row]
            passages.append(f"{match['source']}: {match['text']}")
        return "\n\n".join(passages)
    return search_corpus


def build_agent(index, all_chunks):
    search_tool = make_tool(index, all_chunks)
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

    # bind_tools tells the model the tool exists. It cannot run it: that stays
    # our job, which is why the tool itself is returned too.
    model_with_tools = model.bind_tools([search_tool])
    return model_with_tools, search_tool


def ask(question, model, tool, max_rounds=MAX_TOOL_ROUNDS):
    messages = [HumanMessage(content=question)]

    # The model controls the flow: it may search zero times, once, or several
    # times before answering. Bounded so a stuck model can't loop forever.
    for _ in range(max_rounds):
        response = model.invoke(messages)

        if not response.tool_calls:
            content = response.content
            # Newer models return a list of typed blocks, not a plain string.
            if isinstance(content, list):
                return "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            return content

        # The model's own reply has to go back too, or the results below
        # answer a request it has no record of making.
        messages.append(response)

        for call in response.tool_calls:
            passage = tool.invoke(call["args"])
            # tool_call_id pairs this result with the request it answers.
            messages.append(ToolMessage(passage, tool_call_id=call["id"]))

    raise RuntimeError(
        f"Agent did not produce an answer within {max_rounds} tool rounds."
    )


def build_chunks(articles):
    all_chunks = []
    for article in articles:
        chunks = recursive_chunking(article['text'])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": article['source'],
                "text": chunk,
                "index": i
            })
    return all_chunks


def main():
    load_dotenv()

    # Fail before the ~15s of indexing rather than after it.
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Create a .env file with your key."
        )

    articles = file_loader()
    all_chunks = build_chunks(articles)

    # Indexing happens once at startup; asking happens per question.
    # That split is why this is a REPL and not a one-shot script.
    embeddings = embed([chunk['text'] for chunk in all_chunks])
    index = build_index(embeddings)

    model_with_tools, search_tool = build_agent(index, all_chunks)

    print(f"Indexed {len(all_chunks)} chunks from {len(articles)} documents.")
    print("Ask a question, or press Ctrl-D to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue

        # A transient API error or a hit iteration cap should return you to the
        # prompt, not end the session.
        try:
            print(ask(question, model_with_tools, search_tool), "\n")
        except GoogleAPIError as e:
            print(f"Model error: {e}\n")
        except RuntimeError as e:
            print(f"{e}\n")


if __name__ == "__main__":
    main()
