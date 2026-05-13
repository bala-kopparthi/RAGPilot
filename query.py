"""
query.py — Interactive question-answering CLI for the RAG system.

What this script does (the RAG "retrieval + generation" phase):
  1. Takes your question (typed at the prompt or passed as a CLI argument)
  2. Embeds the question into a vector using the same model used in ingest.py
  3. Searches ChromaDB for the chunks whose vectors are closest to the question
  4. Builds a prompt: [retrieved chunks] + [your question]
  5. Sends that prompt to llama3.2 running locally via Ollama
  6. Prints the answer and shows which source chunks were used

Interview tip: Step 3 is "retrieval" and step 5 is "generation" — that's the
"R" and "G" in RAG. The LLM never sees your full document library, only the
most relevant chunks for each specific question.

This uses LangChain's LCEL (LangChain Expression Language) — the modern way
to compose chains using the pipe operator (|). Think of it like Unix pipes:
  retriever | format_docs | prompt | llm | parse_output
"""

import sys
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DIR  = "./chroma_db"       # must match the path used in ingest.py
COLLECTION  = "rag_capstone"      # must match the collection name in ingest.py
EMBED_MODEL = "nomic-embed-text"  # same embedding model used during ingest
LLM_MODEL   = "llama3.2"         # the local LLM that generates the answer

# How many chunks to retrieve per question.
# 4 is a good default: enough context, few enough to stay under token limits.
TOP_K = 4


# ── Prompt template ───────────────────────────────────────────────────────────

# This is the exact text sent to the LLM. {context} is replaced with the
# retrieved chunks joined together; {question} is replaced with your question.
# "Only use the provided context" is the core RAG constraint — it stops the
# LLM from hallucinating answers from its training data.
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY
the information provided in the context below. If the answer is not in the
context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Accept question as a CLI argument or ask interactively
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("\nRAG Query System — ask a question about your documents.")
        print("(Run 'python query.py' again to ask another question)\n")
        question = input("Your question: ").strip()

    if not question:
        print("No question entered. Exiting.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  QUESTION: {question}")
    print(f"{'='*60}")

    # Step 1: Connect to the same embedding model used at ingest time.
    # The question must be embedded with the SAME model as the documents —
    # mixing models would make vector comparisons meaningless.
    print("\n[1/4] Loading embedding model ...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Step 2: Open the existing ChromaDB (read-only — do NOT recreate it).
    print("[2/4] Connecting to ChromaDB vector store ...")
    if not Path(CHROMA_DIR).exists():
        print(f"\n  [ERROR] No ChromaDB found at '{CHROMA_DIR}'.")
        print("  Run 'python ingest.py' first to build the vector store.")
        sys.exit(1)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )

    # Step 3: Set up the retriever.
    # "similarity" search finds the TOP_K chunks whose embeddings are closest
    # (by cosine distance) to the question embedding.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Step 4: Set up the LLM — llama3.2 running locally via Ollama.
    # temperature=0 means deterministic output (no randomness). Good for
    # factual Q&A where you want consistent, grounded answers.
    print("[3/4] Connecting to LLM (llama3.2 via Ollama) ...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Step 5: Build the LCEL chain using the pipe (|) operator.
    #
    # RunnableParallel runs two branches simultaneously:
    #   "context" branch: question → retriever → format_docs (joins chunks)
    #   "question" branch: question passed through unchanged
    # Both results are merged into one dict: {"context": "...", "question": "..."}
    # That dict is fed into the prompt template, then the LLM, then the parser.
    #
    # Diagram:  question
    #            ├─→ retriever → format_docs ─→ {context}  ─┐
    #            └─→ passthrough              ─→ {question} ─┤
    #                                                        ↓
    #                                                    prompt → llm → str
    retrieve_and_format = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )

    chain = retrieve_and_format | prompt | llm | StrOutputParser()

    # Step 6: Run retrieval separately first so we can display the source chunks.
    print("[4/4] Retrieving relevant chunks and generating answer ...\n")
    source_docs = retriever.invoke(question)

    # Invoke the full chain for the final answer
    answer = chain.invoke(question)

    # ── Print the answer ──────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("  ANSWER")
    print(f"{'='*60}")
    print(f"\n{answer}\n")

    # ── Print source chunks (transparency / explainability) ───────────────────
    # Showing which chunks were used is called "grounding" or "citations".
    # It lets users verify the answer and helps debug retrieval problems.
    if source_docs:
        print(f"{'─'*60}")
        print(f"  SOURCES USED ({len(source_docs)} chunk(s) retrieved)")
        print(f"{'─'*60}")
        for i, doc in enumerate(source_docs, 1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"\n  [{i}] {Path(source).name}")
            print(f"      \"{preview}...\"")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
