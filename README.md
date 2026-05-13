# QA Knowledge Assistant — Local RAG Prototype

> Built by [bala-kopparthi](https://github.com/bala-kopparthi) · May 2026

## Overview

QA Knowledge Assistant is a fully local Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions about your own QA documents — test cases, bug reports, and requirements specs — and get grounded, cited answers without sending any data to the cloud. Built as my first hands-on RAG prototype using Claude Code, it demonstrates the core AI engineering skills I am developing as I transition from manual QA into AI engineering: document ingestion, vector search, prompt engineering, and local LLM integration.

---

## Architecture

```
                        INDEXING (ingest.py — run once)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  documents/          Chunk          Embed                   │
│  (.txt/.pdf/.docx) ──────► 500-char ──────► nomic-embed   │
│                            chunks           -text (Ollama)  │
│                              │                   │          │
│                              └────────────────►  ChromaDB  │
│                                              (persisted to  │
│                                               chroma_db/)   │
└─────────────────────────────────────────────────────────────┘

                        QUERYING (query.py — every question)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  nomic-embed-text  ──► Embed question into vector           │
│       │                                                     │
│       ▼                                                     │
│  ChromaDB          ──► Cosine similarity search             │
│       │                ──► Top-4 most relevant chunks       │
│       │                                                     │
│       ▼                                                     │
│  Prompt Template   ──► "Answer using ONLY this context:     │
│                          {chunks} \n Question: {question}"  │
│       │                                                     │
│       ▼                                                     │
│  llama3.2 (Ollama) ──► Generate answer                     │
│       │                                                     │
│       ▼                                                     │
│  Answer + Source Chunks printed to terminal                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Why Chosen |
|---|---|---|
| LLM | llama3.2 3B via Ollama | Runs fully locally, no API key, fast enough for prototype Q&A |
| Embeddings | nomic-embed-text via Ollama | Purpose-built for retrieval tasks; 768-dim vectors; free and local |
| Vector Store | ChromaDB (persistent local) | Zero-config, file-based persistence, no server process needed |
| RAG Framework | LangChain 1.x (LCEL) | Industry standard; LCEL pipe syntax is composable and readable |
| PDF loader | pypdf | Lightweight, pure-Python, handles text-layer PDFs without dependencies |
| DOCX loader | python-docx | Native `.docx` support without LibreOffice or conversion tools |
| Language | Python 3.11+ | Matches LangChain and ChromaDB minimum requirements |

---

## Key Design Decisions

### Chunk size 500 / overlap 75
Five hundred characters is large enough to hold one complete idea (a test step, an acceptance criterion, a bug description paragraph) while being small enough for `nomic-embed-text` to encode that idea precisely. Shorter chunks lose context; longer chunks blur multiple topics into one vector, weakening retrieval precision. The 75-character overlap ensures that a sentence falling at a chunk boundary appears in both adjacent chunks — no context is lost at the seams.

### ChromaDB over Pinecone or Weaviate
A portfolio prototype should have zero infrastructure dependencies. ChromaDB writes directly to a local folder (`chroma_db/`) — no account, no server, no Docker. Pinecone and Weaviate are excellent production choices, but they require API keys and internet access, which conflicts with the offline-first goal of this project.

### llama3.2 (3B) over larger models
A 3.2B parameter model fits comfortably in RAM on a standard developer laptop and responds in seconds via Ollama. The RAG pattern compensates for the model's smaller knowledge base by supplying the relevant facts as context — the LLM's job is reading comprehension and formatting, not recall. A larger model (llama3:70b, Mistral, etc.) would improve answer quality but is not necessary to demonstrate the architecture.

### Separate embedding model from the LLM
`nomic-embed-text` is a bi-encoder fine-tuned specifically for semantic similarity tasks. Using a general-purpose LLM to produce embeddings would degrade retrieval quality because those models are not optimised to produce vectors where "close in space = similar in meaning" for retrieval. Separation of concerns: one model encodes meaning, the other generates language.

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** installed and running — [ollama.com](https://ollama.com)

### 1 — Pull the required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify both are available:

```bash
ollama list
```

### 2 — Clone the repository

```bash
git clone https://github.com/bala-kopparthi/RAG-prototype-claudecode.git
cd RAG-prototype-claudecode
```

### 3 — Create the virtual environment

```bash
python3 -m venv .venv
```

### 4 — Activate the virtual environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Your terminal prompt will show `(.venv)` when active.

### 5 — Install dependencies

```bash
pip install -r requirements.txt
```

### 6 — Add documents

The `documents/` folder already contains four sample QA artifacts (test cases, a bug report, a requirements spec). Add your own `.txt`, `.pdf`, or `.docx` files here.

### 7 — Run the ingest pipeline

```bash
python ingest.py
```

This loads, chunks, embeds, and stores all documents in `chroma_db/`. Run this once, and again whenever you add or change documents.

### 8 — Ask questions

```bash
# Question as a CLI argument
python query.py "What is the severity of BUG-117?"

# Or interactive mode
python query.py
```

---

## Usage Examples

### Example 1 — Bug report lookup

```
$ python query.py "What caused BUG-117 and what is the suggested fix?"

============================================================
  QUESTION: What caused BUG-117 and what is the suggested fix?
============================================================
[1/4] Loading embedding model ...
[2/4] Connecting to ChromaDB vector store ...
[3/4] Connecting to LLM (llama3.2 via Ollama) ...
[4/4] Retrieving relevant chunks and generating answer ...

============================================================
  ANSWER
============================================================

BUG-117 was caused by a case-sensitivity mismatch in the email lookup service.
The password reset endpoint passes the raw user-submitted email (preserving
uppercase letters) to the email dispatch queue without normalising it to
lowercase first. The dispatch service then performs a case-sensitive lookup
against the stored (always-lowercase) username, finds no match, and silently
drops the request — returning HTTP 200 to the UI despite never sending the email.

The suggested fix is to normalise the submitted email to lowercase before
passing it to the email dispatch service:
  submitted_email = request.email.strip().lower()

────────────────────────────────────────────────────────────
  SOURCES USED (4 chunk(s) retrieved)
────────────────────────────────────────────────────────────

  [1] bug_report_BUG-117.txt
      "ROOT CAUSE HYPOTHESIS: Suspected case-sensitivity mismatch in the email
       lookup service. The authentication service likely stores all usernames..."
```

---

### Example 2 — Requirements lookup

```
$ python query.py "What are the acceptance criteria for MFA backup codes?"

  ANSWER
============================================================

According to the requirements, when a user who has lost access to their
authenticator app clicks "Use a backup code" and enters one of their 8 backup
codes, access is granted. The used backup code is then invalidated and cannot
be reused. The user also sees a warning: "You have N backup codes remaining.
Consider re-registering your authenticator app."

  SOURCES USED: requirements_REQ-042.txt — AC-5 (Backup Code Usage)
```

---

### Example 3 — Test case lookup

```
$ python query.py "What should I check after clicking Place Order in the checkout test?"

  ANSWER
============================================================

After clicking "Place Order" (Step 8 of TC-CHECKOUT-001), you should check:
1. A loading spinner appears for no longer than 5 seconds.
2. The user is redirected to /order-confirmation with a unique Order ID.
3. The page displays "Thank you for your order!" and lists the purchased items.
4. In Mailtrap, a confirmation email arrives within 60 seconds with the subject
   "Your ShopFast Order #<ORDER_ID> is confirmed", containing correct items,
   total, and estimated delivery date.

  SOURCES USED: sample_test_case.txt — TC-CHECKOUT-001 Expected Results
```

---

## Project Structure

```
RAG-prototype-claudecode/
│
├── documents/                    Source documents for the RAG pipeline
│   ├── README.md                 Guide for adding your own documents
│   ├── sample_test_case.txt      E-commerce checkout — 3 test cases + bug IDs
│   ├── bug_report_BUG-117.txt    Password reset bug — severity High, root cause analysis
│   ├── requirements_REQ-042.txt  MFA feature spec — user story, ACs, NFRs, sign-off
│   └── test_case_TC-051.txt      Password reset — 9-step end-to-end test case
│
├── chroma_db/                    ← NOT in git (auto-created by ingest.py)
├── .venv/                        ← NOT in git (create locally with python3 -m venv .venv)
│
├── ingest.py                     Indexing pipeline: load → chunk → embed → store
├── query.py                      Query CLI: question → retrieve → generate → answer
├── requirements.txt              Pinned Python dependencies (150 packages)
├── .gitignore                    Excludes venv, chroma_db, .DS_Store, PDFs, secrets
└── README.md                     This file
```

---

## What I Learned

**The relationship between embedding models and LLMs**
I assumed at the start that the LLM could just "read" the documents. Building this taught me that there are actually two distinct models doing two different jobs: `nomic-embed-text` converts text into vectors (numbers that capture meaning) so that similarity search is possible, and `llama3.2` reads those retrieved chunks and writes a coherent answer. They cannot be swapped — using the LLM to produce embeddings would give poor retrieval results because it is not trained for that task.

**Why chunking strategy matters**
When I ingested the 470-page Jorgensen software testing textbook, the pipeline created 2,266 chunks. I then asked about the waterfall V-model lifecycle — a topic covered in Chapter 11. The retrieval kept returning chapter heading chunks rather than the detailed content, because that content lives in a figure (a diagram) which PyPDF cannot extract as text. The chunk that did come back was garbled figure-label text: `"How What How What How Preliminary design..."`. This was a concrete demonstration that chunking strategy and document pre-processing are as important as the model itself. The best LLM in the world cannot answer from context it never received.

**How ANN algorithms enable fast retrieval**
ChromaDB uses HNSW (Hierarchical Navigable Small World graphs) for approximate nearest-neighbour search. Instead of comparing the question vector against all 2,286 stored vectors one by one (which would be O(n)), HNSW builds a layered graph structure at ingest time that allows retrieval in O(log n). This is why the query step feels instant even with thousands of chunks — the expensive work happens at ingest time, not query time.

**The retrieve-then-generate pattern**
The most important architectural insight: RAG is not about making the LLM smarter — it is about giving it the right information at the right time. The prompt template I wrote explicitly says "Answer using ONLY the information provided in the context below." This constraint is what prevents hallucination. When the retrieved chunks contain the answer, the output is accurate and cited. When they do not (as in the diagram example), the system correctly says "I don't have enough information" rather than making something up. Grounding beats model size.

---

## Limitations & Future Improvements

| Limitation | Impact | Planned Fix |
|---|---|---|
| No re-ranker | Retrieved chunks are returned by vector similarity only — a chunk that is semantically close but factually less relevant may rank above a better one | Add a cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) as a second-pass filter |
| No hybrid search | Pure vector search can miss content when document vocabulary differs from the question | Combine ChromaDB (dense) with BM25 (sparse keyword) using `EnsembleRetriever` |
| PDF figures not extracted | Content inside diagrams, flowcharts, and figures is invisible to the pipeline | Integrate `unstructured` with OCR, or use a vision-capable model to describe figures |
| No evaluation framework | No way to measure retrieval precision or answer accuracy systematically | Implement RAGAs or a custom eval suite with labelled Q&A pairs (planned for v2) |
| No conversation memory | Each query is stateless — follow-up questions lose context | Add `ConversationBufferWindowMemory` for multi-turn Q&A sessions |
| Single language | Only English documents produce reliable results | Test with multilingual embedding models (e.g., `multilingual-e5`) |
| No incremental ingest | Re-running `ingest.py` re-embeds all documents, not just new ones | Add content-hash deduplication to skip unchanged files |

---

## Background

I am a manual QA engineer with experience in test case design, bug reporting, and requirements analysis, currently transitioning into AI engineering. QA skills transfer more directly to RAG evaluation than most people expect: writing a good test case (clear preconditions, precise expected results, edge case coverage) is structurally the same skill as writing a good RAG evaluation — you define the question, the correct answer, and the conditions under which the system should pass or fail. Building this prototype was my way of making that connection concrete: I used real QA artifacts as the document corpus, and I applied a tester's mindset to identify where the pipeline fails (the figure extraction limitation, the cross-chunk confusion bug) and why. I built it entirely with local, open-source tools using Claude Code as my AI engineering pair, with the goal of understanding every layer of the stack — not just running someone else's tutorial.

---

## License

MIT — see [LICENSE](LICENSE).
