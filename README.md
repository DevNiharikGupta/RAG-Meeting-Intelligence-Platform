# Meeting Intelligence Platform — Complete Guide

## What Does This Project Do?

You give it a meeting transcript (.txt file), and it can:
- Answer questions about the meeting ("What did Bob say about the deadline?")
- Pull out action items, decisions, discussion points automatically
- Do both at once through a single API call

It uses RAG (Retrieval-Augmented Generation) — a technique where we first search for relevant parts of the transcript, then ask an LLM to answer based on those parts.

---

## Project Files

```
AI/
├── .env              # config (model names, paths, chunk size)
├── requirements.txt  # python packages
├── ingest.py         # Task 1: read transcript → chunk → embed → store
├── query.py          # Task 2: question → search → LLM → answer
├── insights.py       # Task 3: transcript → LLM → structured JSON
├── server.py         # Task 4: FastAPI server wrapping all tasks
└── data/
    ├── transcripts/
    │   └── sprint_planning.txt   # sample meeting
    └── chroma_db/                # vector database (auto-created)
```

---

## Setup (One Time)

### 1. Install Ollama

macOS: `brew install ollama`
Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
Or download: https://ollama.com/download

### 2. Start Ollama and pull models

```bash
# start ollama (in a separate terminal)
ollama serve

# pull the embedding model (converts text to numbers, ~274MB)
ollama pull nomic-embed-text

# pull the language model (reads text and writes answers, ~2.3GB)
ollama pull phi3
```

### 3. Set up Python

```bash
cd AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### Run each task separately

```bash
source venv/bin/activate

python3 ingest.py      # stores transcript chunks in ChromaDB
python3 query.py       # asks 3 test questions
python3 insights.py    # extracts action items, decisions, etc.
```

### Run the API server

```bash
python3 server.py
# open http://localhost:8000/docs to test in browser
```

---

## What Each File Does

### ingest.py (Task 1 — Data Pipeline)

Reads a `.txt` transcript file, splits it into ~500 character chunks, converts each chunk into a number vector using `nomic-embed-text`, and stores everything in ChromaDB.

**Why chunk?** A full transcript is too long for search. Small chunks let us find the exact part that answers a question.

**Why overlap (50 chars)?** So sentences at chunk boundaries don't get cut off — the next chunk repeats the last 50 characters.

**Functions:**
- `read_transcript(filepath)` — reads a .txt file
- `split_into_chunks(text)` — breaks text into overlapping pieces
- `get_embedding_fn()` — creates the Ollama embedding model connection
- `store_in_chroma(chunks, source_name)` — embeds and stores in ChromaDB

### query.py (Task 2 — RAG Query System)

Takes a question, searches ChromaDB for the 5 most relevant chunks, formats them as context, sends context + question to phi3, returns the answer.

**How search works:** The question gets converted to a vector (same model as ingest). ChromaDB compares it against all stored chunk vectors and returns the closest matches by meaning — not keywords.

**Functions:**
- `get_chroma_db()` — opens the ChromaDB filled by ingest.py
- `find_relevant_chunks(question)` — semantic search, returns top 5 matches
- `build_context(results)` — formats chunks into labeled text for the LLM
- `ask_llm(question, context)` — sends to phi3, gets answer back
- `query(question)` — runs the full pipeline, returns answer + sources

### insights.py (Task 3 — Insight Extraction)

Sends the full transcript to phi3 with a prompt asking for structured JSON output containing action items, decisions, discussion points, and participant contributions.

**How it differs from Task 2:** Task 2 searches for specific answers. Task 3 summarizes the entire meeting into categories. Task 2 uses ChromaDB, Task 3 does not.

**Functions:**
- `extract_insights(text)` — sends text to LLM, parses JSON response
- `parse_json_response(raw_text)` — handles messy LLM output (tries 3 ways to extract valid JSON)

### server.py (Task 4 — FastAPI Orchestrator)

Wraps everything into HTTP endpoints. The key endpoint is `/orchestrate` which runs insight extraction + optional question answering in one call.

**Endpoints:**
- `POST /ingest` — give it a file path, it chunks and stores the transcript
- `POST /query` — give it a question, get a RAG answer
- `POST /insights` — give it a file path, get structured insights
- `POST /orchestrate` — give it a file path + optional question, get everything

---

## .env Config

```
OLLAMA_BASE_URL=http://localhost:11434    # where Ollama runs
OLLAMA_EMBED_MODEL=nomic-embed-text      # converts text to vectors
OLLAMA_LLM_MODEL=phi3                    # generates answers
CHROMA_DB_PATH=./data/chroma_db          # where vectors are saved
CHUNK_SIZE=500                           # characters per chunk
CHUNK_OVERLAP=50                         # overlap between chunks
TOP_K=5                                  # how many chunks to retrieve
```

---

## How the Pieces Connect

```
ingest.py writes to ──> ChromaDB (data/chroma_db/)
                            ^
query.py reads from ────────┘

insights.py reads ──> transcript .txt files (no ChromaDB needed)

server.py imports ──> ingest.py, query.py, insights.py
                      (calls their functions through HTTP endpoints)
```

---

## Testing with curl

```bash
# ingest a transcript
curl -X POST "http://localhost:8000/ingest?file_path=./data/transcripts/sprint_planning.txt"

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was decided about rate limiting?"}'

# extract insights
curl -X POST http://localhost:8000/insights \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/transcripts/sprint_planning.txt"}'

# orchestrate (insights + question together)
curl -X POST http://localhost:8000/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/transcripts/sprint_planning.txt", "question": "Who has action items?"}'
```

