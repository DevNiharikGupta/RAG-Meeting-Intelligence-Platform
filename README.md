# Meeting Intelligence Platform — Complete Guide

## What Does This Project Do?

You give it a meeting transcript (.txt file), and it can:
- Answer questions about the meeting ("What did Bob say about the deadline?")
- Pull out action items, decisions, discussion points automatically
- Do both at once through a single API call
- Store all data in **Databricks Delta tables** alongside local ChromaDB
- Use a **multi-agent system** (LangGraph) that automatically routes your request to the right agent

It uses RAG (Retrieval-Augmented Generation) — a technique where we first search for relevant parts of the transcript, then ask an LLM to answer based on those parts.

---

## Project Files

```
AI/
├── .env                  # config (model names, paths, Databricks creds)
├── requirements.txt      # python packages
├── ingest.py             # Task 1: read transcript → chunk → embed → store
├── query.py              # Task 2: question → search → LLM → answer
├── insights.py           # Task 3: transcript → LLM → structured JSON
├── server.py             # Task 4: FastAPI server wrapping all tasks
├── databricks_store.py   # Task 5 Part 1: Databricks Delta table integration
├── agents.py             # Task 5 Part 2: Multi-agent system (LangGraph)
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

### 4. Set up Databricks (optional)

This step is optional — everything works locally without Databricks. But if you want cloud storage too:

1. Create a free account at https://community.cloud.databricks.com
2. Go to SQL Warehouses → start the Serverless Starter Warehouse
3. Click "Connection details" tab → copy Server Hostname and HTTP Path
4. Go to Settings → Developer → Access tokens → Generate new token
5. Fill in your `.env`:

```
DATABRICKS_HOST=https://community.cloud.databricks.com
DATABRICKS_TOKEN=dapi...your-token...
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_CATALOG=
DATABRICKS_SCHEMA=default
```

6. Create the tables:

```bash
python3 databricks_store.py
```

If it prints "Databricks tables ready." you're good.

---

## How to Run

### Run each task separately

```bash
source venv/bin/activate

python3 ingest.py      # stores transcript chunks in ChromaDB + Databricks
python3 query.py       # asks 3 test questions
python3 insights.py    # extracts action items, decisions, etc.
python3 agents.py      # runs multi-agent pipeline (3 test cases)
```

### Run the API server

```bash
python3 server.py
# open http://localhost:8000/docs to test in browser
```

---

## What Each File Does

### ingest.py (Task 1 — Data Pipeline)

Reads a `.txt` transcript file, splits it into ~500 character chunks, converts each chunk into a number vector using `nomic-embed-text`, and stores everything in ChromaDB. If Databricks is configured, chunks also go to the `transcript_chunks` Delta table.

**Why chunk?** A full transcript is too long for search. Small chunks let us find the exact part that answers a question.

**Why overlap (50 chars)?** So sentences at chunk boundaries don't get cut off — the next chunk repeats the last 50 characters.

**Functions:**
- `read_transcript(filepath)` — reads a .txt file
- `split_into_chunks(text)` — breaks text into overlapping pieces
- `get_embedding_fn()` — creates the Ollama embedding model connection
- `store_in_chroma(chunks, source_name)` — embeds and stores in ChromaDB
- `ingest_file(filepath)` — full pipeline: read → chunk → ChromaDB + Databricks

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

Wraps everything into HTTP endpoints. The key endpoint is `/orchestrate` which runs insight extraction + optional question answering in one call. All endpoints automatically save to Databricks when configured.

**Endpoints:**
- `POST /ingest` — give it a file path, it chunks and stores the transcript
- `POST /query` — give it a question, get a RAG answer
- `POST /insights` — give it a file path, get structured insights
- `POST /orchestrate` — give it a file path + optional question, get everything
- `POST /agent` — smart endpoint: Router Agent decides what to do automatically
- `GET /databricks/chunks` — read stored chunks from Databricks
- `GET /databricks/insights` — read stored insights from Databricks

### databricks_store.py (Task 5 Part 1 — Databricks Integration)

Handles saving data to Databricks Delta tables alongside local ChromaDB. Creates two tables: `transcript_chunks` (stores chunked text) and `meeting_insights` (stores extracted JSON insights). If Databricks is not configured, all functions silently skip — nothing breaks.

**Functions:**
- `is_databricks_configured()` — checks if credentials are filled in
- `get_connection()` — opens SQL connection to Databricks warehouse
- `create_tables()` — creates Delta tables (safe to run multiple times)
- `save_chunks_to_databricks(chunks, source_name)` — writes chunks to Delta table
- `save_insights_to_databricks(insights, source_name)` — writes insights to Delta table
- `get_all_chunks(source_name)` — reads chunks back from Databricks
- `get_all_insights(source_name)` — reads insights back from Databricks

### agents.py (Task 5 Part 2 — Multi-Agent System)

Uses LangGraph to build a workflow of 4 specialized agents that collaborate:

```
User input → Router Agent → decides route
                |
                ├── "question"  → Retrieval Agent → Summary Agent → response
                ├── "insights"  → Insight Agent   → Summary Agent → response
                └── "both"      → Retrieval + Insight → Summary Agent → response
```

**Agents:**
- `router_agent` — classifies the request as "question", "insights", or "both"
- `retrieval_agent` — searches ChromaDB and answers using RAG (reuses query.py logic)
- `insight_agent` — extracts structured insights from transcript (reuses insights.py logic)
- `summary_agent` — combines outputs from other agents into one final response

**Functions:**
- `build_agent_graph()` — creates and compiles the LangGraph state graph
- `run_agents(user_input, file_path)` — runs the full multi-agent pipeline

---

## .env Config

```
# Ollama (required)
OLLAMA_BASE_URL=http://localhost:11434    # where Ollama runs
OLLAMA_EMBED_MODEL=nomic-embed-text      # converts text to vectors
OLLAMA_LLM_MODEL=phi3                    # generates answers
CHROMA_DB_PATH=./data/chroma_db          # where vectors are saved
CHUNK_SIZE=500                           # characters per chunk
CHUNK_OVERLAP=50                         # overlap between chunks
TOP_K=5                                  # how many chunks to retrieve

# Databricks (optional — leave empty to use only local storage)
DATABRICKS_HOST=https://community.cloud.databricks.com
DATABRICKS_TOKEN=your-token-here
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_CATALOG=                      # leave empty for Community Edition
DATABRICKS_SCHEMA=default
```

---

## How the Pieces Connect

```
ingest.py writes to ──> ChromaDB (data/chroma_db/)
            |               ^
            |   query.py reads from ┘
            |
            └───────────> Databricks (transcript_chunks table)

insights.py reads ──> transcript .txt files
            |
            └───────────> Databricks (meeting_insights table)

agents.py  ──> Router Agent decides route
            ├── calls query.py (Retrieval Agent)
            ├── calls insights.py (Insight Agent)
            └── Summary Agent combines results

server.py imports ──> ingest.py, query.py, insights.py,
                      databricks_store.py, agents.py
                      (calls their functions through HTTP endpoints)
```

---

## Testing with curl

### Core Endpoints

```bash
# ingest a transcript (stores in ChromaDB + Databricks)
curl -X POST "http://localhost:8000/ingest?file_path=./data/transcripts/sprint_planning.txt"

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was decided about rate limiting?"}'

# extract insights (saves to Databricks too)
curl -X POST http://localhost:8000/insights \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/transcripts/sprint_planning.txt"}'

# orchestrate (insights + question together)
curl -X POST http://localhost:8000/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/transcripts/sprint_planning.txt", "question": "Who has action items?"}'
```

### Agent Endpoint (Multi-Agent)

```bash
# question only — Router sends to Retrieval Agent
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What was decided about rate limiting?"}'

# insights only — Router sends to Insight Agent
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"user_input": "", "file_path": "./data/transcripts/sprint_planning.txt"}'

# both — Router sends to Retrieval + Insight Agents
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Who has action items?", "file_path": "./data/transcripts/sprint_planning.txt"}'
```

### Databricks Endpoints

```bash
# get all chunks stored in Databricks
curl http://localhost:8000/databricks/chunks

# get chunks for a specific file
curl "http://localhost:8000/databricks/chunks?source_file=sprint_planning.txt"

# get all insights stored in Databricks
curl http://localhost:8000/databricks/insights

# get insights for a specific file
curl "http://localhost:8000/databricks/insights?source_file=sprint_planning.txt"
```
