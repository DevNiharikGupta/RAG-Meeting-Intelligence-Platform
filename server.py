import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from ingest import read_transcript, split_into_chunks, store_in_chroma, ingest_file
from query import query
from insights import extract_insights
from databricks_store import (
    is_databricks_configured,
    save_insights_to_databricks,
    get_all_chunks,
    get_all_insights,
    create_tables,
)

app = FastAPI(title="Meeting Intelligence API", version="2.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class InsightRequest(BaseModel):
    file_path: str

class OrchestrateRequest(BaseModel):
    file_path: str
    question: Optional[str] = None


@app.post("/ingest")
def ingest_transcript(file_path: str):
    """Chunk a transcript file, embed it, store in ChromaDB + Databricks."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    count = ingest_file(file_path)

    return {
        "message": "Transcript ingested",
        "chunks_created": count,
        "databricks_enabled": is_databricks_configured(),
    }


@app.post("/query")
def query_meetings(req: QueryRequest):
    """Ask a question, get a RAG-powered answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = query(req.question, top_k=req.top_k)

    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@app.post("/insights")
def get_insights(req: InsightRequest):
    """Extract insights from a transcript file, save to Databricks too."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    text = read_transcript(req.file_path)
    insights = extract_insights(text)

    if is_databricks_configured():
        save_insights_to_databricks(insights, os.path.basename(req.file_path))

    return {
        "insights": insights,
        "databricks_saved": is_databricks_configured(),
    }


@app.post("/orchestrate")
def orchestrate(req: OrchestrateRequest):
    """Main endpoint - ingest + insights + optionally answer a question."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    text = read_transcript(req.file_path)
    source = os.path.basename(req.file_path)

    response = {"insights": extract_insights(text)}

    if is_databricks_configured():
        save_insights_to_databricks(response["insights"], source)

    if req.question and req.question.strip():
        result = query(req.question)
        response["query_answer"] = result["answer"]
        response["query_sources"] = result["sources"]

    response["databricks_enabled"] = is_databricks_configured()
    return response


# ---- Databricks-specific endpoints ----
@app.get("/databricks/chunks")
def get_databricks_chunks(source_file: Optional[str] = None):
    """Read stored chunks from Databricks Delta table."""
    if not is_databricks_configured():
        raise HTTPException(status_code=400, detail="Databricks not configured")

    rows = get_all_chunks(source_name=source_file)
    return {"count": len(rows), "chunks": rows}


@app.get("/databricks/insights")
def get_databricks_insights(source_file: Optional[str] = None):
    """Read stored insights from Databricks Delta table."""
    if not is_databricks_configured():
        raise HTTPException(status_code=400, detail="Databricks not configured")

    rows = get_all_insights(source_name=source_file)
    return {"count": len(rows), "insights": rows}


if __name__ == "__main__":
    import uvicorn
    print("Starting server... Docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
