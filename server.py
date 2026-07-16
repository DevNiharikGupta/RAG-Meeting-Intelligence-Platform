import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from ingest import read_transcript, split_into_chunks, store_in_chroma
from query import query
from insights import extract_insights

app = FastAPI(title="Meeting Intelligence API", version="1.0.0")


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
    """Chunk a transcript file, embed it, store in ChromaDB."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    text = read_transcript(file_path)
    chunks = split_into_chunks(text)
    count = store_in_chroma(chunks, source_name=os.path.basename(file_path))

    return {"message": "Transcript ingested", "chunks_created": count}



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
    """Extract action items, decisions, discussion points from a transcript file."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    text = read_transcript(req.file_path)
    insights = extract_insights(text)
    return {"insights": insights}


@app.post("/orchestrate")
def orchestrate(req: OrchestrateRequest):
    """Main endpoint - extracts insights + optionally answers a question."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    text = read_transcript(req.file_path)

    # always extract insights
    response = {"insights": extract_insights(text)}

    # if a question was also asked, answer it using RAG
    if req.question and req.question.strip():
        result = query(req.question)
        response["query_answer"] = result["answer"]
        response["query_sources"] = result["sources"]

    return response



if __name__ == "__main__":
    import uvicorn
    print("Starting server... Docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
