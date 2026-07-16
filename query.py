import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---- config from .env ----
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3") # used in place of llama3 due to size issue with llama3
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))

COLLECTION = "meeting_transcripts"


# ---- Step 1: Connect to the same ChromaDB we filled in Task 1 ----
def get_chroma_db():
    """Open the ChromaDB that was populated by ingest.py"""
    embedding_fn = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PATH
    )
    return db


# ---- Step 2: Search for relevant chunks ----
def find_relevant_chunks(question, top_k=None):
    if top_k is None:
        top_k = TOP_K

    db = get_chroma_db()
    results = db.similarity_search_with_score(question, k=top_k)

    return results


# ---- Step 3: Build context from retrieved chunks ----
def build_context(search_results):
    context_pieces = []

    for i, (doc, score) in enumerate(search_results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_num = doc.metadata.get("chunk_index", "?")
        text = doc.page_content

        piece = f"[Chunk {i} | From: {source} | Part {chunk_num}]\n{text}"
        context_pieces.append(piece)

    # join all pieces with a separator between them
    full_context = "\n\n---\n\n".join(context_pieces)
    return full_context


# ---- Step 4: Ask the LLM to answer using the context ----
PROMPT_TEMPLATE = """You are a helpful meeting assistant. Answer the question
based ONLY on the meeting transcript context provided below.

If the context doesn't contain enough information to answer, say
"I don't have enough information from the transcripts to answer that."

Do not make up anything. Only use what's in the context.

Context from meeting transcripts:
{context}

Question: {question}

Answer:"""


def ask_llm(question, context):
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # create a simple chain: prompt -> llm
    chain = prompt | llm

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer


# ---- Full query pipeline: question -> search -> context -> LLM -> answer ----
def query(question, top_k=None):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    # step 1: find relevant chunks
    print("Searching ChromaDB for relevant chunks...")
    search_results = find_relevant_chunks(question, top_k)
    print(f"  Found {len(search_results)} relevant chunks")

    # step 2: build context string from those chunks
    context = build_context(search_results)

    # step 3: send to LLM
    print("Asking LLM for an answer...")
    answer = ask_llm(question, context)

    # collect source info for reference
    sources = []
    for doc, score in search_results:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "relevance_score": round(float(score), 4),
            "preview": doc.page_content[:100] + "..."
        })

    print(f"\nAnswer: {answer}")
    print(f"\nSources used: {len(sources)} chunks")
    for s in sources:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score: {s['relevance_score']})")

    return {
        "answer": answer,
        "sources": sources
    }


# ---- Run directly to test with sample questions ----
if __name__ == "__main__":
    # make sure you've run ingest.py first!
    test_questions = [
        "What was decided about rate limiting?",
        "Who is responsible for the dashboard redesign?",
        "What technology will be used for push notifications?",
    ]

    for q in test_questions:
        result = query(q)
        print("\n" + "=" * 60 + "\n")
