import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# config from .env
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
COLLECTION = "meeting_transcripts"


def read_transcript(filepath):
    """Read a .txt file and return its text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text):
    """Break long text into smaller overlapping pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def get_embedding_fn():
    """Returns an Ollama embedding function."""
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


def store_in_chroma(chunks, source_name="unknown"):
    """Embed chunks and save them to ChromaDB."""
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_index": i, "total_chunks": len(chunks)}
        )
        documents.append(doc)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedding_fn(),
        persist_directory=CHROMA_PATH
    )
    db.add_documents(documents)
    return len(documents)


# run directly to test
if __name__ == "__main__":
    filepath = "./data/transcripts/sprint_planning.txt"

    print(f"Reading: {filepath}")
    text = read_transcript(filepath)
    print(f"  {len(text)} characters")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text)
    print(f"  {len(chunks)} chunks created")

    print("Embedding and storing in ChromaDB...")
    count = store_in_chroma(chunks, source_name=os.path.basename(filepath))
    print(f"  {count} chunks stored. Done!")
