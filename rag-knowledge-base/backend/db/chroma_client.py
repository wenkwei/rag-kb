import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
)

vector_store = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PERSIST_DIR),
)

# Native Chroma client for filtered operations
_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
_collection = _client.get_or_create_collection("rag_knowledge_base")


def get_vector_store() -> Chroma:
    return vector_store


def get_collection():
    return _collection


def get_embedding_function():
    return embeddings
