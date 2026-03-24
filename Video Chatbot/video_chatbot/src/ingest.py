import chromadb
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from .config import DB_PATH, DOCS_DIR
from .llm_setup import setup_ai_models


def create_vector_db():
    import os
    import shutil
    import chromadb
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from .config import DB_PATH, DOCS_DIR
    from .llm_setup import setup_ai_models
    setup_ai_models()
    # 🔴 1. HARD RESET VECTOR DB
    if os.path.exists(DB_PATH):
        print("Clearing existing vector database...")
        shutil.rmtree(DB_PATH)

    os.makedirs(DB_PATH, exist_ok=True)

    # 🔴 2. VERIFY TRANSCRIPT EXISTS
    print(f"Reading transcript from: {DOCS_DIR}")
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise RuntimeError("No transcript found. Run video captioning first.")

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"Indexed {len(documents)} chunks for current video")
    # 🔴 3. CREATE FRESH CHROMA CLIENT + COLLECTION
    db = chromadb.PersistentClient(path=DB_PATH)

    chroma_collection = db.create_collection(
        name="video_rag",
        metadata={"source": "current_video"}
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Indexing data...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    index.storage_context.persist()
    print(f"Database updated at: {DB_PATH}")

    # 🔴 4. LOAD LLM + EMBEDDINGS *AFTER* INDEXING
    

    return index