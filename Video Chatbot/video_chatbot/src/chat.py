import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from .config import DB_PATH
from .llm_setup import setup_ai_models


class ChatBot:
    def __init__(self):
        print("initializing ChatBot...")

        self.llm, _ = setup_ai_models()

        print(f"Connecting to DB at: {DB_PATH}")
        try:
            db = chromadb.PersistentClient(path=DB_PATH)
            chroma_collection = db.get_or_create_collection("video_rag")
            count = chroma_collection.count()
            print(f"Database contains {count} fragments/chunks.")

            if count == 0:
                print("WARNING: DATABASE IS EMPTY! Please run Option 1 first.")

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(vector_store)

            self.chat_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            print("Chat Engine Created.")

        except Exception as e:
            print(f"Error during init: {e}")

    def ask(self, query):
        print(f"Asking LLM: '{query}'")
        try:
            response = self.chat_engine.query(query)

            for node in response.source_nodes:
                print("SOURCE:", node.node.text[:120])
            return response
        except Exception as e:
            return f"Error during query: {e}"