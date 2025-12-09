from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Chunk folder
CHUNKS_DIR = "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/chunks"

# 2. Local embedding model (BGE-large)
embed_model = HuggingFaceEmbedding(
    model_name="/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models/bge-large-en-v1.5",
    normalize=True,
)

# 3. Configure PGVector store
vector_store = PGVectorStore.from_params(
    database="dge",
    host="localhost",
    port="5432",
    user="aakashwalavalkar",
    password="aakash1234",
    table_name="rag_chunks",   # table will be auto-created
    embed_dim=1024             # MUST match BGE-large
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Load .md chunks
documents = SimpleDirectoryReader(CHUNKS_DIR).load_data()

# 5. Build the index = embedding + inserting into PGVector
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("RAG index created & stored in PostgreSQL 'dge.rag_chunks'")