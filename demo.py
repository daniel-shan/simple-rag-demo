# Setup:
# python -m venv .myenv
# source .myenv/bin/activate
# pip install chromadb sentence-transformers

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Real embedding function using SentenceTransformer.
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        # 'input' is the expected parameter name for ChromaDB's embedding interface.
        embeddings = self.model.encode(input)
        return embeddings.tolist()

# Instantiate the embedding function.
embedding_fn = SentenceTransformerEmbeddingFunction()

# Configure ChromaDB with a persistence directory and disable telemetry.
settings = Settings(persist_directory=".chromadb_sentence", anonymized_telemetry=False)
client = chromadb.Client(settings=settings)

# Create (or get) a collection using the embedding function.
collection = client.create_collection(name="real_docs", embedding_function=embedding_fn)

# Add sample documents to the collection.
documents = [
    "Artificial intelligence is fun.",
    "The Declaration of Independence was signed in 1776.",
    "The sky is blue."
]

# Provide non-empty metadata for each document.
metadatas = [{"source": "example"} for _ in documents]

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=documents,
    metadatas=metadatas
)

# Define a query for retrieval.
query = "What should I do today?"
results = collection.query(query_texts=[query], n_results=1)

# Extract the retrieved document and its distance.
retrieved_doc = results["documents"][0][0]
distance = results["distances"][0][0]

print("Query:", query)
print("Retrieved Document:", retrieved_doc)
print("Distance:", distance)
