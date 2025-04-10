import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Real embedding function using SentenceTransformer.
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def __call__(self, input):
        # 'input' is expected by ChromaDB's embedding interface.
        embeddings = self.model.encode(input)
        return embeddings.tolist()

# Instantiate the embedding function.
embedding_fn = SentenceTransformerEmbeddingFunction()

# Configure ChromaDB persistence and disable telemetry.
settings = Settings(persist_directory=".chromadb_advanced", anonymized_telemetry=False)
client = chromadb.Client(settings=settings)

# Create (or reuse) a collection using our embedding function.
collection = client.create_collection(name="advanced_docs", embedding_function=embedding_fn)

# Define documents with additional metadata.
documents = [
    "The latest advancements in quantum computing are remarkable.",
    "Artificial intelligence is impacting many industries.",
    "Climate change is an urgent global issue that requires immediate action.",
    "Quantum entanglement challenges our understanding of physics.",
    "Deep learning models are revolutionizing data analysis."
]
ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Each document now has metadata, for example, topic and category.
metadatas = [
    {"topic": "quantum", "category": "science"},
    {"topic": "ai", "category": "technology"},
    {"topic": "climate", "category": "environment"},
    {"topic": "quantum", "category": "science"},
    {"topic": "ai", "category": "technology"}
]

# Add documents to the collection.
collection.add(ids=ids, documents=documents, metadatas=metadatas)

# ================================================
# 1. Metadata Filtering: Retrieve only documents tagged with "quantum".
# ================================================
query = "What are the recent developments in quantum physics?"
filter_criteria = {"topic": "quantum"}

# Note: Replace 'filter' with 'where' per the updated API.
results = collection.query(query_texts=[query], where=filter_criteria, n_results=2)
retrieved_docs = results["documents"][0]
distances = results["distances"][0]

print("Filtered Query Results (topic 'quantum'):")
for doc, dist in zip(retrieved_docs, distances):
    print(f" - {doc} (distance: {dist:.4f})")

# ================================================
# 2. Advanced RAG: Multi-document Retrieval and Prompt Construction.
# ================================================
# Retrieve multiple documents (without filtering) to build richer context.
multi_results = collection.query(query_texts=[query], n_results=3)
retrieved_docs_multi = multi_results["documents"][0]

# Construct a prompt that aggregates context from the retrieved documents.
rag_prompt = f"Question: {query}\n"
for i, doc in enumerate(retrieved_docs_multi, start=1):
    rag_prompt += f"\nContext {i}: {doc}"
rag_prompt += "\n\nAnswer:"

print("\nConstructed RAG Prompt:")
print(rag_prompt)

# ================================================
# 3. Updating Metadata: Change a document's metadata.
# ================================================
# For example, reclassify document 'doc3' from 'environment' to 'science'.
print("\nBefore metadata update:")
original_result = collection.query(query_texts=["How urgent is climate change?"], n_results=1)
print("Retrieved document:", original_result["documents"][0][0])

# Update metadata for document 'doc3'.
collection.update(ids=["doc3"], metadatas=[{"topic": "climate", "category": "science"}])

print("\nAfter metadata update:")
updated_result = collection.query(query_texts=["How urgent is climate change?"], n_results=1)
print("Retrieved document:", updated_result["documents"][0][0])
