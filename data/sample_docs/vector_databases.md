# Vector Databases

A vector database stores dense embeddings and supports fast similarity search
over them. While a traditional database looks up rows by exact keys, a vector
database finds the vectors closest to a query vector, which makes it the
standard storage layer for RAG systems.

## How Similarity Search Works

Each chunk of text is converted into a fixed-length vector by an embedding
model. At query time the question is embedded with the same model, and the
database returns the chunks whose vectors are nearest to the query vector.
Cosine similarity and inner product are the most common distance metrics;
when vectors are L2-normalized, the two are equivalent.

## Approximate Nearest Neighbors

Exact search scans every vector, which is too slow at scale. Production
systems use approximate nearest neighbor (ANN) algorithms such as HNSW
(Hierarchical Navigable Small World) or IVF (inverted file index) to trade a
small amount of recall for large speedups. FAISS, the library used in this
project, implements both exact and ANN indexes.

## Popular Options

Well-known vector databases include Pinecone, Weaviate, Qdrant, Milvus, and
pgvector (a Postgres extension). For prototypes and portfolios, an in-process
library like FAISS or Chroma is simpler: no server to run, and the index can
be persisted to disk as a file.
