# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation. Instead of relying solely on a language model's
parametric memory, a RAG system first retrieves relevant documents from an
external knowledge base and then conditions the model's answer on them.

## Why RAG?

Large language models have a fixed knowledge cutoff and tend to hallucinate
when asked about facts they have not seen during training. RAG addresses both
problems: the knowledge base can be updated at any time without retraining,
and grounding the answer in retrieved passages makes the output verifiable.

## The RAG Pipeline

A typical RAG pipeline has four stages:

1. **Ingestion**: documents are loaded, cleaned, and split into chunks.
2. **Indexing**: each chunk is embedded into a dense vector and stored in a
   vector database for similarity search.
3. **Retrieval**: at query time, the question is embedded and the top-k most
   similar chunks are fetched.
4. **Generation**: the retrieved chunks are inserted into the prompt as
   context, and the language model produces an answer with citations.

## Chunking Strategies

Chunking has a large impact on retrieval quality. Common strategies include
fixed-size chunking with overlap, paragraph-aware chunking that respects
natural boundaries, and semantic chunking that groups sentences by topic.
Smaller chunks improve retrieval precision; larger chunks preserve context.
A common starting point is 500 characters with 50 characters of overlap.
