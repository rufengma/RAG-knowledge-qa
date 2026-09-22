# Evaluating RAG Systems

Evaluation is what separates a demo from a system you can trust. RAG
evaluation has two layers: retrieval quality and answer quality.

## Retrieval Metrics

**Hit rate@k** measures the fraction of questions for which at least one
relevant document appears in the top-k retrieved chunks. **Recall@k** measures
the fraction of all relevant documents that appear in the top-k. Both are
computed against a golden set of question/answer pairs with known source
documents — no language model required.

## Answer Quality

Retrieval metrics do not tell you whether the final answer is correct.
Common approaches include:

- **Faithfulness**: is every claim in the answer supported by the retrieved
  sources? This can be checked by a human or by an LLM-as-judge.
- **Answer relevance**: does the answer actually address the question?
- **Human preference**: side-by-side comparison of two system variants.

## Building a Golden Set

Start small: 10–30 question/answer pairs written against your own documents,
each labeled with the source files a good retriever should find. Re-run the
eval after every change to chunking, embeddings, or top-k so regressions are
visible immediately.
