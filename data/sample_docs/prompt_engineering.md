# Prompt Engineering

Prompt engineering is the practice of designing inputs that steer a language
model toward useful, reliable outputs. In a RAG system, the prompt is where
retrieved context, instructions, and the user question come together.

## Key Techniques

**Clear instructions.** Tell the model exactly what to do: "Answer using only
the sources below. Cite every claim with [1], [2]."

**Few-shot examples.** Showing one or two input/output examples is often more
effective than a long description of the desired format.

**Delimiters.** Wrap retrieved context in clear markers (e.g. "Sources:" with
numbered entries) so the model can distinguish evidence from instructions.
This also reduces prompt-injection risk from untrusted documents.

## Grounding and Refusals

A grounded RAG prompt includes an explicit fallback: if the sources do not
contain the answer, the model should say "I don't know" instead of guessing.
This single instruction dramatically reduces hallucinations in practice.

## Temperature

For factual Q&A, keep the sampling temperature low (0.0–0.3). Higher values
add creativity at the cost of faithfulness to the retrieved sources.
