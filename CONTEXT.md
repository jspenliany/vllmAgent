# CONTEXT.md — RAG Pipeline for 400 Scattered Text Documents

## Glossary

**Document** — A single .txt file from the corpus of 400. Variable length (short to very long). Topics are scattered/diverse.

**Chunk** — A semantically coherent segment of a Document, produced by semantic chunking (embedding-based boundary detection). Each chunk carries its text and a reference to its parent Document section.

**Parent Section** — The larger document section (e.g., heading-delimited region) that contains a Chunk. Used for citation context.

**Dense Vector** — 1024-dim embedding from bge-m3 representing semantic meaning of a Chunk.

**Sparse Vector** — High-dim sparse embedding from bge-m3 (BM25/SPLADE-style) representing keyword/entity importance of a Chunk.

**Hybrid Search** — Combined retrieval using both Dense and Sparse vectors with weighted fusion (e.g., 0.7 dense + 0.3 sparse).

**Query Expansion** — LLM rewrites user query into multiple sub-queries or a better standalone question before retrieval.

**Answer Generation** — LLM synthesizes final answer from retrieved chunks, with citations.

**Citation** — Reference to the exact source: Document ID + Parent Section title/heading + Chunk text span.

**Milvus Collection** — Single collection storing all chunks with fields: id, text, source_id, section_title, chunk_id, dense_vector, sparse_vector.

**bge-m3 Service** — Containerized inference endpoint exposing `/v1/embeddings` returning both dense and sparse vectors.

**Gemma LLM** — Local LLM (via Ollama/vLLM) used for query expansion and answer generation.

**Ingestion Pipeline** — Batch process: load .txt → semantic chunk → embed (dense+sparse) → upsert Milvus. Runs periodically.

**Retrieval Pipeline** — Online process: expand query → hybrid search → (optional rerank) → pass to LLM for answer with citations.