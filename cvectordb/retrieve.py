#!/usr/bin/env python3
"""
Retrieval + Answer Generation pipeline.
Query Expansion → Hybrid Search → Answer Generation with Citations.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# ============== CONFIG ==============
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "graphrag_chunks"
# bge-m3 embedding service
EMBED_URL = "http://192.168.198.1:8070/v1/embeddings"
EMBED_MODEL = "bge-m3"
# Gemma LLM (OpenAI-compatible API)
LLM_URL = "http://192.168.198.1:8000/v1"
LLM_MODEL = "gemma-4-31b-qat-it"
LLM_API_KEY = "none"
TOP_K_SUBQUERIES = 3
TOP_K_PER_SUBQUERY = 10
TOP_K_FINAL = 8
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
LOG_LEVEL = logging.INFO
# ====================================

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    section_title: str
    section_start: int
    section_end: int
    chunk_id: int
    score: float


class BGE_M3_Embedder:
    """Calls bge-m3 for dense + sparse query embeddings."""

    def __init__(self, url: str = EMBED_URL, model: str = EMBED_MODEL):
        self.url = url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def embed(self, texts: List[str]) -> tuple[list[list[float]], list[dict]]:
        dense_all, sparse_all = [], []
        for text in texts:
            resp = self.session.post(
                self.url,
                json={"input": [text], "model": self.model},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()["data"][0]
            dense_all.append(data["embedding"])
            sparse_all.append(data.get("sparse_embedding", {}))
        return dense_all, sparse_all


class QueryExpander:
    """LLM-based query expansion into diverse sub-queries."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a query rewriter for a RAG system over 400 scattered documents.
Rewrite the user's question into {n} diverse sub-queries that cover:
- Different phrasings / synonyms
- Specific entities / keywords likely in the corpus
- Broader and narrower scope variations

Output ONLY as JSON: {{"sub_queries": ["q1", "q2", ...]}}

User question: {query}
""")
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def expand(self, query: str, n: int = TOP_K_SUBQUERIES) -> List[str]:
        try:
            result = self.chain.invoke({"query": query, "n": n})
            sub_queries = result.get("sub_queries", [])
            # Ensure original query is included
            if query not in sub_queries:
                sub_queries.insert(0, query)
            return sub_queries[:n]
        except Exception as e:
            log.warning(f"Query expansion failed: {e}, using original query")
            return [query]


class HybridRetriever:
    """Milvus hybrid dense+sparse search."""

    def __init__(self, embedder: BGE_M3_Embedder):
        self.embedder = embedder
        self.client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    def search(self, query: str, top_k: int = TOP_K_PER_SUBQUERY) -> List[RetrievedChunk]:
        """Single query hybrid search."""
        dense_vec, sparse_vec = self.embedder.embed([query])

        dense_req = AnnSearchRequest(
            data=dense_vec,
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k
        )
        sparse_req = AnnSearchRequest(
            data=sparse_vec,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=top_k
        )

        results = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(DENSE_WEIGHT, SPARSE_WEIGHT),
            limit=top_k,
            output_fields=["text", "source_id", "section_title", "section_start", "section_end", "chunk_id"]
        )

        chunks = []
        for hit in results[0]:
            entity = hit["entity"]
            chunks.append(RetrievedChunk(
                text=entity["text"],
                source_id=entity["source_id"],
                section_title=entity["section_title"],
                section_start=entity["section_start"],
                section_end=entity["section_end"],
                chunk_id=entity["chunk_id"],
                score=hit["distance"] if "distance" in hit else hit.get("score", 0.0)
            ))
        return chunks

    def multi_query_search(self, queries: List[str], top_k: int = TOP_K_FINAL) -> List[RetrievedChunk]:
        """Run hybrid search for multiple queries, fuse & deduplicate."""
        all_chunks = []
        seen = set()

        # Parallel search for all sub-queries
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = {executor.submit(self.search, q, TOP_K_PER_SUBQUERY): q for q in queries}
            for future in as_completed(futures):
                chunks = future.result()
                for c in chunks:
                    # Deduplicate by (source_id, chunk_id)
                    key = (c.source_id, c.chunk_id)
                    if key not in seen:
                        seen.add(key)
                        all_chunks.append(c)

        # Sort by score descending, take top_k
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:top_k]


class AnswerGenerator:
    """LLM answer generation with citations."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a precise answerer. Given the user's question and retrieved context chunks,
synthesize a concise answer. Rules:
1. Only use information from the provided context
2. Cite every claim using the format: [source_id] section_title (chars X-Y)
   where source_id, section_title, and char range are provided for each chunk
3. If context is insufficient, say "Insufficient information in the provided documents"
4. Be concise; prefer bullet points for multi-part answers
5. Do not add information not in the context

Question: {question}

Context chunks:
{context}

Answer:
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks):
            cite = f"[{c.source_id}] {c.section_title} (chars {c.section_start}-{c.section_end})"
            lines.append(f"--- Chunk {i+1} {cite} ---\n{c.text}\n")
        return "\n".join(lines)

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Insufficient information in the provided documents."

        context = self._format_context(chunks)
        try:
            return self.chain.invoke({"question": question, "context": context})
        except Exception as e:
            log.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"


class RAGPipeline:
    """Full RAG pipeline: expand → retrieve → generate."""

    def __init__(self):
        self.embedder = BGE_M3_Embedder()
        self.llm = ChatOpenAI(
            base_url=LLM_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=0.1,
            timeout=60,
        )
        self.expander = QueryExpander(self.llm)
        self.retriever = HybridRetriever(self.embedder)
        self.generator = AnswerGenerator(self.llm)

    def query(self, question: str) -> Dict[str, Any]:
        log.info(f"Query: {question}")

        # 1. Query expansion
        sub_queries = self.expander.expand(question)
        log.info(f"Sub-queries: {sub_queries}")

        # 2. Hybrid search
        chunks = self.retriever.multi_query_search(sub_queries)
        log.info(f"Retrieved {len(chunks)} chunks")

        # 3. Answer generation
        answer = self.generator.generate(question, chunks)

        # 4. Prepare citations for frontend
        citations = []
        for c in chunks:
            citations.append({
                "source_id": c.source_id,
                "section_title": c.section_title,
                "char_range": f"{c.section_start}-{c.section_end}",
                "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "score": round(c.score, 4)
            })

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "sub_queries": sub_queries,
            "chunk_count": len(chunks)
        }


def main():
    """Interactive CLI for testing."""
    pipeline = RAGPipeline()
    print("RAG Pipeline ready. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            result = pipeline.query(question)

            print(f"\n📝 Answer:\n{result['answer']}\n")
            print("📚 Citations:")
            for i, c in enumerate(result['citations'], 1):
                print(f"  {i}. [{c['source_id']}] {c['section_title']} (chars {c['char_range']}) - score: {c['score']}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Error: {e}")
            print(f"Error: {e}\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()