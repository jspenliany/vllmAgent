#!/usr/bin/env python3
"""
Retrieval + Answer Generation pipeline.
Query Expansion → Hybrid Search → Logic Template Matching → Instantiation → Answer Generation with Citations.

New: Cross-domain analogy via Logic Templates.
1. Detect target domain from query
2. Retrieve applicable logic templates (filter by transferable_domains)
3. Instantiate template to target domain
4. Generate answer with citations
"""

import os
import json
import logging
import re
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
TOP_K_PER_SUBQUERY = 20
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
    logic_template: Optional[Dict] = None
    logic_name: Optional[str] = None
    transferable_domains: Optional[List[str]] = None


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


class DomainDetector:
    """Detect target domain from user query."""

    DOMAIN_KEYWORDS = {
        "道路交通": ["交通", "车祸", "事故", "赔偿", "定损", "交强险", "商业险", "肇事", "逃逸", "醉驾", "闯红灯", "超速", "违章", "扣分", "罚款", "年检", "过户", "上牌", "驾照"],
        "消费纠纷": ["消费", "退款", "退货", "假货", "三包", "维权", "投诉", "12315", "商家", "平台", "外卖", "网购", "预付费", "办卡", "霸王条款"],
        "租赁纠纷": ["租房", "房东", "押金", "退租", "合同", "漏水", "噪音", "中介", "押一付三", "到期", "续租", "涨租"],
        "劳务纠纷": ["工资", "加班", "社保", "公积金", "离职", "赔偿", "劳动合同", "试用期", "年假", "工伤", "裁员", "N+1"],
        "邻里纠纷": ["邻里", "楼上", "楼下", "噪音", "漏水", "采光", "通风", "停车", "公共区域", "装修", "扰民"],
        "医疗纠纷": ["医疗", "医院", "医生", "手术", "误诊", "手术事故", "知情同意", "病历", "鉴定", "医患"],
        "家庭婚姻": ["离婚", "抚养费", "探视权", "财产分割", "彩礼", "继承", "遗嘱", "监护权", "家暴"],
    }

    def detect(self, query: str) -> List[str]:
        """Return list of detected domains, sorted by confidence."""
        scores = {}
        query_lower = query.lower()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[domain] = score
        # Sort by score descending
        return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def get_primary_domain(self, query: str) -> Optional[str]:
        domains = self.detect(query)
        return domains[0] if domains else None


class QueryExpander:
    """LLM-based query expansion into diverse sub-queries."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a query rewriting expert for a RAG system. Generate diverse sub-queries that comprehensively cover all critical dimensions of the user's question.

Output ONLY JSON: {{\"sub_queries\": [\"q1\", \"q2\", ...]}}

【Dimension Coverage Checklist】—— Must generate sub-queries covering at least these dimensions:
1. Core facts / phenomena / definitions
2. Causal mechanisms / underlying logic / core principles
3. Historical evolution / phased changes / key milestones
4. Positive constructions / technical accumulations / experience transfer / institutional building (NOT just criticism/reflection)
5. Key parameters / metrics / data comparisons (where applicable)
6. Concrete operations / implementation paths / implementation details / common pitfalls
7. Negative lessons / risks / limitations / failure cases
8. Cross-domain transfer / analogies / universal frameworks (where applicable)

User question: {query}
Number of sub-queries: {n}
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


class LogicInstantiator:
    """Instantiate logic template to target domain."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
你是 {target_domain} 领域专家。请将以下通用解决逻辑**实例化**到目标场景。

【通用逻辑模板】
{logic_template}

【目标场景】
用户问题: {query}
目标领域: {target_domain}

【实例化要求】
1. 保留框架骨架，每步骤写出：具体动作、依据法规/标准、所需材料、常见误区、时间节点
2. 补充目标领域特有的步骤（如交通事故需「勘察笔录」「定损单」）
3. 标注：⚠️ 必须做 / 💡 建议做 / 📌 关键证据
4. 引用原案例：{source_case_summary}

输出格式：
## {logic_name} —— {target_domain} 实例化指南

### 步骤 1: {action}
- 具体操作: ...
- 依据: ...
- 所需材料: ...
- ⚠️/💡/📌 ...

...
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    def instantiate(self, query: str, logic_template: Dict, target_domain: str) -> str:
        try:
            logic_name = logic_template.get("logic_template", {}).get("name", "通用解决框架")
            source_summary = logic_template.get("source_case_summary", "无原始案例摘要")
            logic_json = json.dumps(logic_template, ensure_ascii=False, indent=2)
            return self.chain.invoke({
                "query": query,
                "target_domain": target_domain,
                "logic_template": logic_json,
                "logic_name": logic_name,
                "source_case_summary": source_summary
            })
        except Exception as e:
            log.error(f"Logic instantiation failed: {e}")
            return f"逻辑实例化失败: {e}"


class LogicAwareRetriever:
    """Retrieve logic templates applicable to target domain."""

    def __init__(self, embedder: BGE_M3_Embedder):
        self.embedder = embedder
        self.client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    def find_applicable_logic(self, query: str, target_domain: str, top_k: int = 5) -> List[Dict]:
        """Find logic templates applicable to target domain."""
        dense_vec, sparse_vec = self.embedder.embed([query])

        # Filter by transferable_domains containing target_domain
        filter_expr = f'transferable_domains like "%{target_domain}%"'

        dense_req = AnnSearchRequest(
            data=dense_vec,
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=20
        )
        sparse_req = AnnSearchRequest(
            data=sparse_vec,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=20
        )

        results = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(DENSE_WEIGHT, SPARSE_WEIGHT),
            limit=20,
            filter=filter_expr,
            output_fields=["logic_template", "logic_name", "source_case_summary", "source_id", "text", "section_title", "section_start", "section_end", "chunk_id"]
        )

        # Deduplicate by logic_name (same logic template may appear in multiple chunks)
        seen = set()
        unique_results = []
        for hit in results[0]:
            entity = hit["entity"]
            logic_name = entity.get("logic_name")
            if logic_name and logic_name not in seen:
                seen.add(logic_name)
                entity["score"] = hit.get("distance", hit.get("score", 0.0))
                unique_results.append(entity)
                if len(unique_results) >= top_k:
                    break
        return unique_results


class AnswerGenerator:
    """LLM answer generation with citations and logic template context."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a domain expert. Based on the user's question, instantiated logic guide (if any), and retrieved context, generate a **structured, multi-dimensional, well-cited** answer.

【Required Answer Structure】—— Must include the following modules (where applicable):
1. **Core Conclusion / Direct Answer** (1-2 sentences)
2. **Core Mechanism / Principle / Logic Framework** (step-by-step, structured)
3. **Positive Constructions / Technical Accumulations / Institutional Evolution** (NOT just criticism/reflection)
4. **Key Parameters / Metrics / Data Comparison Tables** (where quantifiable metrics are involved)
5. **Concrete Operations / Implementation Steps / Implementation Details / Common Pitfalls**
6. **Negative Lessons / Risks / Limitations / Boundary Conditions**
7. **Cross-Domain Transfer / Analogies / Universal Insights** (where applicable)
8. **Uncertainties / Controversies / Evolution Directions**

【Citation Rules】:
- Every key claim must be cited: [source_id] section_title (chars X-Y)
- Prioritize citing the original case from the instantiated logic guide
- If context is insufficient, explicitly state: "Insufficient information in the provided documents"

【Output Format】:
- Use Markdown heading hierarchy (## / ###)
- Use lists, tables, bold for readability
- Annotate: ⚠️ Must / 💡 Recommended / 📌 Key Evidence / 🔄 Transfer Condition

---

User Question: {question}

Instantiated Logic Guide:
{instantiated_logic}

Relevant Context Chunks:
{context}

---
Generate your answer:
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks):
            cite = f"[{c.source_id}] {c.section_title} (chars {c.section_start}-{c.section_end})"
            lines.append(f"--- Chunk {i+1} {cite} ---\n{c.text}\n")
        return "\n".join(lines)

    def generate(self, question: str, chunks: List[RetrievedChunk], instantiated_logic: str = "") -> str:
        if not chunks and not instantiated_logic:
            return "提供的文档中没有相关信息。"

        context = self._format_context(chunks)
        try:
            return self.chain.invoke({
                "question": question,
                "context": context,
                "instantiated_logic": instantiated_logic or "（无适用逻辑模板，基于上下文直接回答）"
            })
        except Exception as e:
            log.error(f"Answer generation failed: {e}")
            return f"生成回答失败: {e}"


class RAGPipeline:
    """Full RAG pipeline: domain detection → logic retrieval → instantiation → retrieval → generation."""

    def __init__(self):
        self.embedder = BGE_M3_Embedder()
        self.llm = ChatOpenAI(
            base_url=LLM_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=0.1,
            timeout=180,  # Increase from 60 to 120 seconds
            max_retries=3,  # Add retries
            request_timeout=180,  # Explicit request timeout
        )
        self.domain_detector = DomainDetector()
        self.expander = QueryExpander(self.llm)
        self.logic_retriever = LogicAwareRetriever(self.embedder)
        self.logic_instantiator = LogicInstantiator(self.llm)
        self.retriever = HybridRetriever(self.embedder)
        self.generator = AnswerGenerator(self.llm)

    def query(self, question: str) -> Dict[str, Any]:
        log.info(f"Query: {question}")

        # 1. Domain detection
        target_domain = self.domain_detector.get_primary_domain(question)
        if not target_domain:
            # Fallback: use first detected or default
            domains = self.domain_detector.detect(question)
            target_domain = domains[0] if domains else "通用"
        log.info(f"Target domain: {target_domain}")

        # 2. Query expansion
        sub_queries = self.expander.expand(question)
        log.info(f"Sub-queries: {sub_queries}")

        # 3. Logic template retrieval (if domain detected)
        instantiated_logic = ""
        logic_template_used = None
        if target_domain != "通用":
            logic_results = self.logic_retriever.find_applicable_logic(question, target_domain, top_k=3)
            if logic_results:
                # Use the top logic template
                top_logic = logic_results[0]
                logic_template_used = top_logic.get("logic_template")
                logic_name = top_logic.get("logic_name", "未知")
                log.info(f"Found applicable logic: {logic_name}")
                
                # 4. Logic instantiation
                instantiated_logic = self.logic_instantiator.instantiate(
                    question, logic_template_used, target_domain
                )
                log.info("Logic instantiated")

        # 5. Regular chunk retrieval
        chunks = self.retriever.multi_query_search(sub_queries)
        log.info(f"Retrieved {len(chunks)} chunks---before generation")

        # 6. Answer generation
        answer = self.generator.generate(question, chunks, instantiated_logic)
        log.info(f"Retrieved {len(chunks)} chunks---after generation")
        # 7. Prepare citations
        citations = []
        for c in chunks:
            citations.append({
                "source_id": c.source_id,
                "section_title": c.section_title,
                "char_range": f"{c.section_start}-{c.section_end}",
                "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "score": round(c.score, 4),
                "logic_name": c.logic_name,
                "transferable_domains": c.transferable_domains
            })

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "sub_queries": sub_queries,
            "chunk_count": len(chunks),
            "target_domain": target_domain,
            "logic_template_used": logic_template_used.get("logic_template", {}).get("name") if logic_template_used else None,
            "instantiated_logic_preview": instantiated_logic[:300] + "..." if instantiated_logic else None
        }


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
            output_fields=["text", "source_id", "section_title", "section_start", "section_end", "chunk_id", "logic_template", "logic_name", "transferable_domains"]
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
                score=hit["distance"] if "distance" in hit else hit.get("score", 0.0),
                logic_template=entity.get("logic_template"),
                logic_name=entity.get("logic_name"),
                transferable_domains=entity.get("transferable_domains", [])
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
                    log.info(f"Processing chunk {c.chunk_id} chunk text: {c.text}")
                    # Deduplicate by (source_id, chunk_id)
                    key = (c.source_id, c.chunk_id)
                    if key not in seen:
                        seen.add(key)
                        all_chunks.append(c)

        # Sort by score descending, take top_k
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:top_k]



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