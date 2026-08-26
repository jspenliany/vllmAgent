#!/usr/bin/env python3
"""
Batch ingestion pipeline for 400 scattered .txt documents.
Loads → semantic chunks → embeds (dense+sparse) → upserts to Milvus.
Run periodically (cron/systemd timer) for incremental updates.

New: Logic Template Extraction for cross-domain analogy transfer.
Each document gets a "logic_template" JSON extracted by LLM,
enabling cross-domain reasoning (e.g., neighbor dispute logic → traffic dispute).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusClient
)
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ============== CONFIG ==============
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "graphrag_chunks"
DIM = 1024
# bge-m3 embedding service (used for BOTH semantic chunking + vector embeddings)
EMBED_URL = "http://192.168.198.1:8070/v1/embeddings"
EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 90
# Gemma LLM (OpenAI-compatible API) for logic extraction
LLM_URL = "http://192.168.198.1:8000/v1"
LLM_MODEL = "gemma-4-31b-qat-it"
LLM_API_KEY = "none"
LLM_TIMEOUT = 120
LLM_REQUEST_TIMEOUT = 120
LLM_RAW_TEXT = 6000
BATCH_EMBED_SIZE = 32
BATCH_INSERT_SIZE = 128
DOCS_DIR = Path("../resources/txt_corpus")  # <-- change to your 400 .txt files
MANIFEST_PATH = Path("./ingest_manifest.json")
LOG_LEVEL = logging.INFO
# ====================================

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============== LOGIC EXTRACTION PROMPT ==============
LOGIC_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""您是一位经验丰富的分析师，擅长从各个领域（技术、商业、科学、教育、法律、医疗、工程、政策等）的文档中提取**可复用的问题解决框架**。

提取**逻辑模板**——一种结构化的、可复用的问题解决框架，可以应用于其他领域中的类似问题。

仅输出 JSON 数据（不包含额外文本）：

{{
    "logic_template": {{
        "name": "框架的简要名称（例如，“根本原因分析与纠正措施框架”、“利益相关者协调与决策框架”、“迭代实验与验证框架”）",
        "description": "此框架解决的问题的一句话总结",
        "steps": [
            {{
                "step": 1,
                "phase": "阶段的简短名称（例如，“问题定义”、“数据收集”、“假设形成”）",
                "key_question": "此阶段回答的核心问题",
                "method": "使用的具体技术、工具或方法",
                "outputs": "产生的关键交付成果或决策",
                "success_criteria": "如何判断此阶段已完成"
            }},
            {{
                "step": 2,
                "phase": "阶段的简短名称（例如，“问题定义”、“数据收集”、“假设形成”） {{ “根本原因分析”、“解决方案设计”、“实验设计”）”
                “key_question”：此阶段回答的核心问题，
                “method”：使用的具体技术、工具或方法，
                “outputs”：产生的关键交付成果或决策，
                “success_criteria”：如何判断此阶段已完成”
            }},
            {{
                “step”：3，
                “phase”：阶段简称（例如，“解决方案实施”、“验证”、“迭代”），
                “key_question”：此阶段回答的核心问题，
                “method”：使用的具体技术、工具或方法，
                “outputs”：产生的关键交付成果或决策，
                “success_criteria”：如何判断此阶段已完成”
            }},
            {{
                “step”：4，
                “phase”：阶段简称（例如，“验证”、“优化”， {{ '部署')",
                "key_question": "此阶段回答的核心问题",
                "method": "使用的具体技术、工具或方法",
                "outputs": "产生的关键交付成果或决策",
                "success_criteria": "如何判断此阶段已完成"
            }},
            {{
                "step": 5,
                "phase": "阶段简称（例如，“监控”、“反馈收集”、“持续改进”）",
                "key_question": "此阶段回答的核心问题",
                "method": "使用的具体技术、工具或方法",
                "outputs": "产生的关键交付成果或决策",
                "success_criteria": "如何判断此阶段已完成"
            }},
            {{
                "step": 6,
                "phase": "阶段简称（例如，“知识获取”、“文档编制”、“知识转移”）",
                "key_question": "本阶段解答的核心问题",
                "method": "使用的具体技术、工具或方法",
                "outputs": "产生的关键交付成果或决策",
                "success_criteria": "如何判断本阶段完成"
            }},
            {{
                "step": 7,
                "phase": "阶段简称（例如，“审查”、“回顾”、“框架演进”）",
                "key_question": "本阶段解答的核心问题",
                "method": "使用的具体技术、工具或方法",
                "outputs": "产生的关键交付成果或决策",
                "success_criteria": "如何判断本阶段完成"
            }}
        ],
        "guardrails": [
            "必须遵守的约束、界限或原则",
            "需要避免的常见失效模式",
            "伦理/法律/安全界限"
        ],
        "applicability": {{
            "core_domain": "源文档的主要领域（例如，“软件调试”、“公共卫生政策”、“制造质量”）",
            "transferable_domains": ["domain1", "domain2", "domain3"],
            "transfer_conditions": "此框架的适用条件（例如，“涉及多个利益相关者和可衡量结果的问题”、“具有反馈回路和可观察指标的系统”）"
        }},
        "key_assumptions": [
            "框架所依赖的假设（例如，“利益相关者是理性行为者”、“数据可用”、“系统可观察”）"
        ]
    }},
    "source_case_summary": "源文档的问题、方法和结果的 200 字摘要"
}}

Document content:
{doc_text}
""")

# LLM for logic extraction (initialized in main)
logic_extraction_chain = None


@dataclass
class FileManifest:
    path: str
    hash: str
    size: int
    mtime: float
    chunk_count: int = 0
    section_count: int = 0
    logic_extracted: bool = False
    logic_template: Optional[Dict[str, Any]] = None


class BGE_M3_Embeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper for local bge-m3 service.
    Used by SemanticChunker for boundary detection.
    """

    def __init__(self, url: str = EMBED_URL, model: str = EMBED_MODEL, batch_size: int = BATCH_EMBED_SIZE):
        self.url = url
        self.model = model
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.session.post(
            self.url,
            json={"input": texts, "model": self.model},
            timeout=EMBED_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (for chunking boundary detection)."""
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_vecs.extend(self._embed_batch(batch))
        return all_vecs

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (for chunking)."""
        return self._embed_batch([text])[0]


class BGE_M3_Embedder:
    """Calls bge-m3 container for dense + sparse embeddings (for vector storage)."""

    def __init__(self, url: str = EMBED_URL, model: str = EMBED_MODEL, batch_size: int = BATCH_EMBED_SIZE):
        self.url = url
        self.model = model
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def embed(self, texts: List[str]) -> tuple[list[list[float]], list[dict]]:
        """Returns (dense_vectors, sparse_vectors)."""
        dense_all, sparse_all = [], []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.session.post(
                self.url,
                json={"input": batch, "model": self.model},
                timeout=EMBED_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            dense_all.extend([d["embedding"] for d in data])
            sparse_all.extend([d.get("sparse_embedding", {}) for d in data])
        return dense_all, sparse_all


class ManifestStore:
    """Simple JSON manifest for incremental ingestion."""

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, FileManifest] = {}
        self.load()

    def load(self):
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self.data = {k: FileManifest(**v) for k, v in raw.items()}
            log.info(f"Loaded manifest: {len(self.data)} files")

    def save(self):
        with open(self.path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.data.items()}, f, indent=2)

    def get(self, key: str) -> Optional[FileManifest]:
        return self.data.get(key)

    def upsert(self, manifest: FileManifest):
        self.data[manifest.path] = manifest

    def delete(self, key: str):
        self.data.pop(key, None)

    def all_keys(self) -> set:
        return set(self.data.keys())


def all_keys(self) -> set:
        return set(self.data.keys())


def extract_json_from_response(response: str) -> Optional[str]:
    """Extract valid JSON from LLM response, handling markdown code blocks."""
    import re
    log.info(f"Extracting JSON from response: before")
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    log.info(f"Extracting JSON from response: after")
    # Try to find bare JSON object
    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    return None


def validate_logic_template(data: Dict) -> bool:
    """Validate logic template has required structure."""
    try:
        lt = data.get("logic_template")
        log.info(f"Validating logic template: {lt}")
        if not lt or not isinstance(lt, dict):
            return False

        # Required fields
        required = ["name", "steps", "applicability"]
        for field in required:
            if field not in lt:
                return False

        # Steps must be list with at least 1 step
        steps = lt.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            return False

        # Each step must have required fields
        for step in steps:
            if not all(k in step for k in ["step", "phase", "key_question", "method"]):
                return False

        # Applicability must have transferable_domains as list
        app = lt.get("applicability")
        if not isinstance(app, dict):
            return False
        if not isinstance(app.get("transferable_domains"), list):
            return False

        return True
    except Exception:
        return False


def extract_logic_template_long(text: str, llm_chain, max_input=LLM_RAW_TEXT) -> Optional[Dict]:
    """分层提取：长文档先摘要，再从摘要提取逻辑"""

    if len(text) <= max_input:
        return extract_logic_template(text, llm_chain)

    # Step 1: 分块摘要（并行，快）
    chunk_size = int(LLM_RAW_TEXT/2)
    overlap = 200
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    log.info(f"batch mode.....Chunk count: {len(chunks)}")
    summaries = []
    for chunk in chunks:
        summary_prompt = f"用 400 字总结文本的核心问题、方法、逻辑、事实、结论：\n\n{chunk}"
        summary = llm_chain.invoke({"doc_text": summary_prompt})  # 复用同一 LLM
        summaries.append(summary)

    # Step 2: 合并摘要再提取逻辑
    combined_summary = "\n\n---\n\n".join(summaries)
    return extract_logic_template(combined_summary, llm_chain)

def extract_logic_template(text: str, llm_chain) -> Optional[Dict[str, Any]]:
    """Extract logic template from document text using LLM."""
    try:
        log.info(f"enter into Extracting logic template: {text}")
        # Use first 4000 chars for logic extraction (cost control)
        truncated = text[:LLM_RAW_TEXT]
        result = llm_chain.invoke(truncated)
        # Handle both string (raw) and dict (parsed by JsonOutputParser)
        if isinstance(result, dict):
            logic_data = result
        elif isinstance(result, str):
            json_str = extract_json_from_response(result)
            if not json_str:
                log.warning("No valid JSON found in LLM response")
                return None
            logic_data = json.loads(json_str)
        else:
            log.warning(f"Unexpected result type: {type(result)}")
            return None

        # Validate required structure
        if not validate_logic_template(logic_data):
            log.warning("Logic template validation failed")
            return None

        return logic_data
    except Exception as e:
        log.warning(f"Logic extraction failed .....****.....: {e}")
        return None

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_files(docs_dir: Path) -> List[Path]:
    return sorted(docs_dir.rglob("*.txt"))


def split_sections(text: str) -> List[Dict[str, Any]]:
    """
    Infer sections from plain text heuristics:
    - ALL CAPS lines (likely headings)
    - Numbered sections (1., 1.1, etc.)
    - Lines followed by blank line then content
    Returns list of {title, start_char, end_char, text}
    """
    lines = text.splitlines(keepends=True)
    sections = []
    current_title = "Document Start"
    section_start = 0
    char_pos = 0

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # ALL CAPS (at least 3 chars, mostly letters)
        if len(stripped) >= 3 and stripped.isupper() and any(c.isalpha() for c in stripped):
            return True
        # Numbered: 1.  1.1  2)  (1)  etc.
        import re
        if re.match(r"^[\d\(\)\.\s]+$", stripped) and any(c.isdigit() for c in stripped):
            return True
        return False

    for i, line in enumerate(lines):
        if is_heading(line) and i > 0:
            # Close previous section
            section_end = char_pos
            if section_end > section_start:
                sections.append({
                    "title": current_title,
                    "start_char": section_start,
                    "end_char": section_end,
                    "text": text[section_start:section_end]
                })
            current_title = line.strip()
            section_start = char_pos
        char_pos += len(line)

    # Last section
    if char_pos > section_start:
        sections.append({
            "title": current_title,
            "start_char": section_start,
            "end_char": char_pos,
            "text": text[section_start:char_pos]
        })

    return sections if sections else [{
        "title": "Document Start",
        "start_char": 0,
        "end_char": len(text),
        "text": text
    }]


# ingest.py - update semantic_chunk_section function

def semantic_chunk_section(section_text: str, embeddings: BGE_M3_Embeddings) -> List[Dict[str, Any]]:
    """
    Hybrid: Semantic boundaries + hard size limits.
    """
    # 1. First: Semantic chunking (find semantic boundaries)
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,  # 85th percentile = more aggressive splitting (was 70)
        min_chunk_size=200,  # Minimum meaningful chunk
    )
    lc_docs = chunker.create_documents([section_text])

    # 2. Hard limit: Split oversized chunks by tokens (~500 tokens max)
    MAX_TOKENS = 500  # ~350-400 Chinese chars, safe for embedding + LLM context

    def estimate_tokens(text: str) -> int:
        # Rough: Chinese ~1.5 chars/token, English ~4 chars/token
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def split_by_tokens(text: str, max_tokens: int) -> List[str]:
        """Split text by token limit, preferring sentence boundaries."""
        if estimate_tokens(text) <= max_tokens:
            return [text]

        # Split by sentences first
        import re
        sentences = re.split(r'(?<=[。！？!?.])', text)
        chunks = []
        current = ""
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = estimate_tokens(sent)
            if current_tokens + sent_tokens > max_tokens and current:
                chunks.append(current)
                current = sent
                current_tokens = sent_tokens
            else:
                current += sent
                current_tokens += sent_tokens
        if current:
            chunks.append(current)
        return chunks

    # 3. Apply token limit to each semantic chunk
    final_chunks = []
    for doc in lc_docs:
        sub_chunks = split_by_tokens(doc.page_content, MAX_TOKENS)
        for sub in sub_chunks:
            # Find char offsets in section_text (approximate)
            start = section_text.find(sub[:50])
            if start == -1:
                start = 0
            end = start + len(sub)
            final_chunks.append({
                "text": sub,
                "start_char": start,
                "end_char": end
            })

    return final_chunks

def process_file(
    file_path: Path,
    embedder: BGE_M3_Embedder,
    embeddings: BGE_M3_Embeddings,
    manifest: ManifestStore,
    logic_chain
) -> Optional[FileManifest]:
    """Process a single .txt file: load → section → semantic chunk → embed → logic extraction."""
    rel_path = str(file_path.relative_to(DOCS_DIR))
    stat = file_path.stat()
    fhash = file_hash(file_path)

    # Check manifest
    existing = manifest.get(rel_path)
    if existing and existing.hash == fhash:
        log.debug(f"Unchanged: {rel_path}")
        return existing

    log.info(f"Processing: {rel_path} ({stat.st_size} bytes)")

    # Load
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Section split
    sections = split_sections(text)

    # Extract logic template from full document (once per file)
    logic_template = extract_logic_template_long(text, logic_chain)
    if logic_template:
        log.info(f"  → Logic template extracted: {logic_template.get('logic_template', {}).get('name', 'unnamed')}")
    else:
        log.warning(f"  → Logic extraction failed, proceeding without logic template")

    all_chunks = []
    for sec_idx, sec in enumerate(sections):
        sec_chunks = semantic_chunk_section(sec["text"], embeddings)
        for chunk_idx, chunk in enumerate(sec_chunks):
            global_start = sec["start_char"] + chunk["start_char"]
            global_end = sec["start_char"] + chunk["end_char"]
            chunk_data = {
                "text": chunk["text"],
                "source_id": rel_path,
                "section_title": sec["title"],
                "section_start": global_start,
                "section_end": global_end,
                "chunk_id": chunk_idx,
                "section_idx": sec_idx,
                # Logic template fields - ONLY include if extraction succeeded
                "logic_template": logic_template,
                "logic_name": logic_template.get("logic_template", {}).get("name") if logic_template else None,
                "transferable_domains": logic_template.get("logic_template", {}).get("applicability", {}).get(
                    "transferable_domains", []) if logic_template else [],
            }
            all_chunks.append(chunk_data)

    if not all_chunks:
        log.warning(f"No chunks produced for {rel_path}")
        return None

    # Embed all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    dense_vecs, sparse_vecs = embedder.embed(chunk_texts)

    # Attach vectors
    for i, chunk in enumerate(all_chunks):
        chunk["dense_vector"] = dense_vecs[i]
        chunk["sparse_vector"] = sparse_vecs[i]

    # Upsert to Milvus
    upsert_chunks(all_chunks)

    # Update manifest
    new_manifest = FileManifest(
        path=rel_path,
        hash=fhash,
        size=stat.st_size,
        mtime=stat.st_mtime,
        chunk_count=len(all_chunks),
        section_count=len(sections),
        logic_extracted=logic_template is not None,
        logic_template=logic_template
    )
    manifest.upsert(new_manifest)
    log.info(f"  → {len(all_chunks)} chunks, {len(sections)} sections")
    return new_manifest


def upsert_chunks(chunks: List[Dict[str, Any]]):
    """Batch upsert chunks to Milvus."""
    if not chunks:
        return

    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    # Ensure collection is loaded
    client.load_collection(COLLECTION_NAME)

    # Delete old chunks for these source_ids
    source_ids = list(set(c["source_id"] for c in chunks))
    for sid in source_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'source_id == "{sid}"'
        )

    # Prepare insert data
    data = []
    for c in chunks:
        data.append({
            "text": c["text"],
            "source_id": c["source_id"],
            "section_title": c["section_title"],
            "section_start": c["section_start"],
            "section_end": c["section_end"],
            "chunk_id": c["chunk_id"],
            "dense_vector": c["dense_vector"],
            "sparse_vector": c["sparse_vector"],
            "logic_template": c.get("logic_template"),
            "logic_name": c.get("logic_name"),
            "transferable_domains": c.get("transferable_domains", []),
        })

    # Batch insert
    for i in range(0, len(data), BATCH_INSERT_SIZE):
        batch = data[i:i + BATCH_INSERT_SIZE]
        client.insert(collection_name=COLLECTION_NAME, data=batch)
    client.close()


def ensure_collection():
    """Create Milvus collection with hybrid indexes if not exists."""
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    client.drop_collection(COLLECTION_NAME)

    if client.has_collection(COLLECTION_NAME):
        log.info(f"Collection '{COLLECTION_NAME}' exists")
        client.close()
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source_id", DataType.VARCHAR, max_length=256)
    schema.add_field("section_title", DataType.VARCHAR, max_length=2048)
    schema.add_field("section_start", DataType.INT64)
    schema.add_field("section_end", DataType.INT64)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # Logic template fields for cross-domain analogy
    schema.add_field("logic_template", DataType.JSON)
    schema.add_field("logic_name", DataType.VARCHAR, max_length=512)
    schema.add_field("transferable_domains", DataType.ARRAY,element_type=DataType.VARCHAR, max_length=256, max_capacity=32)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2}
    )
    # Inverted index for transferable_domains filtering
    index_params.add_index(
        field_name="transferable_domains",
        index_type="INVERTED"
    )
    # JSON field index - REQUIRES json_cast_type
    index_params.add_index(
        field_name="logic_name",
        index_type="INVERTED"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    client.close()
    log.info(f"Created collection '{COLLECTION_NAME}' with hybrid indexes + logic template fields")


def main():
    log.info("Starting ingestion pipeline")

    # 1. Ensure collection exists
    ensure_collection()

    # 2. Initialize LLM for logic extraction
    global logic_extraction_chain
    llm = ChatOpenAI(
        base_url=LLM_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        temperature=0.1,
        timeout=LLM_TIMEOUT,
        max_retries=2
    )
    logic_extraction_chain = LOGIC_EXTRACT_PROMPT | llm | JsonOutputParser()
    log.info("Logic extraction chain initialized")

    # 2. Load manifest
    manifest = ManifestStore(MANIFEST_PATH)

    # 3. Discover files
    files = discover_files(DOCS_DIR)
    log.info(f"Discovered {len(files)} .txt files")

    # 3. Diff: find deleted files
    current_files = {str(f.relative_to(DOCS_DIR)) for f in files}
    deleted = manifest.all_keys() - current_files
    if deleted:
        log.info(f"Deleting {len(deleted)} removed files from Milvus")
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        for sid in deleted:
            client.delete(collection_name=COLLECTION_NAME, filter=f'source_id == "{sid}"')
            manifest.delete(sid)
        client.close()

    # 4. Process each file (parallelize embedding-heavy step)
    embedder = BGE_M3_Embedder()
    embeddings = BGE_M3_Embeddings()
    processed = 0
    for f in files:
        try:
            process_file(f, embedder, embeddings, manifest, logic_extraction_chain)
            processed += 1
        except Exception as e:
            log.error(f"Failed {f}: {e}")

    # 5. Save manifest
    manifest.save()
    log.info(f"Done. Processed {processed}/{len(files)} files. Manifest saved.")


if __name__ == "__main__":
    main()