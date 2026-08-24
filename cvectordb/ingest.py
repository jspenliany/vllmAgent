#!/usr/bin/env python3
"""
Batch ingestion pipeline for 400 scattered .txt documents.
Loads → semantic chunks → embeds (dense+sparse) → upserts to Milvus.
Run periodically (cron/systemd timer) for incremental updates.
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
from langchain_huggingface import HuggingFaceEmbeddings

# ============== CONFIG ==============
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "graphrag_chunks"
DIM = 1024
EMBED_URL = "http://localhost:8000/v1/embeddings"  # bge-m3 service
BATCH_EMBED_SIZE = 32
BATCH_INSERT_SIZE = 128
DOCS_DIR = Path("/data/txt_corpus")  # <-- change to your 400 .txt files
MANIFEST_PATH = Path("./ingest_manifest.json")
LOG_LEVEL = logging.INFO
# ====================================

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class FileManifest:
    path: str
    hash: str
    size: int
    mtime: float
    chunk_count: int = 0
    section_count: int = 0


class BGE_M3_Embedder:
    """Calls bge-m3 container for dense + sparse embeddings."""

    def __init__(self, url: str = EMBED_URL, batch_size: int = BATCH_EMBED_SIZE):
        self.url = url
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
                json={"input": batch, "model": "bge-m3"},
                timeout=60
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


def semantic_chunk_section(section_text: str, embedder: BGE_M3_Embedder) -> List[Dict[str, Any]]:
    """
    Use LangChain SemanticChunker with bge-m3 embeddings.
    Returns list of chunks within the section.
    """
    # Use a local embedding model for chunking decisions (fast, no API call)
    # We'll use a small fast model just for boundary detection
    chunker = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,  # 70th percentile similarity = boundary
        min_chunk_size=100,
    )
    lc_docs = chunker.create_documents([section_text])
    chunks = []
    for doc in lc_docs:
        # Find char offsets in section_text (approximate)
        start = section_text.find(doc.page_content[:50])
        if start == -1:
            start = 0
        end = start + len(doc.page_content)
        chunks.append({
            "text": doc.page_content,
            "start_char": start,
            "end_char": end
        })
    return chunks


def process_file(
    file_path: Path,
    embedder: BGE_M3_Embedder,
    manifest: ManifestStore
) -> Optional[FileManifest]:
    """Process a single .txt file: load → section → semantic chunk → embed."""
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

    all_chunks = []
    for sec_idx, sec in enumerate(sections):
        # Semantic chunk within section
        sec_chunks = semantic_chunk_section(sec["text"], embedder)
        for chunk_idx, chunk in enumerate(sec_chunks):
            # Global char offsets in original document
            global_start = sec["start_char"] + chunk["start_char"]
            global_end = sec["start_char"] + chunk["end_char"]
            all_chunks.append({
                "text": chunk["text"],
                "source_id": rel_path,
                "section_title": sec["title"],
                "section_start": global_start,
                "section_end": global_end,
                "chunk_id": chunk_idx,
                "section_idx": sec_idx
            })

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
        section_count=len(sections)
    )
    manifest.upsert(new_manifest)
    log.info(f"  → {len(all_chunks)} chunks, {len(sections)} sections")
    return new_manifest


def upsert_chunks(chunks: List[Dict[str, Any]]):
    """Batch upsert chunks to Milvus."""
    if not chunks:
        return

    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
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
        })

    # Batch insert
    for i in range(0, len(data), BATCH_INSERT_SIZE):
        batch = data[i:i + BATCH_INSERT_SIZE]
        client.insert(collection_name=COLLECTION_NAME, data=batch)
    client.close()


def ensure_collection():
    """Create Milvus collection with hybrid indexes if not exists."""
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if client.has_collection(COLLECTION_NAME):
        log.info(f"Collection '{COLLECTION_NAME}' exists")
        client.close()
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source_id", DataType.VARCHAR, max_length=256)
    schema.add_field("section_title", DataType.VARCHAR, max_length=512)
    schema.add_field("section_start", DataType.INT64)
    schema.add_field("section_end", DataType.INT64)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

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

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    client.close()
    log.info(f"Created collection '{COLLECTION_NAME}' with hybrid indexes")


def main():
    log.info("Starting ingestion pipeline")

    # 1. Ensure collection exists
    ensure_collection()

    # 2. Load manifest
    manifest = ManifestStore(MANIFEST_PATH)

    # 3. Discover files
    files = discover_files(DOCS_DIR)
    log.info(f"Discovered {len(files)} .txt files")

    # 4. Diff: find deleted files
    current_files = {str(f.relative_to(DOCS_DIR)) for f in files}
    deleted = manifest.all_keys() - current_files
    if deleted:
        log.info(f"Deleting {len(deleted)} removed files from Milvus")
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        for sid in deleted:
            client.delete(collection_name=COLLECTION_NAME, filter=f'source_id == "{sid}"')
            manifest.delete(sid)
        client.close()

    # 5. Process each file (parallelize embedding-heavy step)
    embedder = BGE_M3_Embedder()
    processed = 0
    for f in files:
        try:
            process_file(f, embedder, manifest)
            processed += 1
        except Exception as e:
            log.error(f"Failed {f}: {e}")

    # 6. Save manifest
    manifest.save()
    log.info(f"Done. Processed {processed}/{len(files)} files. Manifest saved.")


if __name__ == "__main__":
    main()