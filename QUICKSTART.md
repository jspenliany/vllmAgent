# Quick Start Guide

## Prerequisites (Containers Running)

```bash
# Milvus (standalone)
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  -v /var/lib/milvus:/var/lib/milvus \
  milvusdb/milvus:v2.4.4 milvus run standalone

# bge-m3 (TEI / text-embeddings-inference)
docker run -d --name bge-m3 \
  -p 8000:80 \
  -v /data/models/bge-m3:/data \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id BAAI/bge-m3 --max-batch-size 32

# Gemma (Ollama)
docker run -d --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama:latest
# Then: docker exec ollama ollama pull gemma2:27b
```

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Configure Paths

Edit `ingest.py`:
```python
DOCS_DIR = Path("/path/to/your/400_txt_files")
```

## Run Ingestion (First Time / Periodic)

```bash
# Full ingestion
python ingest.py

# Incremental (only new/changed files)
python ingest.py
```

Output:
```
2024-01-15 10:30:00 INFO Starting ingestion pipeline
2024-01-15 10:30:01 INFO Created collection 'graphrag_chunks' with hybrid indexes
2024-01-15 10:30:01 INFO Loaded manifest: 0 files
2024-01-15 10:30:01 INFO Discovered 400 .txt files
2024-01-15 10:30:01 INFO Processing: doc_001.txt (45KB)
2024-01-15 10:30:15 INFO   → 23 chunks, 5 sections
...
2024-01-15 10:45:00 INFO Done. Processed 400/400 files. Manifest saved.
```

## Test Retrieval

```bash
# Interactive CLI
python retrieve.py

# Or use programmatically
from retrieve import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("What are the main causes of climate change?")
print(result["answer"])
print(result["citations"])
```

Example output:
```
📝 Answer:
Based on the retrieved documents, the main causes of climate change include:
- Greenhouse gas emissions from fossil fuel combustion [climate_report.txt] Executive Summary (chars 120-450)
- Deforestation reducing carbon sinks [forestry_2023.txt] Section 3.1 (chars 2000-2300)
- Industrial processes releasing methane [ipcc_summary.txt] Key Findings (chars 500-800)

📚 Citations:
  1. [climate_report.txt] Executive Summary (chars 120-450) - score: 0.8923
  2. [forestry_2023.txt] Section 3.1 (chars 2000-2300) - score: 0.8412
  3. [ipcc_summary.txt] Key Findings (chars 500-800) - score: 0.8105
```

## Automate Periodic Ingestion

```bash
# systemd timer (run daily at 2 AM)
# /etc/systemd/system/rag-ingest.service
[Unit]
Description=RAG Ingestion
After=network.target docker.service

[Service]
Type=oneshot
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 ingest.py
User=youruser

# /etc/systemd/system/rag-ingest.timer
[Unit]
Description=Daily RAG Ingestion

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target

sudo systemctl daemon-reload
sudo systemctl enable --now rag-ingest.timer
```

## Tuning Knobs

| Parameter | File | Default | When to Adjust |
|-----------|------|---------|----------------|
| `breakpoint_threshold_amount` | ingest.py | 70 | Chunks too large/small |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | retrieve.py | 0.7 / 0.3 | Keyword vs semantic balance |
| `TOP_K_PER_SUBQUERY` | retrieve.py | 10 | Recall vs latency |
| `TOP_K_FINAL` | retrieve.py | 8 | Context window vs noise |
| `LLM_MODEL` | retrieve.py | gemma2:27b | Model capability |

## Add Cross-Encoder Reranker (Phase 2)

```python
# In retrieve.py, after multi_query_search:
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

pairs = [(question, c.text) for c in chunks]
scores = reranker.predict(pairs)
for c, s in zip(chunks, scores):
    c.rerank_score = float(s)
chunks.sort(key=lambda c: c.rerank_score, reverse=True)
chunks = chunks[:TOP_K_FINAL]
```

## Monitoring

```bash
# Milvus stats
curl http://localhost:9091/metrics | grep milvus

# Collection stats
python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
print(c.get_collection_stats('graphrag_chunks'))
"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` Milvus | Check container: `docker logs milvus` |
| `404 /v1/embeddings` | bge-m3 not ready: `docker logs bge-m3` |
| OOM on embed | Reduce `BATCH_EMBED_SIZE` in ingest.py |
| Slow ingestion | Increase `ThreadPoolExecutor` workers for file I/O |
| Poor retrieval | Adjust `DENSE_WEIGHT`/`SPARSE_WEIGHT`, add reranker |