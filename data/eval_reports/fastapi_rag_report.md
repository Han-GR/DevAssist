# FastAPI RAG 评测与调参记录

## 1. 目标

- 跑通 FastAPI 文档知识库的离线评测
- 对 chunk_size / top_k / rerank 阈值做小范围调参，并给出一套“默认可用”的推荐配置

## 2. 本次评测的前置条件

### 2.1 启动依赖服务

```bash
docker compose up -d
```

### 2.2 Ingest FastAPI 官方文档（两套 chunk_size 对比）

说明：
- 评测数据集是中文问答，而 FastAPI 官方文档是英文，纯 token 重叠指标会偏“保守”；本次同时输出 embedding 余弦相似度版本的指标，便于跨语言对比。
- 本次为了控成本，只拉取并 ingest 了前 30 篇 markdown（足够覆盖数据集中的常见主题）。要做更严谨评测，可去掉 `--limit` 全量 ingest。

#### A) chunk_size=512, overlap=64（collection=fastapi_docs）

```bash
docker compose run --rm backend python scripts/ingest_fastapi_docs.py \
  --limit 30 \
  --collection fastapi_docs \
  --chunk-size 512 \
  --overlap 64
```

#### B) chunk_size=800, overlap=100（collection=fastapi_docs_cs800）

```bash
docker compose run --rm backend python scripts/ingest_fastapi_docs.py \
  --limit 30 \
  --collection fastapi_docs_cs800 \
  --chunk-size 800 \
  --overlap 100
```

## 3. 评测命令（离线，无 LLM）

说明：
- `--no-llm`：不生成答案，直接用数据集里的 `reference_answer` 作为 answer；但仍会真实走检索得到 contexts，因此可用于调检索参数。
- `--embedding-metrics`：额外计算 3 个 embedding 相似度指标（cosine），在中文问答 + 英文文档场景下更稳定。

### 3.1 基线（chunk_size=512）

```bash
docker compose run --rm backend python scripts/eval_runner.py \
  --dataset /data/datasets/fastapi_eval_50.jsonl \
  --no-llm \
  --top-k 5 \
  --candidate-multiplier 4 \
  --rerank-min-score 0.0 \
  --embedding-metrics \
  --output /data/eval_reports/fastapi_full_top5_cm4_thr0.json
```

### 3.2 调参：candidate_multiplier=6（chunk_size=512）

```bash
docker compose run --rm backend python scripts/eval_runner.py \
  --dataset /data/datasets/fastapi_eval_50.jsonl \
  --no-llm \
  --top-k 5 \
  --candidate-multiplier 6 \
  --rerank-min-score 0.0 \
  --embedding-metrics \
  --output /data/eval_reports/fastapi_full_top5_cm6_thr0.json
```

### 3.3 对比：chunk_size=800（collection=fastapi_docs_cs800）

```bash
docker compose run --rm backend python scripts/eval_runner.py \
  --dataset /data/datasets/fastapi_eval_50.jsonl \
  --no-llm \
  --collection fastapi_docs_cs800 \
  --top-k 5 \
  --candidate-multiplier 4 \
  --rerank-min-score 0.0 \
  --embedding-metrics \
  --output /data/eval_reports/fastapi_full_cs800_top5_cm4_thr0.json
```

## 4. 结果摘要（avg）

指标说明（embedding 版本）：
- emb_relevance：cos(question, answer)
- emb_faithfulness：cos(answer, contexts_joined)
- emb_context_recall：cos(reference_answer, contexts_joined)

| 配置 | collection | top_k | candidate_multiplier | rerank_min_score | avg emb_relevance | avg emb_faithfulness | avg emb_context_recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 基线（512） | fastapi_docs | 5 | 4 | 0.0 | 0.627 | 0.542 | 0.542 |
| 调参（512） | fastapi_docs | 5 | 6 | 0.0 | 0.627 | 0.543 | 0.543 |
| chunk 对比（800） | fastapi_docs_cs800 | 5 | 4 | 0.0 | 0.627 | 0.546 | 0.546 |

结论（以本次 30 篇文档样本为准）：
- chunk_size 从 512 提到 800 后，embedding 版本的 recall 有小幅提升（0.542 → 0.546）。
- candidate_multiplier 从 4 提到 6 有非常小的提升（0.542 → 0.543），但会增加检索候选规模与 rerank 成本。

## 5. 推荐参数（当前阶段）

- ingestion：chunk_size=800，overlap=100（建议先用单独 collection 验证，确认收益后再考虑调整默认值）
- retrieval/generation：
  - top_k=5（最终给 LLM 的上下文块数量，控制 prompt 长度）
  - candidate_multiplier=4（如你更偏向 recall，可提高到 6）
  - rerank_min_score=0.0（保持默认；并依赖“空结果 fallback”防止跨语言/关键词稀疏导致 contexts 为空）

## 6. 备注

- token 重叠版指标在“中文问答 + 英文文档”场景下会偏低或不稳定，优先参考 embedding 指标。
- Chroma telemetry 在当前环境会出现 `capture() takes 1 positional argument but 3 were given` 警告，不影响检索与评测结果。

