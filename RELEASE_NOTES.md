# Release Notes

## v1.0.0

发布日期：2026-07-19

### 亮点（中文）

- 完整闭环：Chat + RAG + Agent（工具调用）+ 本地/远端模型路由
- 可部署：Docker Compose 一键启动，Nginx 统一入口，仅暴露 80 端口
- 可观测：请求级 request_id + 访问日志（含 duration_ms/user_id）+ Agent trace 落库
- 可复现：数据快照、版本化目录、评测流水线与脚本化压测/冒烟

### 主要能力

- Chat
  - `POST /chat` 支持非流式与 SSE 流式（`?stream=true`）
  - 支持多轮会话（`conversation_id`），并将消息持久化到 PostgreSQL
  - 支持模型路由：`model_source=remote|local`，可选 `model` 覆盖模型名（配合 vLLM LoRA adapter）
- RAG（检索增强问答）
  - 文档 ingestion：`POST /ingest` 支持 `.txt/.md/.py/.js`
  - Hybrid retriever：向量检索 + BM25 关键词检索，后接 rerank
  - `POST /search` 作为独立检索接口
  - 回答生成器会输出 citations（Sources）
- Agent
  - `POST /agent` 支持非流式与 SSE（meta/step/final/done）
  - ReAct 循环，支持工具注册表与工具选择
  - 内置工具：`search_docs`（检索）与 `execute_code`（Docker 沙箱执行 Python）
  - trace 落库到 `agent_traces`，并在前端 admin 页面展示
- Deployment & Ops
  - Nginx 路由分流（API → backend；页面/静态资源 → frontend），SSE 关闭 buffering
  - Redis 滑动窗口限流（默认 30 req/min per user；优先 `x-user-id`，否则按 client IP）
  - `GET /health`（backend + frontend），docker-compose healthcheck + restart policies
  - 后端 Docker 镜像多阶段构建（runtime 不携带 tests）
- Tooling
  - 端到端 smoke：`backend/scripts/smoke_e2e.py`（health → chat → ingest → search → agent）
  - 并发压测：`backend/scripts/load_test_chat.py`（P50/P95/P99 + JSON/MD/HTML 报告）

### Breaking / 注意事项

- 默认部署入口变更为 Nginx：对外访问 `http://localhost`（80 端口）
- 前端默认 `NEXT_PUBLIC_API_URL=http://localhost`（匹配 Nginx）；若用“本地开发模式”直连后端，需要改回 `http://localhost:8000`

### Release Notes (English)

- End-to-end: chat + RAG + tool-calling agent + remote/local model routing
- Deployable with docker-compose, single entrypoint via Nginx (port 80)
- Observable: request_id + access logs (duration_ms/user_id) + persisted agent traces
- Reproducible: scripted smoke/load tests and evaluation artifacts

### Tagging (optional, run by yourself)

```bash
git tag -a v1.0.0 -m "DevAssist v1.0.0"
git push origin v1.0.0
```
