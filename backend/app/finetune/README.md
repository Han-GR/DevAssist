# Fine-tuning Data Specification (SFT)

This document defines the **Supervised Fine-Tuning (SFT)** dataset format used by DevAssist.

## Goals

- A minimal, stable JSONL schema that can be produced from chat logs and human edits.
- Deterministic formatting to support cleaning, deduplication, and evaluation splits.
- Privacy-safe by default (no secrets, no sensitive user content).

## File Format

- Encoding: UTF-8
- Container: JSON Lines (one JSON object per line)
- Recommended files (repo root):
  - `data/datasets/sft_train.jsonl`
  - `data/datasets/sft_eval.jsonl`
- Git policy:
  - Large datasets are ignored by default (`data/datasets/*.jsonl`).
  - Keep a tiny example file tracked, e.g. `data/datasets/sft.sample.jsonl`.

## SFT Schema (JSON object per line)

Required fields:

```json
{
  "instruction": "You are a Python expert. Answer the user's question precisely.",
  "input": "How do I stream responses from FastAPI?",
  "output": "Use Server-Sent Events (SSE) with a generator that yields 'data:' lines..."
}
```

Optional fields:

```json
{
  "instruction": "You are a senior engineer. Give a direct answer, then a minimal example.",
  "input": "Explain ReAct tool calling briefly.",
  "output": "ReAct alternates between reasoning and actions...",
  "meta": {
    "id": "c3f9b1b1-2c9d-4d70-a43b-6b6c9b64a4a2",
    "source": "chat_export",
    "conversation_id": "6a50aabddf5096dc7fa8b024",
    "message_ids": ["..."],
    "tags": ["agent", "react", "tools"],
    "created_at": "2026-07-13T08:22:11Z"
  }
}
```

### Field Semantics

- `instruction` (string, required)
  - System-level behavior constraints. Keep it short and consistent.
  - Prefer imperatives: "Answer...", "Use...", "Return...".
- `input` (string, required)
  - The user request or task description.
  - May include code blocks.
  - Use `""` if your dataset needs an instruction-only sample, but keep it rare.
- `output` (string, required)
  - The target answer (final response).
  - Must not include secrets or access tokens.
- `meta` (object, optional)
  - Non-training metadata for filtering, debugging, attribution, and splits.
  - Training code should ignore unknown fields.

## Normalization Rules

- Trim trailing whitespace on each field; keep internal whitespace intact.
- Newlines are allowed inside strings.
- Prefer Markdown in `output` when it improves readability.
- Avoid overly long samples:
  - Recommended soft limits:
    - `instruction` <= 512 chars
    - `input` <= 8,000 chars
    - `output` <= 12,000 chars
  - If exceeded, either summarize or split into multiple samples.

## Safety & Privacy Rules (Must)

- Remove:
  - API keys, tokens, cookies, credentials
  - private URLs with embedded credentials
  - personally identifying information (PII) if present in logs
- If a sample cannot be safely sanitized, discard it.

## Validation Checklist

Before training, ensure:

- Every line is valid JSON.
- Required keys exist: `instruction`, `input`, `output`.
- All required fields are strings.
- No empty `output`.
- No obvious secrets (`sk-`, `AKIA`, `Bearer `, etc.).

## Evaluation Dataset (Fine-tuning)

In addition to SFT train/eval splits, keep a **task-oriented evaluation set** to track improvements over time.

Recommended file:

- `data/datasets/finetune_eval.sample.jsonl` (tracked)

Format (JSONL, one object per line):

```json
{
  "id": "eval-0001",
  "category": "normal",
  "instruction": "You are a senior software engineer. Answer concisely and accurately.",
  "input": "FastAPI 里怎么写一个带请求体校验的 POST 接口？给最小可运行例子。",
  "rubric": {
    "must_include": ["FastAPI", "Pydantic", "POST", "uvicorn"],
    "must_not_include": ["sk-", "AKIA"],
    "notes": "Prefer a minimal runnable code example."
  }
}
```

Notes:

- `category`: `normal` / `edge` / `adversarial`
- `rubric.must_include`: expected key points (lightweight)
- This eval set is designed for rule-based checks and later LLM-as-judge evaluation.

## Notes

- DPO (preference) data will be specified separately when we reach the DPO stage.
