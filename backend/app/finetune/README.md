# Fine-tuning Data Specification (SFT)

This document defines the **Supervised Fine-Tuning (SFT)** dataset format used by DevAssist.

## Goals

- A minimal, stable JSONL schema that can be produced from chat logs and human edits.
- Deterministic formatting to support cleaning, deduplication, and evaluation splits.
- Privacy-safe by default (no secrets, no sensitive user content).

## Reproduction Guide

This is a pragmatic recipe to reproduce a full fine-tuning iteration end-to-end:

- data preparation (export/clean/split)
- SFT training (quick run → sweep → final run)
- offline evaluation (rubric + optional judge)
- DPO alignment and three-way comparison

Notes:

- Use `python3` explicitly (some environments do not provide a `python` alias).
- Training is expected to run on Linux + NVIDIA GPU; macOS/CPU is only for sanity checks.

## Model Card

Before publishing or comparing runs, update the model card with the chosen dataset snapshot, hyperparameters, and headline metrics:

- [MODEL_CARD.md](file:///Users/hanhan/Projects/PersonalOpenSource/Python/DevAssist/backend/app/finetune/MODEL_CARD.md)

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

## Training Environment Setup

DevAssist keeps runtime dependencies and training dependencies separated.

Install training dependencies (recommended to use `uv`):

```bash
cd backend
uv pip install -r requirements-finetune.txt
```

Quick environment check:

```bash
python3 scripts/check_training_env.py
```

Notes:

- Local macOS can run CPU-only for quick sanity checks, but real fine-tuning is recommended on Linux + NVIDIA GPU.
- If `cuda_available=false`, you can still run small CPU tests, but training speed will be very slow.

## Baseline Inference Check

Before training, run a **baseline inference** to verify:

- model weights can be loaded
- tokenizer works
- generation works end-to-end

Recommended command:

```bash
cd backend
python3 scripts/baseline_infer.py --model Qwen/Qwen2.5-7B-Instruct --prompt "Explain what SSE is in one paragraph."
```

Tips:

- CPU-only runs are for sanity checks only.
- For GPU runs, ensure CUDA is available and `torch` is installed with CUDA support.

## SFT Training (LoRA)

Run SFT training:

```bash
cd backend
python3 scripts/train_sft.py --model Qwen/Qwen2.5-7B-Instruct --train data/datasets/sft_train.jsonl --output data/models/qwen2.5-7b-lora
```

After training, you can run cleaning first and retrain if needed:

```bash
cd backend
python3 scripts/clean_sft.py --input data/datasets/sft_train.jsonl --output data/datasets/sft_train.cleaned.jsonl
```

## Initial Training Run (500 samples)

To get a quick signal, run a small experiment on 500 samples for 3 epochs:

```bash
cd backend
python3 scripts/run_sft_500.py --train data/datasets/sft_train.cleaned.jsonl --epochs 3 --output data/models/qwen2.5-7b-lora-500
```

Logging/monitoring:

- Use Transformers `report_to` to enable logging backends, e.g.:

```bash
cd backend
python3 scripts/run_sft_500.py --report-to wandb --run-name sft-500
```

## Hyperparameter Tuning (LoRA r / learning rate)

When comparing LoRA settings, keep everything else fixed (dataset, epochs, batch size, prompt formatting), and only change:

- `lora_r`
- `learning_rate`

Recommended sweep script (defaults to **dry-run**, prints commands only):

```bash
cd backend
python3 scripts/sweep_sft_lora.py --base-model Qwen/Qwen2.5-7B-Instruct
```

Run the sweep with real execution (GPU environment recommended):

```bash
cd backend
python3 scripts/sweep_sft_lora.py --execute --lora-rs 8,16,32 --lrs 5e-5,1e-4,2e-4
```

Outputs:

- A manifest JSONL file (append-only):
  - `data/datasets/sweeps/sft_lora_sweep.manifest.jsonl`
- Per-run evaluation reports:
  - `data/eval_reports/sweeps/.../finetune_eval_pipeline_report.md`

How to pick a "best" config (pragmatic):

- Primary metric: rubric `pass_rate` on `scope=all`
- Tie-breakers: higher `avg_include_rate`, lower `violation_rate`
- Always sanity-check a few samples manually (rubric is substring-based)

## Final Training Run (Chosen Best Config)

After you decide the best `(lora_r, learning_rate)` from the sweep, run a single final training + evaluation.

Dry-run (prints the exact training/eval commands and writes a run plan JSON):

```bash
cd backend
python3 scripts/train_sft_final.py --tag final --lora-r 16 --lr 2e-4
```

Execute (recommended on GPU):

```bash
cd backend
python3 scripts/train_sft_final.py --execute --tag final --lora-r 16 --lr 2e-4
```

Outputs:

- run plan (JSON):
  - `data/datasets/runs/sft_final.plan.json`
- model output directory:
  - `data/models/{base}-{variant}-{date}-{commit}-final-r{r}-a{alpha}-lr{lr}`
- evaluation reports:
  - `data/eval_reports/final/.../finetune_eval_pipeline_report.md`

## Key Config Reference

You can reproduce most runs by editing only these “knobs” (and keeping everything else fixed).

Scripts:

- `scripts/train_sft.py`
  - LoRA: `--lora-r`, `--lora-alpha`, `--lora-dropout`
  - Training: `--epochs`, `--lr`, `--max-seq-len`, `--batch-size`, `--grad-accum`, `--seed`
  - Output: `--output`, `--versioned-output` (optional)
- `scripts/sweep_sft_lora.py`
  - Grid: `--lora-rs`, `--lrs`
  - Safety: defaults to dry-run; add `--execute` to actually run
  - Manifest: `--manifest` (append-only JSONL)
- `scripts/train_sft_final.py`
  - “Chosen best” single run: `--lora-r`, `--lr`, `--tag`, `--execute`
  - Run plan: `--plan-json`
- `scripts/finetune_eval_runner.py`
  - Rubric eval: add `--enable-rubric`
  - Judge eval: add `--enable-judge` (requires LLM config)

## Notes

- DPO (preference) data will be specified separately when we reach the DPO stage.

## DPO Dataset (Preference Pairs)

Recommended file:

- `data/datasets/dpo_pairs.jsonl`

Format (JSONL, one object per line):

```json
{
  "prompt": "system:\\n...\\n\\nuser:\\n...",
  "chosen": "better answer",
  "rejected": "worse answer",
  "meta": {
    "case_id": "eval-0001",
    "category": "normal",
    "reason": "rubric"
  }
}
```

Generate pairs via LLM (best-effort, based on `finetune_eval` rubric):

```bash
cd backend
python3 scripts/generate_dpo_pairs.py --evalset data/datasets/finetune_eval.sample.jsonl --output data/datasets/dpo_pairs.jsonl --count 300
```

## DPO Training (LoRA)

Run DPO training:

```bash
cd backend
python3 scripts/train_dpo.py --model Qwen/Qwen2.5-7B-Instruct --pairs data/datasets/dpo_pairs.jsonl --output data/models/qwen2.5-7b-dpo-lora
```

Continue DPO from an existing SFT LoRA adapter:

```bash
cd backend
python3 scripts/train_dpo.py --model Qwen/Qwen2.5-7B-Instruct --init-adapter data/models/qwen2.5-7b-lora --pairs data/datasets/dpo_pairs.jsonl --output data/models/qwen2.5-7b-dpo-lora
```

## Three-way Comparison (Base vs SFT vs DPO)

Run a three-way comparison on the finetune evalset:

```bash
cd backend
python3 scripts/eval_base_sft_dpo.py --base-model Qwen/Qwen2.5-7B-Instruct --sft-adapter data/models/qwen2.5-7b-lora --dpo-adapter data/models/qwen2.5-7b-dpo-lora
```

## LLM-as-Judge Evaluation

This evaluation generates answers using a target model (base or LoRA), then asks a separate judge model (via `LLMClient`) to score each answer.

Evaluate the base model:

```bash
cd backend
python3 scripts/judge_eval.py --base-model Qwen/Qwen2.5-7B-Instruct --limit 50 --output-json data/eval_reports/judge_report.base.json
```

Evaluate a LoRA adapter:

```bash
cd backend
python3 scripts/judge_eval.py --base-model Qwen/Qwen2.5-7B-Instruct --adapter data/models/qwen2.5-7b-lora --limit 50 --output-json data/eval_reports/judge_report.lora.json
```

Use a specific judge provider/model:

```bash
cd backend
python3 scripts/judge_eval.py --judge-provider deepseek --judge-model deepseek-chat --limit 50
```

## Dataset Snapshots

For reproducibility, DevAssist supports a lightweight snapshot workflow:

- copy selected dataset files into a snapshot directory
- write a `manifest.json` that includes sha256 + non-empty line counts

Create a snapshot:

```bash
cd backend
python3 scripts/snapshot_datasets.py --label pre-clean
```

By default, snapshots are stored under:

- `data/datasets/snapshots/{YYYYMMDD}-{commit}/`

Git policy:

- snapshot contents are ignored by default to avoid committing large datasets
- `manifest.json` is allowed (small, useful for audit)

## Finetune Evaluation Pipeline

Run multiple finetune evaluations and generate a single summary report (best-effort):

```bash
cd backend
python3 scripts/finetune_eval_runner.py --sft-adapter data/models/qwen2.5-7b-lora --dpo-adapter data/models/qwen2.5-7b-dpo-lora --limit 50
```

The pipeline produces:

- `data/eval_reports/finetune_sft_vs_base_report.md`
- `data/eval_reports/finetune_three_way_report.md`
- `data/eval_reports/finetune_eval_pipeline_report.md`

## vLLM Serving (GPU)

DevAssist supports serving a base model (and optionally a LoRA adapter) via vLLM's OpenAI-compatible server.

Install vLLM dependencies (keep them separated from runtime and training deps):

```bash
cd backend
uv pip install -r requirements-vllm.txt
```

Serve the base model only:

```bash
cd backend
python3 scripts/serve_vllm_lora.py --base-model Qwen/Qwen2.5-7B-Instruct --api-key devassist-local
```

Serve the base model + one LoRA adapter:

```bash
cd backend
python3 scripts/serve_vllm_lora.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-name devassist-lora \
  --lora-path data/models/<your-lora-adapter-dir> \
  --api-key devassist-local
```

Smoke test (base model):

```bash
cd backend
python3 scripts/vllm_smoke_test.py --base-url http://localhost:8000/v1 --api-key devassist-local --model Qwen/Qwen2.5-7B-Instruct
```

Smoke test (LoRA adapter):

```bash
cd backend
python3 scripts/vllm_smoke_test.py --base-url http://localhost:8000/v1 --api-key devassist-local --model devassist-lora
```

Notes:

- The LoRA adapter is exposed as a standalone `model` id (e.g. `devassist-lora`) and can be selected via the OpenAI `model` parameter.
- This is expected to run on Linux + NVIDIA GPU. macOS/CPU is only for reading code and dry-run scripts.

## vLLM Benchmark

Benchmark the OpenAI-compatible `/v1/chat/completions` endpoint with concurrency and basic latency percentiles.

Example (base model):

```bash
cd backend
python3 scripts/bench_vllm.py \
  --base-url http://localhost:8000/v1 \
  --api-key devassist-local \
  --model Qwen/Qwen2.5-7B-Instruct \
  --requests 50 \
  --concurrency 5 \
  --output-json data/eval_reports/vllm_bench.base.json \
  --output-md data/eval_reports/vllm_bench.base.md
```

Example (LoRA adapter):

```bash
cd backend
python3 scripts/bench_vllm.py \
  --base-url http://localhost:8000/v1 \
  --api-key devassist-local \
  --model devassist-lora \
  --requests 50 \
  --concurrency 5 \
  --output-json data/eval_reports/vllm_bench.lora.json \
  --output-md data/eval_reports/vllm_bench.lora.md
```

The JSON output includes:

- latency_ms: avg/stdev/p50/p95/p99/min/max
- throughput: req_per_s / completion_tokens_per_s / total_tokens_per_s
- tokens: summed prompt/completion/total tokens (from OpenAI usage fields)

## Results Recording Template

To compare multiple runs reliably, keep an explicit record per run (append-only JSONL is recommended).

Suggested fields:

- `base_model`, `dataset_snapshot` (or file sha256), `hyperparams`
- `output_dir` and `report_paths`
- summarized metrics for quick filtering (rubric + judge)

Example (JSON object):

```json
{
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "run_tag": "final-r16-a32-lr2e-4",
  "dataset": {
    "train_path": "data/datasets/sft_train.cleaned.jsonl",
    "eval_path": "data/datasets/sft_eval.jsonl",
    "evalset_path": "data/datasets/finetune_eval.sample.jsonl",
    "snapshot": "data/datasets/snapshots/20260718-unknown/manifest.json"
  },
  "hyperparams": {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 0.0002,
    "epochs": 3,
    "max_seq_length": 2048,
    "batch_size": 1,
    "grad_accum": 8,
    "seed": 42
  },
  "artifacts": {
    "output_dir": "data/models/qwen2.5-7b-lora-20260718-unknown-final-r16-a32-lr2e-4",
    "reports": [
      "data/eval_reports/final/.../finetune_eval_pipeline_report.md",
      "data/eval_reports/final/.../finetune_three_way_report.md"
    ]
  },
  "metrics": {
    "rubric": {
      "pass_rate.all": 0.72,
      "avg_include_rate.all": 0.81,
      "violation_rate.all": 0.01
    }
  }
}
```
