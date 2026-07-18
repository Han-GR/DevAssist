# DevAssist Fine-tuned Model Card

## 基本信息 / Overview

- Model name: `{base}-{variant}-{date}-{commit}`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adaptation: LoRA (PEFT)
- Primary language: Chinese + English
- Repository: DevAssist
- Maintainers: (fill)
- License: (fill)

## 预期用途 / Intended Use

适用场景：

- 编程问答（Python/FastAPI/前后端工程实践）
- 代码审查与重构建议（偏工程可执行）
- 微调/评测脚本写作与调参建议（LoRA/SFT/DPO）

不适用场景：

- 需要强事实性、实时性或权威引用的回答（例如法律/医疗/财务）
- 输出需满足严格合规审计的生产决策（需要人类复核）

## 输入输出格式 / I/O Format

- Chat-style input: system + user messages
- SFT training schema: JSONL lines with `instruction`, `input`, `output`
- Optional metadata: `meta` is allowed and should not affect training logic

## 训练数据 / Training Data

数据来源（建议在发布时填写）：

- Chat export: (fill)
- Human-curated: (fill)
- Synthetic: (fill)

清洗与过滤：

- Secrets/credentials redaction (e.g., `sk-`, `AKIA`, `Bearer `)
- Deduplication + length filtering + heuristic quality score
- Dataset snapshots supported via `scripts/snapshot_datasets.py`

建议记录：

- `train_path`: (fill)
- `eval_path`: (fill)
- `evalset_path` (task-oriented): `data/datasets/finetune_eval.sample.jsonl`
- `dataset_snapshot_manifest`: (fill)

## 训练配置 / Training Configuration

### SFT (LoRA)

关键超参（示例，请用实际值替换）：

- `lora_r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `target_modules`: `["q_proj", "v_proj"]`
- `learning_rate`: 2e-4
- `epochs`: 3
- `max_seq_length`: 2048
- `batch_size`: 1
- `grad_accum`: 8
- `seed`: 42

训练脚本：

- `scripts/train_sft.py` / `scripts/train_sft_final.py`

### DPO（如适用）/ DPO (Optional)

- `scripts/train_dpo.py`
- init adapter from SFT: supported

## 评测 / Evaluation

### Rubric-based (Heuristic)

Eval set:

- `data/datasets/finetune_eval.sample.jsonl`

Metrics (fill the final numbers):

- `pass_rate.all`: (fill)
- `avg_include_rate.all`: (fill)
- `violation_rate.all`: (fill)

Reports:

- `data/eval_reports/.../finetune_eval_pipeline_report.md`

### LLM-as-Judge (Optional)

If enabled, record:

- judge provider/model: (fill)
- score summary: (fill)
- raw outputs: `data/eval_reports/judge_report.*.json`

## 局限性 / Limitations

- Heuristic rubric evaluation is substring-based and may not reflect true correctness
- Domain coverage depends on training data; out-of-domain tasks may degrade
- Multi-step tool-calling reliability is not guaranteed without explicit agent scaffolding
- May hallucinate; always verify code and claims

## 安全与隐私 / Safety & Privacy

- Do not include secrets in training data
- Model outputs may accidentally reproduce memorized sensitive content if the dataset is compromised
- For production usage, combine with:
  - input sanitization
  - output filtering
  - human review for sensitive domains

## 版本与产物 / Versioning & Artifacts

Naming convention:

- `{base}-{variant}-{date}-{commit}` (directory-safe slug)

Artifacts to keep:

- LoRA adapter directory (weights + tokenizer)
- run plan JSON (e.g., `data/datasets/runs/sft_final.plan.json`)
- evaluation reports directory
- dataset snapshot manifest

## 变更记录 / Changelog

Use an append-only record (JSONL recommended) for each run:

- date, commit, dataset snapshot, hyperparams, artifacts, headline metrics
