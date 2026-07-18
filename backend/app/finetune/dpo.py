from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DPOTrainConfig:
    model_name_or_path: str
    dpo_pairs_path: Path
    output_dir: Path
    init_adapter_path: Path | None = None
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    warmup_ratio: float = 0.03
    seed: int = 42
    beta: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    trust_remote_code: bool = True
    report_to: tuple[str, ...] = ()
    run_name: str | None = None
    save_log_history: bool = True


def train_dpo(config: DPOTrainConfig) -> Path:
    """
    运行 LoRA DPO 训练（基于 TRL 的 DPOTrainer）。

    Args:
        config: 训练配置（模型、dpo_pairs 路径、输出目录与 LoRA/训练超参）。

    Returns:
        输出目录路径（包含 LoRA adapter 与 tokenizer 等文件）。

    Raises:
        ValueError: 配置参数不合法（例如 model_name_or_path 为空）。
        FileNotFoundError: dpo_pairs_path 不存在。
        Exception: Transformers/训练过程抛出的异常会原样向上抛出。

    Notes:
        - 训练数据使用 DPO JSONL schema：prompt/chosen/rejected。
        - 该函数会延迟导入 torch/transformers/peft/trl，避免未安装训练依赖时 import 失败。
        - 默认只训练 LoRA adapter，不会覆盖 base model 权重。
        - 如果提供 init_adapter_path，会在训练前先加载该 adapter 作为初始权重继续训练。
    """

    if not config.model_name_or_path.strip():
        raise ValueError("model_name_or_path must not be empty")
    if not config.dpo_pairs_path.exists():
        raise FileNotFoundError(str(config.dpo_pairs_path))
    if config.init_adapter_path is not None and not config.init_adapter_path.exists():
        raise FileNotFoundError(str(config.init_adapter_path))

    import torch  # type: ignore
    from datasets import load_dataset  # type: ignore
    from peft import LoraConfig, PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore
    from trl import DPOTrainer  # type: ignore

    torch.manual_seed(int(config.seed))

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )

    if config.init_adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(config.init_adapter_path), is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=int(config.lora_r),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    ds = load_dataset("json", data_files={"train": str(config.dpo_pairs_path)}, split="train")
    splitted = ds.train_test_split(test_size=0.02, seed=int(config.seed))
    train_ds = splitted["train"]
    eval_ds = splitted["test"]

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=int(config.per_device_train_batch_size),
        per_device_eval_batch_size=int(config.per_device_eval_batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        num_train_epochs=float(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        warmup_ratio=float(config.warmup_ratio),
        logging_steps=int(config.logging_steps),
        save_steps=int(config.save_steps),
        eval_steps=int(config.eval_steps),
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=list(config.report_to) if config.report_to else [],
        run_name=str(config.run_name) if config.run_name else None,
        seed=int(config.seed),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        beta=float(config.beta),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=int(config.max_seq_length),
        max_prompt_length=min(int(config.max_seq_length), 1024),
    )

    trainer.train()

    trainer.model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    if config.save_log_history:
        log_path = config.output_dir / "train_log_history.json"
        log_path.write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return config.output_dir
