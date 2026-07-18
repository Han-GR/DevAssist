from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SFTTrainConfig:
    model_name_or_path: str
    train_path: Path
    output_dir: Path
    eval_path: Path | None = None
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    warmup_ratio: float = 0.03
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    trust_remote_code: bool = True


def _build_training_text(*, tokenizer: Any, instruction: str, input_text: str, output_text: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    out = (output_text or "").strip()

    messages = [
        {"role": "system", "content": inst},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": out},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return "\n\n".join(
        [
            f"### Instruction:\n{inst}",
            f"### Input:\n{inp}",
            f"### Response:\n{out}",
        ]
    )


def train_sft(config: SFTTrainConfig) -> Path:
    """
    运行 LoRA SFT 训练（基于 Transformers + PEFT + TRL）。

    Args:
        config: 训练配置（模型、数据集路径、输出目录与 LoRA/训练超参）。

    Returns:
        输出目录路径（包含 LoRA adapter 与 tokenizer 等文件）。

    Raises:
        ValueError: 配置参数不合法（例如路径缺失、关键字段为空）。
        FileNotFoundError: train_path/eval_path 不存在。
        Exception: Transformers/训练过程抛出的异常会原样向上抛出。

    Notes:
        - 训练数据使用 Day61 定义的 SFT JSONL schema：instruction/input/output。
        - 该函数会延迟导入 torch/transformers/peft/trl，避免未安装训练依赖时 import 失败。
        - 默认只训练 LoRA adapter，不会覆盖 base model 权重。
    """

    if not config.model_name_or_path.strip():
        raise ValueError("model_name_or_path must not be empty")
    if not config.train_path.exists():
        raise FileNotFoundError(str(config.train_path))
    if config.eval_path is not None and not config.eval_path.exists():
        raise FileNotFoundError(str(config.eval_path))

    import torch  # type: ignore
    from datasets import load_dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore

    torch.manual_seed(int(config.seed))

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )

    lora = LoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    train_ds = load_dataset("json", data_files={"train": str(config.train_path)}, split="train")
    if config.eval_path is not None:
        eval_ds = load_dataset("json", data_files={"eval": str(config.eval_path)}, split="eval")
    else:
        splitted = train_ds.train_test_split(test_size=0.02, seed=int(config.seed))
        train_ds = splitted["train"]
        eval_ds = splitted["test"]

    def _formatting_func(example: dict[str, Any]) -> str:
        return _build_training_text(
            tokenizer=tokenizer,
            instruction=str(example.get("instruction") or ""),
            input_text=str(example.get("input") or ""),
            output_text=str(example.get("output") or ""),
        )

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
        report_to=[],
        seed=int(config.seed),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=_formatting_func,
        max_seq_length=int(config.max_seq_length),
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    return config.output_dir

