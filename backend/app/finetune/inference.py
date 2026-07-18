from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42
    trust_remote_code: bool = True


def build_chat_messages(*, instruction: str, input_text: str) -> list[dict[str, str]]:
    """
    构造用于指令模型推理的 messages（OpenAI-style）。

    Args:
        instruction: 系统指令（system prompt）。
        input_text: 用户输入（user prompt）。

    Returns:
        messages 列表。

    Raises:
        ValueError: instruction 或 input_text 为空。

    Notes:
        - 下游可通过 tokenizer.apply_chat_template 生成最终 prompt。
    """

    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")
    if not inp:
        raise ValueError("input_text must not be empty")

    return [
        {"role": "system", "content": inst},
        {"role": "user", "content": inp},
    ]


def format_prompt(*, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """
    将 messages 格式化为可用于 Transformers 的 prompt 字符串。

    Args:
        tokenizer: Transformers tokenizer（或兼容对象）。
        messages: OpenAI-style messages。

    Returns:
        prompt 字符串。

    Raises:
        ValueError: messages 为空或结构不合法。

    Notes:
        - 优先使用 tokenizer.apply_chat_template（更贴近模型真实指令格式）。
        - 不支持 chat template 时退化为简单拼接（仅用于 sanity check）。
    """

    if not messages:
        raise ValueError("messages must not be empty")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("invalid message object")

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: list[str] = []
    for m in messages:
        parts.append(f"{m['role']}:\n{m['content']}".strip())
    parts.append("assistant:\n")
    return "\n\n".join(parts)


def generate_text(*, model: Any, tokenizer: Any, prompt: str, config: InferenceConfig) -> str:
    """
    使用 Transformers model.generate 生成文本。

    Args:
        model: AutoModelForCausalLM（或兼容对象）。
        tokenizer: 对应 tokenizer。
        prompt: 已格式化的 prompt 字符串。
        config: 推理配置。

    Returns:
        生成的完整文本（decode 后的字符串，skip_special_tokens=True）。

    Raises:
        Exception: torch/transformers 在推理阶段抛出的异常会原样向上抛出。

    Notes:
        - 该函数会延迟导入 torch，避免在未安装训练依赖的环境里 import 失败。
    """

    import torch  # type: ignore

    torch.manual_seed(int(config.seed))

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(config.max_new_tokens),
        do_sample=float(config.temperature) > 0,
        temperature=float(config.temperature),
        top_p=float(config.top_p),
    )
    return str(tokenizer.decode(output_ids[0], skip_special_tokens=True))


def load_base_model_and_tokenizer(*, model_name_or_path: str, config: InferenceConfig) -> tuple[Any, Any]:
    """
    加载 base 模型与 tokenizer。

    Args:
        model_name_or_path: HuggingFace 模型名或本地路径。
        config: 推理配置（主要用 trust_remote_code）。

    Returns:
        (model, tokenizer)。

    Raises:
        ValueError: model_name_or_path 为空。
        Exception: Transformers 加载失败时抛出。
    """

    name = (model_name_or_path or "").strip()
    if not name:
        raise ValueError("model_name_or_path must not be empty")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    return model, tokenizer


def load_lora_model_and_tokenizer(
    *,
    base_model_name_or_path: str,
    adapter_path: str,
    config: InferenceConfig,
) -> tuple[Any, Any]:
    """
    加载 base 模型并挂载 LoRA adapter。

    Args:
        base_model_name_or_path: base 模型名或本地路径。
        adapter_path: LoRA adapter 目录路径（train_sft 的 output_dir）。
        config: 推理配置（主要用 trust_remote_code）。

    Returns:
        (model, tokenizer)。

    Raises:
        ValueError: base_model_name_or_path 或 adapter_path 为空。
        Exception: 加载失败时抛出。

    Notes:
        - 这里使用 PeftModel.from_pretrained 将 adapter 挂载到 base model 上。
    """

    base = (base_model_name_or_path or "").strip()
    adapter = (adapter_path or "").strip()
    if not base:
        raise ValueError("base_model_name_or_path must not be empty")
    if not adapter:
        raise ValueError("adapter_path must not be empty")

    model, tokenizer = load_base_model_and_tokenizer(model_name_or_path=base, config=config)

    from peft import PeftModel  # type: ignore

    model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer

