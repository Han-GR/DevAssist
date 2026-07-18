from __future__ import annotations

from typing import Any


def build_chat_messages(*, user_prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """
    构造 OpenAI-style messages（用于本地模型的 chat template）。

    Args:
        user_prompt: 用户输入内容。
        system_prompt: 可选系统指令；为空时不输出 system 消息。

    Returns:
        messages 列表，元素为 {"role": "...", "content": "..."}。

    Raises:
        ValueError: user_prompt 为空。

    Notes:
        - 该结构可用于 tokenizer.apply_chat_template（如果 tokenizer 支持）。
    """

    up = (user_prompt or "").strip()
    if not up:
        raise ValueError("user_prompt must not be empty")

    sp = (system_prompt or "").strip()
    messages: list[dict[str, str]] = []
    if sp:
        messages.append({"role": "system", "content": sp})
    messages.append({"role": "user", "content": up})
    return messages


def format_prompt_for_tokenizer(*, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """
    将 messages 格式化为模型可用的 prompt 字符串。

    Args:
        tokenizer: Transformers tokenizer（任意具备 apply_chat_template 的对象亦可）。
        messages: OpenAI-style messages。

    Returns:
        prompt 字符串。

    Raises:
        ValueError: messages 为空或结构不合法。

    Notes:
        - 优先使用 tokenizer.apply_chat_template（更贴近指令模型的真实格式）。
        - 如果 tokenizer 不支持 chat template，则退化为简单拼接格式（仅用于 sanity check）。
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
        role = str(m["role"]).strip()
        content = str(m["content"]).strip()
        parts.append(f"{role}:\n{content}")
    parts.append("assistant:\n")
    return "\n\n".join(parts)

