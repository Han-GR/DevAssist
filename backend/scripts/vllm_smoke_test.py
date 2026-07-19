from __future__ import annotations

import argparse
import json
from openai import OpenAI


def normalize_openai_base_url(raw: str) -> str:
    """
    规范化 OpenAI client 的 base_url。

    Args:
        raw: 用户输入的 base_url，允许为 http(s)://host:port 或以 /v1 结尾的完整路径。

    Returns:
        以 /v1 结尾的 base_url。
    """

    s = raw.rstrip("/")
    if s.endswith("/v1"):
        return s
    return f"{s}/v1"


def main() -> int:
    """
    vLLM OpenAI 兼容服务 smoke test（base / LoRA）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 网络请求失败或服务端返回错误时抛出。
    """

    parser = argparse.ArgumentParser(description="Smoke-test a vLLM OpenAI-compatible server (optionally with a LoRA adapter).")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="devassist-local")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=str, default="Explain SSE in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    base_url = normalize_openai_base_url(str(args.base_url))
    client = OpenAI(base_url=base_url, api_key=str(args.api_key))

    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f"models={json.dumps(model_ids, ensure_ascii=False)}")

    completion = client.chat.completions.create(
        model=str(args.model),
        messages=[{"role": "user", "content": str(args.prompt)}],
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
    )
    content = completion.choices[0].message.content or ""
    print(f"answer={content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

