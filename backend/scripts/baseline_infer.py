from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.baseline import build_chat_messages, format_prompt_for_tokenizer


def _load_transformers():
    """
    延迟导入 Transformers 依赖。

    Args:
        无。

    Returns:
        (torch, AutoTokenizer, AutoModelForCausalLM)。

    Raises:
        ImportError: 未安装 torch/transformers 时抛出。
    """

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return torch, AutoTokenizer, AutoModelForCausalLM


def main() -> int:
    """
    本地模型 baseline 推理脚本（训练前 sanity check）。

    Args:
        无（从命令行解析）。

    Returns:
        进程退出码。

    Raises:
        Exception: 模型加载或推理失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Baseline inference for local instruct model (sanity check).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--system",
        type=str,
        default="You are a senior software engineer. Answer concisely and accurately.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch, AutoTokenizer, AutoModelForCausalLM = _load_transformers()
    torch.manual_seed(int(args.seed))

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    load_ms = int((time.perf_counter() - t0) * 1000)

    messages = build_chat_messages(user_prompt=args.prompt, system_prompt=args.system)
    prompt = format_prompt_for_tokenizer(tokenizer=tokenizer, messages=messages)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    t1 = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=float(args.temperature) > 0,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    gen_ms = int((time.perf_counter() - t1) * 1000)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"model={args.model}")
    print(f"load_ms={load_ms}")
    print(f"gen_ms={gen_ms}")
    print("")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

