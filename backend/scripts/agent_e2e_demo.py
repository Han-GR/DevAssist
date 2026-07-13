"""
端到端 Agent Demo：写 FastAPI endpoint，并在沙箱中执行测试验证。

用法（推荐）：
1) 先确保 LLM 配置可用（backend/.env）：
   - LLM_PROVIDER / LLM_API_KEY / LLM_MODEL
2) 确保 sandbox 镜像里包含 fastapi（否则 import 会失败）：
   - 在 backend/.env 里设置：SANDBOX_IMAGE=devassist-backend
   - 并确保该镜像存在：docker compose build backend
3) 运行：
   python backend/scripts/agent_e2e_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap

import docker

from app.agent.builtin_tools import create_execute_code_tool
from app.agent.react import ReActAgent
from app.agent.tools import ToolRegistry
from app.core.config import get_settings, setup_logging
from app.core.llm import LLMClient


def _truncate(value: str, limit: int = 800) -> str:
    """
    截断长文本，避免终端输出过长。

    Args:
        value (str): 原始文本。
        limit (int): 最大保留字符数，默认 800。

    Returns:
        str: 截断后的文本；若未超限则返回原文本。

    Raises:
        None

    Notes/Examples:
        主要用于打印 thought/tool_args/observation 等可能很长的字段。
    """
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n...<truncated>"


def _build_prompt() -> str:
    """
    构造用于端到端演示的用户 Prompt（英文）。

    Args:
        None

    Returns:
        str: 用于触发 Agent “写代码→执行→验证→总结”的提示词。

    Raises:
        None

    Notes/Examples:
        为了让 Agent 行为稳定，这里强制要求调用 execute_code 并以 ALL_TESTS_PASSED 作为成功证据。
    """
    return textwrap.dedent(
        """
        You are a senior Python developer.

        Task:
        1) Write a minimal FastAPI app with a single endpoint:
           - GET /sum
           - query params: a:int, b:int
           - returns: {"sum": a + b}
        2) Write tests for this endpoint using fastapi.testclient.TestClient.
           - Use plain assert statements (do NOT rely on pytest).
           - On success, print exactly: ALL_TESTS_PASSED
        3) You MUST call the execute_code tool to run the tests in the sandbox.
           - Put the FastAPI app and tests in a single Python script string passed to execute_code.
           - Do NOT answer with a final response until the sandbox run prints ALL_TESTS_PASSED and exit_code==0.

        Output requirements:
        - Follow the ReAct format required by the system prompt.
        - In your final answer, include:
          - The FastAPI code
          - The test code
          - The sandbox stdout/stderr/exit_code evidence (briefly summarized)
        """
    ).strip()


async def _run(*, show_steps: bool) -> int:
    """
    执行一次端到端 demo，并可选打印 ReAct steps。

    Args:
        show_steps (bool): 是否打印每一步的 thought/action/observation，便于学习。

    Returns:
        int: 进程退出码（0 表示成功）。

    Raises:
        SystemExit: 当 LLM 配置缺失或 Docker 不可用时抛出，直接中断 demo。

    Notes/Examples:
        - 若在容器内运行，需要挂载 /var/run/docker.sock 才能让 execute_code 启动沙箱容器。
        - 若 sandbox 镜像缺少 fastapi，请设置 SANDBOX_IMAGE=devassist-backend 并提前 build。
    """
    settings = get_settings()
    setup_logging(settings=settings)

    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is empty. Please configure backend/.env first.")

    try:
        docker.from_env().ping()
    except Exception as exc:
        raise SystemExit(
            "Docker is not reachable. If you are running inside a container, "
            "mount /var/run/docker.sock or run this script on the host."
        ) from exc

    llm = LLMClient.from_settings(settings)
    registry = ToolRegistry()
    registry.register(create_execute_code_tool())

    agent = ReActAgent(llm=llm, tools=registry)
    answer, steps = await agent.run(user_input=_build_prompt())

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

    if show_steps:
        print("\n=== STEPS ===\n")
        for idx, step in enumerate(steps):
            tool = step.tool_name or "-"
            print(f"[{idx}] tool={tool}")
            if step.thought.strip():
                print("THOUGHT:")
                print(_truncate(step.thought))
            if step.action_raw.strip():
                print("ACTION_RAW:")
                print(_truncate(step.action_raw))
            if step.tool_args is not None:
                print("TOOL_ARGS:")
                print(_truncate(str(step.tool_args)))
            if step.observation is not None:
                print("OBSERVATION:")
                print(_truncate(str(step.observation)))
            print()

    return 0


def main() -> None:
    """
    CLI 入口。

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: demo 运行失败时抛出非 0 退出码。

    Notes/Examples:
        python backend/scripts/agent_e2e_demo.py --show-steps
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="打印每一步的 thought/action/observation（便于学习 ReAct 过程）",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(show_steps=bool(args.show_steps))))


if __name__ == "__main__":
    main()
