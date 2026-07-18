from __future__ import annotations

import platform
import sys


def describe_training_env() -> dict[str, object]:
    """
    检查微调训练环境的关键依赖与 GPU 可用性。

    Args:
        无。

    Returns:
        dict: 结构化环境信息，包含 Python/OS/Torch/CUDA 等字段。

    Raises:
        无。

    Notes:
        - 该脚本用于快速排查“依赖没装 / GPU 不可用 / CUDA 版本不匹配”等问题。
        - 若未安装 torch，会返回 torch_installed=false，不会抛异常。
    """

    info: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_installed": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        import torch  # type: ignore
    except Exception as exc:
        info["torch_error"] = str(exc)
        return info

    info["torch_installed"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(getattr(torch.cuda, "is_available", lambda: False)())
    info["cuda_version"] = getattr(torch.version, "cuda", None)

    if info["cuda_available"]:
        count = int(torch.cuda.device_count())
        info["gpu_count"] = count
        gpus: list[dict[str, object]] = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        info["gpus"] = gpus

    return info


def main() -> int:
    env = describe_training_env()
    for k in sorted(env.keys()):
        print(f"{k}={env[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

