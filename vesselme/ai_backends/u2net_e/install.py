from __future__ import annotations

import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_URL = "https://raw.githubusercontent.com/tapos-datta/Retinal_Vessel_Segmentation/main/pre-trained/u2net_e.onnx"
DATA_URL = "https://raw.githubusercontent.com/tapos-datta/Retinal_Vessel_Segmentation/main/pre-trained/u2net_e.onnx.data"
MODEL_SIZE_MIN_BYTES = 500 * 1024
DATA_SIZE_MIN_BYTES = 4 * 1024 * 1024


def app_root() -> Path:
    """返回 VesselMe 用户级 AI 后端目录。"""

    return Path.home() / ".vesselme"


def runtime_dir() -> Path:
    """U²Net-E 独立 Python 运行时目录。"""

    return app_root() / "runtimes" / "u2net_e"


def runtime_python() -> Path:
    """返回跨平台 venv Python 路径。"""

    if sys.platform.startswith("win"):
        return runtime_dir() / "Scripts" / "python.exe"
    return runtime_dir() / "bin" / "python"


def weight_dir() -> Path:
    """U²Net-E 官方 ONNX 权重目录。"""

    return app_root() / "weights" / "u2net_e"


def run(command: list[str]) -> None:
    """执行安装命令；失败时直接把真实异常抛给 UI。"""

    subprocess.run(command, check=True)


def ensure_runtime() -> None:
    """创建独立 venv 并安装最小推理依赖。"""

    if not runtime_python().exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(runtime_dir())
    run([str(runtime_python()), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(runtime_python()), "-m", "pip", "install", "onnxruntime", "numpy", "opencv-python", "Pillow"])


def download(url: str, target: Path, min_size: int) -> None:
    """下载官方权重文件；大小不达标直接报错。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= min_size:
        return
    tmp = target.with_suffix(target.suffix + ".download")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < min_size:
        raise RuntimeError(f"Downloaded file is incomplete: {target}")
    tmp.replace(target)


def ensure_weights() -> None:
    """下载 U²Net-E 官方 ONNX 外部数据权重。"""

    download(MODEL_URL, weight_dir() / "u2net_e.onnx", MODEL_SIZE_MIN_BYTES)
    download(DATA_URL, weight_dir() / "u2net_e.onnx.data", DATA_SIZE_MIN_BYTES)


def smoke_test() -> None:
    """跑最小自检，确认依赖、权重和 runner 都能产出二值 mask。"""

    smoke_image = runtime_dir() / "smoke_input.png"
    if not smoke_image.exists():
        # 自检只验证推理链路，不用 6k 大图，避免安装按钮长时间无响应。
        gradient = np.linspace(0, 255, 384, dtype=np.uint8)
        image = np.stack([np.tile(gradient, (384, 1))] * 3, axis=-1)
        Image.fromarray(image).save(smoke_image)
    command = [
        str(runtime_python()),
        "-m",
        "vesselme.ai_backends.u2net_e.runner",
        "--input",
        str(smoke_image),
        "--output",
        str(runtime_dir() / "smoke_mask.npy"),
        "--model",
        str(weight_dir() / "u2net_e.onnx"),
        "--patch-size",
        "384",
        "--stride",
        "384",
        "--threshold",
        "0.5",
    ]
    run(command)


def main() -> None:
    """一键部署 U²Net-E 后端。"""

    ensure_runtime()
    ensure_weights()
    smoke_test()
    print(f"U²Net-E runtime ready: {runtime_python()}")
    print(f"U²Net-E model ready: {weight_dir() / 'u2net_e.onnx'}")


if __name__ == "__main__":
    main()
