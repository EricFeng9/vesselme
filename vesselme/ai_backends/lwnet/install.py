from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

import numpy as np
from PIL import Image


REPO_URL = "https://github.com/agaldran/lwnet.git"
CONFIG_URL = "https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/big_wnet_hrf_av_1024/config.cfg"
WEIGHT_URL = "https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/big_wnet_hrf_av_1024/model_checkpoint.pth"
WEIGHT_SIZE_MIN_BYTES = 3 * 1024 * 1024


def app_root() -> Path:
    """返回 VesselMe 用户级 AI 后端目录。"""

    return Path.home() / ".vesselme"


def runtime_dir() -> Path:
    """LWNet 独立 Python 运行时目录。"""

    return app_root() / "runtimes" / "lwnet"


def runtime_python() -> Path:
    """返回跨平台 venv Python 路径。"""

    if sys.platform.startswith("win"):
        return runtime_dir() / "Scripts" / "python.exe"
    return runtime_dir() / "bin" / "python"


def source_dir() -> Path:
    """LWNet 官方源码目录。"""

    return runtime_dir() / "source"


def weight_dir() -> Path:
    """LWNet HRF 1024 官方权重目录。"""

    return app_root() / "weights" / "lwnet" / "big_wnet_hrf_av_1024"


def run(command: list[str], cwd: Path | None = None) -> None:
    """执行安装命令；失败时直接把真实异常抛给 UI。"""

    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_runtime() -> None:
    """创建独立 venv 并安装 LWNet 推理所需最小依赖。"""

    if not runtime_python().exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(runtime_dir())
    run([str(runtime_python()), "-m", "pip", "install", "--upgrade", "pip"])
    run([
        str(runtime_python()),
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "Pillow",
        "tqdm",
    ])


def ensure_source() -> None:
    """下载 LWNet 官方源码；已有源码时不覆盖用户运行时缓存。"""

    if source_dir().exists():
        patch_source()
        return
    if shutil.which("git") is None:
        raise RuntimeError("git is required to install LWNet")
    source_dir().parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(source_dir())])
    patch_source()


def patch_source() -> None:
    """修补 LWNet 旧脚本和新版 scikit-image 的兼容问题。"""

    for target in (source_dir() / "predict_one_image.py", source_dir() / "predict_one_image_av.py"):
        text = target.read_text(encoding="utf-8")
        text = text.replace("draw.circle(y0, x0, r, shape=image.shape)", "draw.disk((y0, x0), r, shape=image.shape)")
        target.write_text(text, encoding="utf-8")


def download(url: str, target: Path, min_size: int = 1) -> None:
    """下载官方权重/配置；大小不达标直接报错。"""

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
    """下载 LWNet HRF 1024 官方 checkpoint 和 config。"""

    download(CONFIG_URL, weight_dir() / "config.cfg")
    download(WEIGHT_URL, weight_dir() / "model_checkpoint.pth", WEIGHT_SIZE_MIN_BYTES)


def smoke_test() -> None:
    """跑最小自检，确认源码、依赖、权重和 runner 都能产出二值 mask。"""

    smoke_image = runtime_dir() / "smoke_input.png"
    if not smoke_image.exists():
        # 自检只验证 LWNet 推理链路，不用 6k 大图，避免安装按钮卡住。
        gradient = np.linspace(0, 255, 256, dtype=np.uint8)
        image = np.stack([np.tile(gradient, (256, 1))] * 3, axis=-1)
        Image.fromarray(image).save(smoke_image)
    command = [
        str(runtime_python()),
        "-m",
        "vesselme.ai_backends.lwnet.runner",
        "--input",
        str(smoke_image),
        "--output",
        str(runtime_dir() / "smoke_mask.npy"),
        "--source-dir",
        str(source_dir()),
        "--model-path",
        str(weight_dir()),
        "--im-size",
        "256",
        "--threshold",
        "0.4196",
    ]
    run(command)


def main() -> None:
    """一键部署 LWNet HRF 后端。"""

    ensure_runtime()
    ensure_source()
    ensure_weights()
    smoke_test()
    print(f"LWNet runtime ready: {runtime_python()}")
    print(f"LWNet source ready: {source_dir()}")
    print(f"LWNet weights ready: {weight_dir()}")


if __name__ == "__main__":
    main()
