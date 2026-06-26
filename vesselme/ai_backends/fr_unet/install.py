from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path


WEIGHT_URL = "https://raw.githubusercontent.com/lseventeen/FR-UNet/master/pretrained_weights/DRIVE/checkpoint-epoch40.pth"
WEIGHT_SIZE_MIN_BYTES = 80 * 1024 * 1024


def app_root() -> Path:
    """返回用户级 VesselMe AI 资源目录。"""

    return Path.home() / ".vesselme"


def runtime_dir() -> Path:
    """FR-UNet 独立 Python 运行时目录。"""

    return app_root() / "runtimes" / "fr_unet"


def weight_path() -> Path:
    """官方 DRIVE 权重的固定存放路径。"""

    return app_root() / "weights" / "fr_unet" / "DRIVE" / "checkpoint-epoch40.pth"


def runtime_python() -> Path:
    """按操作系统返回 venv 内部 Python 路径。"""

    if os.name == "nt":
        return runtime_dir() / "Scripts" / "python.exe"
    return runtime_dir() / "bin" / "python"


def run_command(command: list[str]) -> None:
    """执行安装命令；失败时让异常携带原始返回码和命令。"""

    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def create_runtime() -> None:
    """创建独立 venv，使 PyTorch 不进入 VesselMe 主环境。"""

    runtime_dir().parent.mkdir(parents=True, exist_ok=True)
    if not runtime_python().exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(runtime_dir())


def detect_torch_install_command() -> list[str]:
    """根据系统和显卡选择 PyTorch 安装命令。"""

    py = str(runtime_python())
    base = [py, "-m", "pip", "install"]
    if platform.system() == "Windows" and shutil.which("nvidia-smi"):
        return base + ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"]
    return base + ["torch", "torchvision"]


def install_packages() -> None:
    """安装 FR-UNet 推理需要的最小依赖集合。"""

    py = str(runtime_python())
    run_command([py, "-m", "pip", "install", "--upgrade", "pip"])
    run_command(detect_torch_install_command())
    run_command([py, "-m", "pip", "install", "numpy", "opencv-python", "Pillow", "tqdm"])


def download_weight(force: bool = False) -> None:
    """下载官方 DRIVE checkpoint；文件过小视为损坏并直接失败。"""

    target = weight_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= WEIGHT_SIZE_MIN_BYTES and not force:
        print(f"weight already exists: {target}", flush=True)
        return
    if target.exists():
        target.unlink()
    print(f"download {WEIGHT_URL}", flush=True)
    urllib.request.urlretrieve(WEIGHT_URL, target)
    size = target.stat().st_size
    if size < WEIGHT_SIZE_MIN_BYTES:
        raise RuntimeError(f"Downloaded weight is too small: {size} bytes")


def smoke_test() -> None:
    """验证 torch 可导入、权重可加载、模型可执行一次 64x64 前向。"""

    repo_root = Path(__file__).resolve().parents[3]
    code = (
        "import torch;"
        "from vesselme.ai_backends.fr_unet.model import FRUNet;"
        f"ckpt=torch.load(r'{weight_path()}', map_location='cpu', weights_only=False);"
        "sd=ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt;"
        "sd={k.removeprefix('module.'): v for k,v in sd.items()};"
        "m=FRUNet();"
        "m.load_state_dict(sd, strict=True);"
        "m.eval();"
        "y=m(torch.zeros(1,1,64,64));"
        "assert tuple(y.shape)==(1,1,64,64);"
        "print('FR-UNet smoke test passed')"
    )
    run_command([str(runtime_python()), "-c", code_with_path(repo_root, code)])


def code_with_path(repo_root: Path, code: str) -> str:
    """把项目根目录加入 sys.path，让独立 venv 能 import 当前源码。"""

    return f"import sys; sys.path.insert(0, r'{repo_root}'); {code}"


def install(force_weight: bool = False) -> None:
    """完整一键部署流程。"""

    create_runtime()
    install_packages()
    download_weight(force=force_weight)
    smoke_test()
    print("FR-UNet runtime is ready", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install VesselMe FR-UNet runtime.")
    parser.add_argument("--force-weight", action="store_true", help="重新下载权重")
    parser.add_argument("--print-paths", action="store_true", help="只打印运行时路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.print_paths:
        print(f"runtime_python={runtime_python()}")
        print(f"weight_path={weight_path()}")
        return
    install(force_weight=args.force_weight)


if __name__ == "__main__":
    main()
