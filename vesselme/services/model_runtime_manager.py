from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrUnetRuntimeStatus:
    """FR-UNet 后端状态，供 UI 展示和自动分割前检查。"""

    runtime_python: Path
    weight_path: Path
    runtime_ready: bool
    weight_ready: bool

    @property
    def ready(self) -> bool:
        return self.runtime_ready and self.weight_ready


class ModelRuntimeManager:
    """管理 VesselMe 用户目录下的独立 FR-UNet Python 运行时。"""

    WEIGHT_SIZE_MIN_BYTES = 80 * 1024 * 1024

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.app_root = Path.home() / ".vesselme"
        self.runtime_dir = self.app_root / "runtimes" / "fr_unet"
        self.weight_path = self.app_root / "weights" / "fr_unet" / "DRIVE" / "checkpoint-epoch40.pth"

    @property
    def runtime_python(self) -> Path:
        if os.name == "nt":
            return self.runtime_dir / "Scripts" / "python.exe"
        return self.runtime_dir / "bin" / "python"

    def status(self) -> FrUnetRuntimeStatus:
        """检查运行时 Python 和权重是否已经就绪。"""

        return FrUnetRuntimeStatus(
            runtime_python=self.runtime_python,
            weight_path=self.weight_path,
            runtime_ready=self.runtime_python.exists(),
            weight_ready=self.weight_path.exists() and self.weight_path.stat().st_size >= self.WEIGHT_SIZE_MIN_BYTES,
        )

    def build_install_command(self) -> list[str]:
        """构造一键安装命令；本机优先用 fjm 环境，跨平台无 conda 时用当前解释器。"""

        bootstrap_python = self._find_fjm_python()
        return [
            str(bootstrap_python),
            "-m",
            "vesselme.ai_backends.fr_unet.install",
        ]

    def install(self) -> subprocess.CompletedProcess[str]:
        """同步执行一键安装，调用方负责放到后台线程以免阻塞 UI。"""

        return subprocess.run(
            self.build_install_command(),
            cwd=str(self.project_root),
            text=True,
            capture_output=True,
            check=True,
        )

    def build_runner_command(
        self,
        image_path: Path,
        output_path: Path,
        *,
        output_format: str,
        label_name: str,
        device: str = "auto",
        patch_size: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> list[str]:
        """构造 FR-UNet 单图推理命令，所有推理都在独立 venv 子进程中执行。"""

        status = self.status()
        if not status.runtime_ready:
            raise RuntimeError(f"FR-UNet runtime is not installed: {self.runtime_python}")
        if not status.weight_ready:
            raise RuntimeError(f"FR-UNet weight is missing or incomplete: {self.weight_path}")
        return [
            str(self.runtime_python),
            "-m",
            "vesselme.ai_backends.fr_unet.runner",
            "--input",
            str(image_path),
            "--output",
            str(output_path),
            "--weights",
            str(self.weight_path),
            "--device",
            device,
            "--patch-size",
            str(patch_size),
            "--stride",
            str(stride),
            "--batch-size",
            str(batch_size),
            "--threshold",
            str(threshold),
            "--output-format",
            output_format,
            "--label-name",
            label_name,
        ]

    def _find_fjm_python(self) -> Path:
        """定位 fjm 环境 Python；Windows/无 conda 场景用当前解释器支撑跨平台安装按钮。"""

        conda_exe = shutil.which("conda")
        if conda_exe:
            env_name = "fjm"
            if platform.system() == "Windows":
                probe = subprocess.run(
                    [conda_exe, "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
                    text=True,
                    capture_output=True,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    return Path(probe.stdout.strip().splitlines()[-1])
            else:
                home = Path.home()
                candidates = [
                    home / "miniconda3" / "envs" / env_name / "bin" / "python",
                    home / "anaconda3" / "envs" / env_name / "bin" / "python",
                    Path("/opt/miniconda3/envs") / env_name / "bin" / "python",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        return Path(sys.executable)

