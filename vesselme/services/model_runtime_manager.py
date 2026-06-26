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


@dataclass(frozen=True)
class BackendRuntimeStatus:
    """通用后端状态，供 UI 或任务失败信息使用。"""

    runtime_python: Path
    weight_paths: tuple[Path, ...]
    runtime_ready: bool
    weights_ready: bool

    @property
    def ready(self) -> bool:
        return self.runtime_ready and self.weights_ready


class ModelRuntimeManager:
    """管理 VesselMe 用户目录下的独立 FR-UNet Python 运行时。"""

    WEIGHT_SIZE_MIN_BYTES = 80 * 1024 * 1024

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.app_root = Path.home() / ".vesselme"
        self.runtime_dir = self.app_root / "runtimes" / "fr_unet"
        self.weight_path = self.app_root / "weights" / "fr_unet" / "DRIVE" / "checkpoint-epoch40.pth"
        self.u2net_e_runtime_dir = self.app_root / "runtimes" / "u2net_e"
        self.u2net_e_weight_dir = self.app_root / "weights" / "u2net_e"
        self.lwnet_runtime_dir = self.app_root / "runtimes" / "lwnet"
        self.lwnet_source_dir = self.lwnet_runtime_dir / "source"
        self.lwnet_weight_dir = self.app_root / "weights" / "lwnet" / "big_wnet_hrf_av_1024"

    @property
    def runtime_python(self) -> Path:
        if os.name == "nt":
            return self.runtime_dir / "Scripts" / "python.exe"
        return self.runtime_dir / "bin" / "python"

    @property
    def u2net_e_runtime_python(self) -> Path:
        """U²Net-E 独立 venv Python。"""

        if os.name == "nt":
            return self.u2net_e_runtime_dir / "Scripts" / "python.exe"
        return self.u2net_e_runtime_dir / "bin" / "python"

    @property
    def lwnet_runtime_python(self) -> Path:
        """LWNet 独立 venv Python。"""

        if os.name == "nt":
            return self.lwnet_runtime_dir / "Scripts" / "python.exe"
        return self.lwnet_runtime_dir / "bin" / "python"

    def status(self) -> FrUnetRuntimeStatus:
        """检查运行时 Python 和权重是否已经就绪。"""

        return FrUnetRuntimeStatus(
            runtime_python=self.runtime_python,
            weight_path=self.weight_path,
            runtime_ready=self.runtime_python.exists(),
            weight_ready=self.weight_path.exists() and self.weight_path.stat().st_size >= self.WEIGHT_SIZE_MIN_BYTES,
        )

    def u2net_e_status(self) -> BackendRuntimeStatus:
        """检查 U²Net-E ONNX 运行时和权重是否就绪。"""

        weights = (self.u2net_e_weight_dir / "u2net_e.onnx", self.u2net_e_weight_dir / "u2net_e.onnx.data")
        weights_ready = weights[0].exists() and weights[0].stat().st_size >= 500 * 1024
        weights_ready = weights_ready and weights[1].exists() and weights[1].stat().st_size >= 4 * 1024 * 1024
        return BackendRuntimeStatus(
            runtime_python=self.u2net_e_runtime_python,
            weight_paths=weights,
            runtime_ready=self.u2net_e_runtime_python.exists(),
            weights_ready=weights_ready,
        )

    def lwnet_status(self) -> BackendRuntimeStatus:
        """检查 LWNet 运行时、源码和 HRF 权重是否就绪。"""

        weights = (self.lwnet_weight_dir / "model_checkpoint.pth", self.lwnet_weight_dir / "config.cfg")
        weights_ready = weights[0].exists() and weights[0].stat().st_size >= 3 * 1024 * 1024
        weights_ready = weights_ready and weights[1].exists() and self.lwnet_source_dir.exists()
        return BackendRuntimeStatus(
            runtime_python=self.lwnet_runtime_python,
            weight_paths=weights,
            runtime_ready=self.lwnet_runtime_python.exists(),
            weights_ready=weights_ready,
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

    def install_u2net_e(self) -> subprocess.CompletedProcess[str]:
        """同步执行 U²Net-E 一键安装。"""

        return subprocess.run(
            [str(self._find_fjm_python()), "-m", "vesselme.ai_backends.u2net_e.install"],
            cwd=str(self.project_root),
            text=True,
            capture_output=True,
            check=True,
        )

    def install_lwnet(self) -> subprocess.CompletedProcess[str]:
        """同步执行 LWNet 一键安装。"""

        return subprocess.run(
            [str(self._find_fjm_python()), "-m", "vesselme.ai_backends.lwnet.install"],
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

    def build_u2net_e_runner_command(
        self,
        image_path: Path,
        output_path: Path,
        *,
        patch_size: int = 384,
        stride: int = 192,
        threshold: float = 0.5,
    ) -> list[str]:
        """构造 U²Net-E ONNX 推理命令。"""

        status = self.u2net_e_status()
        if not status.runtime_ready:
            raise RuntimeError(f"U²Net-E runtime is not installed: {self.u2net_e_runtime_python}")
        if not status.weights_ready:
            raise RuntimeError(f"U²Net-E weights are missing or incomplete: {self.u2net_e_weight_dir}")
        return [
            str(self.u2net_e_runtime_python),
            "-m",
            "vesselme.ai_backends.u2net_e.runner",
            "--input",
            str(image_path),
            "--output",
            str(output_path),
            "--model",
            str(self.u2net_e_weight_dir / "u2net_e.onnx"),
            "--patch-size",
            str(patch_size),
            "--stride",
            str(stride),
            "--threshold",
            str(threshold),
        ]

    def build_lwnet_runner_command(
        self,
        image_path: Path,
        output_path: Path,
        *,
        im_size: int = 1024,
        threshold: float = 0.4196,
        device: str = "cpu",
    ) -> list[str]:
        """构造 LWNet HRF 推理命令。"""

        status = self.lwnet_status()
        if not status.runtime_ready:
            raise RuntimeError(f"LWNet runtime is not installed: {self.lwnet_runtime_python}")
        if not status.weights_ready:
            raise RuntimeError(f"LWNet source or weights are missing: {self.lwnet_weight_dir}")
        return [
            str(self.lwnet_runtime_python),
            "-m",
            "vesselme.ai_backends.lwnet.runner",
            "--input",
            str(image_path),
            "--output",
            str(output_path),
            "--source-dir",
            str(self.lwnet_source_dir),
            "--model-path",
            str(self.lwnet_weight_dir),
            "--im-size",
            str(im_size),
            "--threshold",
            str(threshold),
            "--device",
            device,
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
