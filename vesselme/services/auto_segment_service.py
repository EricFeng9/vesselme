from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from vesselme.data.io import build_tar_name, write_label_tar
from vesselme.data.models import ImageItem, LabelData, normalize_mask
from vesselme.ai_backends.fr_unet.classical_vessel import AutoSegmentCanceled, segment_clarus_vessels
from vesselme.services.model_runtime_manager import ModelRuntimeManager


class AutoSegmentService:
    """生成自动分割 mask，并转换为 VesselMe 当前图片的标签数据。"""

    def __init__(self, runtime_manager: ModelRuntimeManager | None = None) -> None:
        self.runtime_manager = runtime_manager or ModelRuntimeManager()

    def predict_mask(
        self,
        image_path: Path,
        *,
        cancel_event=None,
        device: str = "auto",
        patch_size: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """调用自动分割后端生成同尺寸 mask。

        当前默认目标是生成大血管辅助标注底稿。FR-UNet DRIVE 权重对超广角图域外失效，
        继续默认使用会生成非血管散点，违背“减少手工清噪”的真实目标。
        """

        return normalize_mask(segment_clarus_vessels(image_path, cancel_event=cancel_event))

    def predict_mask_with_fr_unet(
        self,
        image_path: Path,
        *,
        device: str = "auto",
        patch_size: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """保留 FR-UNet 调用入口，仅供后续适配同域图像或新权重时显式使用。"""

        with tempfile.TemporaryDirectory(prefix="vesselme_fr_unet_") as tmp_dir:
            output_path = Path(tmp_dir) / "mask.npy"
            command = self.runtime_manager.build_runner_command(
                image_path,
                output_path,
                output_format="npy",
                label_name="auto_vessel",
                device=device,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                threshold=threshold,
            )
            subprocess.run(command, cwd=str(self.runtime_manager.project_root), check=True, text=True, capture_output=True)
            mask = np.load(output_path)
        return normalize_mask(mask)

    def create_or_overwrite_label(
        self,
        image_item: ImageItem,
        mask: np.ndarray,
        *,
        label_name: str = "auto_vessel",
        color: tuple[int, int, int] = (255, 255, 255),
        overwrite: bool = False,
    ) -> LabelData:
        """把模型 mask 写入图片标签；存在同名标签时必须显式 overwrite。"""

        if label_name in image_item.labels and not overwrite:
            raise ValueError(f"Label already exists: {label_name}")
        normalized = normalize_mask(mask)
        label = image_item.labels.get(label_name)
        if label is None:
            label = LabelData(
                image_name=image_item.name,
                label_name=label_name,
                display_color=color,
                mask=normalized,
                dirty=True,
                imported_only=True,
            )
            image_item.labels[label_name] = label
        else:
            label.mask = normalized
            label.display_color = color
            label.dirty = True
            label.imported_only = True
        return label

    def save_label_tar(self, image_item: ImageItem, label: LabelData) -> Path:
        """自动分割完成后立即保存为 VesselMe 可自动加载的同级 .tar。"""

        target_path = image_item.path.parent / build_tar_name(image_item.stem, label.label_name)
        write_label_tar(label, target_path)
        return target_path
