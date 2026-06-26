from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import cv2
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
        algorithm: str = "classical",
        roi: tuple[int, int, int, int] | None = None,
        cancel_event=None,
        device: str = "auto",
        patch_size: int = 1024,
        stride: int = 512,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """调用自动分割后端生成同尺寸 mask。

        algorithm 由设置菜单控制。roi 为空时处理全图；roi 存在时只处理裁剪区域，
        返回值是该区域自身大小的二值 mask，主窗口负责并入目标标签。
        """

        if algorithm not in {"classical", "fr_unet"}:
            raise ValueError(f"Unknown auto segment algorithm: {algorithm}")
        if roi is None:
            if algorithm == "fr_unet":
                return self.predict_mask_with_fr_unet(
                    image_path,
                    device=device,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size,
                    threshold=threshold,
                    cancel_event=cancel_event,
                )
            return normalize_mask(segment_clarus_vessels(image_path, cancel_event=cancel_event))

        roi_image_path = self._write_roi_image(image_path, roi)
        try:
            if algorithm == "fr_unet":
                return self.predict_mask_with_fr_unet(
                    roi_image_path,
                    device=device,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size,
                    threshold=threshold,
                    cancel_event=cancel_event,
                )
            return normalize_mask(segment_clarus_vessels(roi_image_path, cancel_event=cancel_event))
        finally:
            roi_image_path.unlink(missing_ok=True)

    def predict_mask_with_fr_unet(
        self,
        image_path: Path,
        *,
        device: str = "auto",
        patch_size: int = 512,
        stride: int = 256,
        batch_size: int = 1,
        threshold: float = 0.5,
        cancel_event=None,
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
            process = subprocess.Popen(
                command,
                cwd=str(self.runtime_manager.project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    process.kill()
                    process.communicate()
                    raise AutoSegmentCanceled("Auto segmentation canceled")
                time.sleep(0.2)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
            mask = np.load(output_path)
        return normalize_mask(mask)

    def _write_roi_image(self, image_path: Path, roi: tuple[int, int, int, int]) -> Path:
        """把图像 ROI 写成临时图片，让两类后端共用同一套单图入口。"""

        x0, y0, x1, y1 = roi
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI: {roi}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError(f"ROI outside image bounds: {roi}, image={(width, height)}")
        roi_image = image[y0:y1, x0:x1]
        tmp = tempfile.NamedTemporaryFile(prefix="vesselme_roi_", suffix=image_path.suffix or ".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        if not cv2.imwrite(str(tmp_path), roi_image):
            raise RuntimeError(f"Failed to write ROI image: {tmp_path}")
        return tmp_path

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
