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
        full_image_scale_long_edge: int | None = None,
        full_image_threshold_mode: str = "normal",
        cancel_event=None,
        device: str = "auto",
        patch_size: int = 1024,
        stride: int = 512,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """调用自动分割后端生成同尺寸 mask。

        algorithm 由设置菜单控制。roi 为空时处理全图，允许先按长边缩放后推理；
        roi 存在时只处理裁剪区域，并始终保持 ROI 原尺寸推理。
        """

        if algorithm not in {"classical", "u2net_e", "lwnet_hrf"}:
            raise ValueError(f"Unknown auto segment algorithm: {algorithm}")
        if roi is None:
            return self._predict_full_image_mask(
                image_path,
                algorithm=algorithm,
                full_image_scale_long_edge=full_image_scale_long_edge,
                full_image_threshold_mode=full_image_threshold_mode,
                cancel_event=cancel_event,
            )

        roi_image_path = self._write_roi_image(image_path, roi)
        try:
            if algorithm == "u2net_e":
                return self.predict_mask_with_u2net_e(roi_image_path, cancel_event=cancel_event)
            if algorithm == "lwnet_hrf":
                return self.predict_mask_with_lwnet(roi_image_path, cancel_event=cancel_event)
            x0, y0, x1, y1 = roi
            return normalize_mask(
                segment_clarus_vessels(
                    roi_image_path,
                    downsample_long_edge=max(x1 - x0, y1 - y0),
                    cancel_event=cancel_event,
                )
            )
        finally:
            roi_image_path.unlink(missing_ok=True)

    def _predict_full_image_mask(
        self,
        image_path: Path,
        *,
        algorithm: str,
        full_image_scale_long_edge: int | None,
        full_image_threshold_mode: str,
        cancel_event=None,
    ) -> np.ndarray:
        """全图自动分割入口；缩放只作用于全图任务，框选任务不会进入这里。"""

        original_size = self._read_image_size(image_path)
        inference_path = image_path
        scaled_path: Path | None = None
        try:
            if full_image_scale_long_edge is not None:
                scaled_path = self._write_scaled_image(image_path, full_image_scale_long_edge)
                if scaled_path is not None:
                    inference_path = scaled_path

            if algorithm == "u2net_e":
                probability = self.predict_probability_with_u2net_e(inference_path, cancel_event=cancel_event)
                if probability.shape != original_size:
                    probability = cv2.resize(
                        probability,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = self._threshold_probability(
                    probability,
                    self._threshold_for_full_scale(full_image_scale_long_edge, full_image_threshold_mode),
                )
            elif algorithm == "lwnet_hrf":
                mask = self.predict_mask_with_lwnet(inference_path, cancel_event=cancel_event)
                if mask.shape != original_size:
                    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = normalize_mask(segment_clarus_vessels(inference_path, cancel_event=cancel_event))
                if mask.shape != original_size:
                    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            return self._postprocess_large_vessel_mask(mask)
        finally:
            if scaled_path is not None:
                scaled_path.unlink(missing_ok=True)

    def _postprocess_large_vessel_mask(self, mask: np.ndarray) -> np.ndarray:
        """轻量清理全图结果：只过滤小连通域，不做二值形态学腐蚀。"""

        normalized = normalize_mask(mask)
        binary = (normalized > 0).astype(np.uint8)
        min_area = max(32, int(binary.size * 0.000002))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros(binary.shape, dtype=np.uint8)
        for component_id in range(1, count):
            if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == component_id] = 255
        return cleaned

    def _threshold_for_full_scale(self, long_edge: int | None, threshold_mode: str) -> float:
        """按全图缩放档位和细化模式选择 U²Net-E 概率阈值。"""

        mode_offsets = {
            "normal": 0.00,
            "thin": 0.05,
            "thinner": 0.10,
            "thinnest": 0.15,
        }
        if threshold_mode not in mode_offsets:
            raise ValueError(f"Invalid full image threshold mode: {threshold_mode}")
        if long_edge == 512:
            base = 0.70
        elif long_edge == 1024:
            base = 0.65
        elif long_edge == 2048:
            base = 0.60
        elif long_edge == 3072:
            base = 0.55
        else:
            base = 0.50
        return min(0.95, base + mode_offsets[threshold_mode])

    def _threshold_probability(self, probability: np.ndarray, threshold: float) -> np.ndarray:
        """平滑概率边界后再二值化，用阈值控制宽度，避免腐蚀造成锯齿边缘。"""

        prob = np.clip(probability.astype(np.float32), 0.0, 1.0)
        prob = cv2.GaussianBlur(prob, (3, 3), 0.8)
        mask = np.zeros(prob.shape, dtype=np.uint8)
        mask[prob >= threshold] = 255
        return mask

    def predict_mask_with_u2net_e(self, image_path: Path, *, cancel_event=None) -> np.ndarray:
        """调用 U²Net-E ONNX 子进程，返回二值 mask。"""

        with tempfile.TemporaryDirectory(prefix="vesselme_u2net_e_") as tmp_dir:
            output_path = Path(tmp_dir) / "mask.npy"
            command = self.runtime_manager.build_u2net_e_runner_command(image_path, output_path)
            self._run_cancelable_command(command, cancel_event=cancel_event)
            mask = np.load(output_path)
        return normalize_mask(mask)

    def predict_probability_with_u2net_e(self, image_path: Path, *, cancel_event=None) -> np.ndarray:
        """调用 U²Net-E ONNX 子进程，返回 0~1 概率图。"""

        with tempfile.TemporaryDirectory(prefix="vesselme_u2net_e_prob_") as tmp_dir:
            output_path = Path(tmp_dir) / "probability.npy"
            command = self.runtime_manager.build_u2net_e_runner_command(
                image_path,
                output_path,
                output_kind="probability",
            )
            self._run_cancelable_command(command, cancel_event=cancel_event)
            probability = np.load(output_path)
        return np.clip(probability.astype(np.float32), 0.0, 1.0)

    def predict_mask_with_lwnet(self, image_path: Path, *, cancel_event=None) -> np.ndarray:
        """调用 LWNet HRF 子进程，返回二值 mask。"""

        with tempfile.TemporaryDirectory(prefix="vesselme_lwnet_") as tmp_dir:
            output_path = Path(tmp_dir) / "mask.npy"
            command = self.runtime_manager.build_lwnet_runner_command(image_path, output_path)
            self._run_cancelable_command(command, cancel_event=cancel_event)
            mask = np.load(output_path)
        return normalize_mask(mask)

    def _run_cancelable_command(self, command: list[str], *, cancel_event=None) -> None:
        """运行后端子进程；取消任务时杀掉真实推理进程。"""

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

    def _write_scaled_image(self, image_path: Path, long_edge: int) -> Path | None:
        """按指定长边写出全图临时缩放图，供全图自动分割提速和压掉细血管。"""

        if long_edge <= 0:
            raise ValueError(f"Invalid full image scale: {long_edge}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        current_long_edge = max(height, width)
        if current_long_edge <= long_edge:
            return None
        scale = long_edge / float(current_long_edge)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        tmp = tempfile.NamedTemporaryFile(prefix="vesselme_full_scaled_", suffix=image_path.suffix or ".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        if not cv2.imwrite(str(tmp_path), resized):
            raise RuntimeError(f"Failed to write scaled image: {tmp_path}")
        return tmp_path

    def _read_image_size(self, image_path: Path) -> tuple[int, int]:
        """读取图片原始尺寸，返回 mask 使用的 (height, width)。"""

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return image.shape[:2]

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
