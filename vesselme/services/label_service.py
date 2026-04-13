from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vesselme.data.io import build_tar_name, export_stroke_on_black, read_label_tar, write_label_tar
from vesselme.data.models import ImageItem, LabelData, is_valid_label_name


class LabelService:
    def __init__(self, tool_version: str = "v0.1") -> None:
        self.tool_version = tool_version

    def make_default_name(self, image_item: ImageItem) -> str:
        idx = 1
        while True:
            name = f"label{idx}"
            if name not in image_item.labels:
                return name
            idx += 1

    def create_label(
        self,
        image_item: ImageItem,
        label_name: str,
        mask_shape: tuple[int, int],
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> LabelData:
        if not is_valid_label_name(label_name):
            raise ValueError("Invalid label name")
        if label_name in image_item.labels:
            raise ValueError("Label already exists")

        label = LabelData(
            image_name=image_item.name,
            label_name=label_name,
            display_color=color,
            mask=np.zeros(mask_shape, dtype=np.uint8),
            dirty=True,
            imported_only=True,
        )
        image_item.labels[label_name] = label
        return label

    def import_tar(self, image_item: ImageItem, tar_path: Path, mask_shape: tuple[int, int]) -> LabelData:
        label = read_label_tar(tar_path, expected_image_name=image_item.name)
        if label.mask is None:
            raise ValueError("Imported mask is empty")
        if label.mask.shape != mask_shape:
            raise ValueError(
                f"Imported mask shape mismatch: expected {mask_shape}, got {label.mask.shape}"
            )

        candidate = label.label_name
        if candidate in image_item.labels:
            i = 2
            while f"{candidate}_{i}" in image_item.labels:
                i += 1
            candidate = f"{candidate}_{i}"
            label.label_name = candidate

        label.imported_only = True
        label.dirty = True
        label.tar_path = None
        image_item.labels[label.label_name] = label
        return label

    def import_image_as_new_label(
        self,
        image_item: ImageItem,
        image_path: Path,
        mask_shape: tuple[int, int],
        label_name: str | None = None,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> LabelData:
        if label_name is None:
            label_name = self.make_default_name(image_item)
        if not is_valid_label_name(label_name):
            raise ValueError("Invalid label name")
        if label_name in image_item.labels:
            raise ValueError("Label already exists")
        mask = self._read_binary_mask_from_image(image_path, mask_shape)
        label = LabelData(
            image_name=image_item.name,
            label_name=label_name,
            display_color=color,
            mask=mask,
            dirty=True,
            imported_only=True,
        )
        image_item.labels[label_name] = label
        return label

    def overwrite_label_mask_from_image(
        self,
        image_item: ImageItem,
        label_name: str,
        image_path: Path,
        mask_shape: tuple[int, int],
    ) -> LabelData:
        if label_name not in image_item.labels:
            raise ValueError("No label selected to overwrite")
        label = image_item.labels[label_name]
        label.mask = self._read_binary_mask_from_image(image_path, mask_shape)
        label.dirty = True
        return label

    def _read_binary_mask_from_image(self, image_path: Path, mask_shape: tuple[int, int]) -> np.ndarray:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Use a robust threshold to avoid turning low-intensity noise/antialiasing
        # pixels into vessels (which visually thickens the mask).
        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[gray >= 128] = 255

        target_h, target_w = mask_shape
        if binary.shape != mask_shape:
            binary = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        mask[binary > 0] = 255
        return mask

    def rename_label(self, image_item: ImageItem, old_name: str, new_name: str) -> LabelData:
        if old_name not in image_item.labels:
            raise KeyError(old_name)
        if not is_valid_label_name(new_name):
            raise ValueError("Invalid label name")
        if new_name in image_item.labels and new_name != old_name:
            raise ValueError("Label already exists")

        label = image_item.labels.pop(old_name)
        old_tar_path = label.tar_path
        label.label_name = new_name
        label.dirty = True

        if old_tar_path and old_tar_path.exists():
            new_tar_path = old_tar_path.parent / build_tar_name(image_item.stem, new_name)
            old_tar_path.rename(new_tar_path)
            label.tar_path = new_tar_path

        image_item.labels[new_name] = label
        return label

    def delete_label(self, image_item: ImageItem, label_name: str) -> None:
        if label_name not in image_item.labels:
            return
        label = image_item.labels.pop(label_name)
        if label.tar_path and label.tar_path.exists():
            label.tar_path.unlink()

    def save_label(self, image_item: ImageItem, label_name: str) -> Path:
        label = image_item.labels[label_name]
        target_path = image_item.path.parent / build_tar_name(image_item.stem, label_name)
        write_label_tar(label, target_path)
        return target_path

    def export_stroke(
        self,
        image_item: ImageItem,
        label_name: str,
        output_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        label = image_item.labels[label_name]
        if label.mask is None:
            raise ValueError("Label mask is empty")
        if output_path is None:
            target_dir = output_dir or image_item.path.parent / "exports"
            output_path = target_dir / f"{image_item.stem}_[{label_name}]_stroke.png"
        export_stroke_on_black(label.mask, label.display_color, output_path)
        return output_path
