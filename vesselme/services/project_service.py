from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vesselme.data.io import build_tar_name, parse_tar_label_filename, read_label_tar
from vesselme.data.models import IMAGE_EXTENSIONS, ImageItem, LabelData


class ProjectService:
    def __init__(self) -> None:
        self.root_dir: Path | None = None
        self.images: list[ImageItem] = []

    def open_folder(self, folder: Path) -> list[ImageItem]:
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(str(folder))

        paths = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        self.root_dir = folder
        self.images = [ImageItem(path=p) for p in paths]

        for item in self.images:
            self._autoload_labels(item)
        return self.images

    def load_image_rgb(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _autoload_labels(self, item: ImageItem) -> None:
        image_stem = item.path.stem
        for tar_path in sorted(item.path.parent.glob("*.tar")):
            parsed = parse_tar_label_filename(tar_path)
            if parsed is None:
                continue
            stem, _ = parsed
            if stem != image_stem:
                continue
            try:
                label = read_label_tar(tar_path, expected_image_name=item.path.name)
                item.labels[label.label_name] = label
            except Exception:
                # Non-blocking by spec: allow continuing when one tar is broken.
                continue

    def infer_target_tar(self, image_item: ImageItem, label: LabelData) -> Path:
        return image_item.path.parent / build_tar_name(image_item.path.stem, label.label_name)
