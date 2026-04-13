from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
INVALID_LABEL_CHARS = set('\\/:*?"<>|')


@dataclass
class LabelData:
    image_name: str
    label_name: str
    display_color: tuple[int, int, int] = (255, 255, 255)
    mask: np.ndarray | None = None
    visible: bool = True
    locked: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    tar_path: Path | None = None
    dirty: bool = False
    imported_only: bool = False

    def ensure_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self.mask is None:
            self.mask = np.zeros(shape, dtype=np.uint8)
            return self.mask
        if self.mask.shape != shape:
            raise ValueError(f"Mask shape mismatch: expected {shape}, got {self.mask.shape}")
        if self.mask.dtype != np.uint8:
            self.mask = self.mask.astype(np.uint8)
        return self.mask


@dataclass
class ImageItem:
    path: Path
    labels: dict[str, LabelData] = field(default_factory=dict)
    thumb_ready: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def has_dirty_label(self) -> bool:
        return any(lbl.dirty for lbl in self.labels.values())

    @property
    def has_saved_label(self) -> bool:
        return any(lbl.tar_path is not None and lbl.tar_path.exists() for lbl in self.labels.values())


def is_valid_label_name(label_name: str) -> bool:
    if not label_name.strip():
        return False
    return not any(c in INVALID_LABEL_CHARS for c in label_name)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask.npy must be 2D single-channel")
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask > 0] = 255
    return out
