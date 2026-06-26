from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AutoSegmentCanceled(RuntimeError):
    """用户取消自动分割任务。"""


def _check_canceled(cancel_event) -> None:
    """在耗时循环中检查取消标志，让运行中任务能尽快停止。"""

    if cancel_event is not None and cancel_event.is_set():
        raise AutoSegmentCanceled("Auto segmentation canceled")


def segment_clarus_vessels(image_path: Path, percentile: float = 97.0, cancel_event=None) -> np.ndarray:
    """针对超广角眼底图的传统血管增强预标注。

    FR-UNet 官方 DRIVE 权重在 Clarus 超广角/拼接图上会严重域外失效。
    这里使用绿色通道、CLAHE 和多尺度黑帽增强暗线结构，只输出预标注起点。
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    _check_canceled(cancel_event)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = np.zeros_like(gray, dtype=np.uint8)
    valid[gray > 15] = 255
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

    green = image[:, :, 1]
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(green)

    response = np.zeros_like(enhanced, dtype=np.float32)
    for kernel_size in (9, 13, 17, 21, 27):
        _check_canceled(cancel_event)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, blackhat.astype(np.float32))

    for length in (17, 25, 33):
        center = length // 2
        for angle in range(0, 180, 15):
            _check_canceled(cancel_event)
            kernel = np.zeros((length, length), dtype=np.uint8)
            radians = np.deg2rad(angle)
            dx = int(np.cos(radians) * center)
            dy = int(np.sin(radians) * center)
            cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
            closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            directional = (closed.astype(np.int16) - enhanced.astype(np.int16)).clip(0, 255)
            response = np.maximum(response, directional.astype(np.float32))

    response[valid == 0] = 0
    values = response[valid > 0]
    if values.size == 0:
        raise RuntimeError("No valid fundus region found")

    threshold = float(np.percentile(values, percentile))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[(response >= threshold) & (valid > 0)] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if label_id % 100 == 0:
            _check_canceled(cancel_event)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= 25:
            cleaned[labels == label_id] = 255

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    cleaned = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    output = np.zeros_like(cleaned, dtype=np.uint8)
    output[cleaned > 0] = 255
    return output
