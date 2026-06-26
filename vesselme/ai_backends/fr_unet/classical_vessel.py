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


def _resize_to_long_edge(image: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    """把大图缩到固定长边，返回缩放图和缩放比例。

    自动分割当前目标是辅助标注大血管。先降采样可以压掉毛细血管、
    纹理和亮点噪声，让后续增强主要响应连续的大血管主干。
    """

    height, width = image.shape[:2]
    scale = long_edge / float(max(height, width))
    if scale >= 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _build_valid_fundus_mask(image: np.ndarray) -> np.ndarray:
    """提取有效眼底区域，避免黑边和拼接空白区域参与阈值计算。"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = np.zeros_like(gray, dtype=np.uint8)
    valid[gray > 15] = 255
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return valid


def _enhance_large_dark_lines(green: np.ndarray, cancel_event=None) -> np.ndarray:
    """增强绿色通道里的大尺度暗线结构。

    这里只使用偏大的结构元素。小尺度结构元素虽然能抓到细血管，
    也会把反光、病灶边缘和背景纹理变成毛躁噪声。
    """

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(green)
    response = np.zeros_like(enhanced, dtype=np.float32)

    # 圆形黑帽负责抓不同宽度的大血管暗线，尺度越大越偏向主血管。
    for kernel_size in (17, 25, 33, 45):
        _check_canceled(cancel_event)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, blackhat.astype(np.float32))

    # 方向闭运算负责补充长条状血管响应，让跨方向的大血管更连续。
    for length in (33, 45, 59):
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

    # 轻微平滑响应图，降低单点亮斑造成的孤立峰值。
    return cv2.GaussianBlur(response, (3, 3), 0)


def _keep_large_connected_vessels(mask: np.ndarray, valid: np.ndarray, cancel_event=None) -> np.ndarray:
    """删除小而散的噪声连通域，并连接大血管上的短断点。"""

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    min_area = max(120, int(mask.size * 0.000025))
    for label_id in range(1, num_labels):
        if label_id % 100 == 0:
            _check_canceled(cancel_event)
        area = stats[label_id, cv2.CC_STAT_AREA]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        longer_side = max(width, height)
        shorter_side = max(1, min(width, height))
        extent = area / float(width * height)
        elongation = longer_side / float(shorter_side)

        # 大血管是长条结构；块状病灶、反光和拼接伪影通常填充率高、长宽比低。
        is_line_like = elongation >= 2.2 or extent <= 0.28
        if area >= min_area and longer_side >= 18 and is_line_like:
            cleaned[labels == label_id] = 255

    # 输出稍微偏粗，方便用户把它当初始 mask 用橡皮修边。
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cleaned = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    cleaned[valid == 0] = 0
    return cleaned


def segment_clarus_vessels(
    image_path: Path,
    percentile: float = 98.2,
    downsample_long_edge: int = 2200,
    cancel_event=None,
) -> np.ndarray:
    """生成面向辅助标注的大血管初始 mask。

    当前目标不是完整细血管分割，而是给用户一个连贯、低噪声、偏粗的大血管底稿。
    因此流程是：降采样压噪声 -> 大尺度血管增强 -> 保守阈值 -> 清小连通域 ->
    连接断点 -> 放回原图尺寸。
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    _check_canceled(cancel_event)

    original_height, original_width = image.shape[:2]
    working_image, _ = _resize_to_long_edge(image, downsample_long_edge)
    _check_canceled(cancel_event)

    valid = _build_valid_fundus_mask(working_image)
    green = working_image[:, :, 1]
    response = _enhance_large_dark_lines(green, cancel_event=cancel_event)
    response[valid == 0] = 0
    values = response[valid > 0]
    if values.size == 0:
        raise RuntimeError("No valid fundus region found")

    threshold = float(np.percentile(values, percentile))
    mask = np.zeros_like(green, dtype=np.uint8)
    mask[(response >= threshold) & (valid > 0)] = 255
    cleaned = _keep_large_connected_vessels(mask, valid, cancel_event=cancel_event)

    output = cv2.resize(cleaned, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    output[output > 0] = 255
    return output
