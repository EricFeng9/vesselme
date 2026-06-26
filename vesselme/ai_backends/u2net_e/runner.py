from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """生成覆盖完整图像边界的滑窗起点。"""

    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _read_rgb_float(image_path: Path) -> np.ndarray:
    """读取图像并转成 ONNX 模型需要的 RGB float32。"""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    """把模型输出压到 0~1，避免不同导出版本 logits/probability 差异。"""

    prob = probability.astype(np.float32)
    if prob.min() < 0.0 or prob.max() > 1.0:
        prob = 1.0 / (1.0 + np.exp(-prob))
    return np.clip(prob, 0.0, 1.0)


def _extract_first_mask(outputs: list[np.ndarray]) -> np.ndarray:
    """从 ONNX 输出中取第一个单通道预测图。"""

    out = outputs[0]
    if out.ndim == 4:
        out = out[:, :1, :, :]
        out = out[0, 0]
    elif out.ndim == 3:
        out = out[0]
    elif out.ndim != 2:
        raise RuntimeError(f"Unexpected U²Net-E output shape: {out.shape}")
    return _normalize_probability(out)


def predict_probability(
    image: np.ndarray,
    model_path: Path,
    *,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """使用 U²Net-E ONNX 做滑窗推理，并在重叠区域平均概率。"""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0 or stride > patch_size:
        raise ValueError("stride must be positive and <= patch_size")

    # U²Net-E 的官方 ONNX 使用外部 .onnx.data 文件。macOS 上 onnxruntime 会默认尝试
    # CoreMLExecutionProvider，但 CoreML 初始化外部权重时会触发 model_path 为空的问题。
    # 这里固定使用 CPUExecutionProvider，保证安装和推理结果稳定、错误可复现。
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    height, width = image.shape[:2]
    padded_h = max(height, patch_size)
    padded_w = max(width, patch_size)
    pad_h = padded_h - height
    pad_w = padded_w - width
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    y_positions = _window_positions(padded_h, patch_size, stride)
    x_positions = _window_positions(padded_w, patch_size, stride)
    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    for y in y_positions:
        for x in x_positions:
            patch = padded[y:y + patch_size, x:x + patch_size]
            tensor = np.transpose(patch, (2, 0, 1))[None, :, :, :].astype(np.float32)
            outputs = session.run(None, {input_name: tensor})
            prob = _extract_first_mask(outputs)
            if prob.shape != (patch_size, patch_size):
                prob = cv2.resize(prob, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            prob_sum[y:y + patch_size, x:x + patch_size] += prob
            count[y:y + patch_size, x:x + patch_size] += 1.0

    probability = prob_sum / np.maximum(count, 1e-6)
    return probability[:height, :width]


def probability_to_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    """把概率图转成 VesselMe 标准二值 mask。"""

    mask = np.zeros(probability.shape, dtype=np.uint8)
    mask[probability >= threshold] = 255
    return mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U²Net-E ONNX inference for VesselMe.")
    parser.add_argument("--input", required=True, help="输入眼底图片路径")
    parser.add_argument("--output", required=True, help="输出 .npy 路径")
    parser.add_argument("--model", required=True, help="u2net_e.onnx 路径")
    parser.add_argument("--patch-size", type=int, default=384, help="滑窗 patch 边长；官方 ONNX 固定输入 384")
    parser.add_argument("--stride", type=int, default=192, help="滑窗步长")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率阈值")
    parser.add_argument(
        "--output-kind",
        choices=("mask", "probability"),
        default="mask",
        help="输出二值 mask 或 0~1 概率图；全图缩放模式使用概率图避免低分辨率二值块放大。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image = _read_rgb_float(Path(args.input))
    probability = predict_probability(
        image,
        Path(args.model),
        patch_size=args.patch_size,
        stride=args.stride,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_kind == "probability":
        np.save(output, probability.astype(np.float32))
    else:
        mask = probability_to_mask(probability, args.threshold)
        np.save(output, mask)


if __name__ == "__main__":
    main()
