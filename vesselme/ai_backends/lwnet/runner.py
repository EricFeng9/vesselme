from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _write_full_mask(mask_path: Path, shape: tuple[int, int]) -> None:
    """给 LWNet 官方脚本提供全白 FOV mask。

    官方脚本默认会用圆形眼底假设自动估计 FOV。框选区域通常不是完整圆形眼底，
    自动估计会不稳定；这里明确告诉 LWNet 当前输入整幅图都属于有效推理区域。
    """

    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError(f"Failed to write LWNet FOV mask: {mask_path}")


def _read_binary_prediction(prediction_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """读取 LWNet A/V 输出，并把所有非背景类别合并成血管 mask。"""

    gray = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"LWNet prediction was not created: {prediction_path}")
    target_h, target_w = target_shape
    if gray.shape != target_shape:
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mask[gray > 0] = 255
    return mask


def run_lwnet(
    image_path: Path,
    output_path: Path,
    *,
    source_dir: Path,
    model_path: Path,
    im_size: int,
    threshold: float,
    device: str,
) -> None:
    """调用 LWNet 官方单图推理脚本，并把结果转换成 mask.npy。"""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    target_shape = image.shape[:2]

    with tempfile.TemporaryDirectory(prefix="vesselme_lwnet_") as tmp_dir:
        tmp = Path(tmp_dir)
        input_copy = tmp / image_path.name
        shutil.copy2(image_path, input_copy)
        mask_path = tmp / "fov_mask.png"
        _write_full_mask(mask_path, target_shape)
        result_dir = tmp / "result"
        script = source_dir / "predict_one_image_av.py"
        if not script.exists():
            raise RuntimeError(f"LWNet predict_one_image_av.py not found: {script}")
        command = [
            sys.executable,
            str(script),
            "--model_path",
            str(model_path),
            "--im_path",
            str(input_copy),
            "--mask_path",
            str(mask_path),
            "--result_path",
            str(result_dir),
            "--im_size",
            str(im_size),
            "--tta",
            "no",
            "--device",
            device,
        ]
        completed = subprocess.run(command, cwd=str(source_dir), text=True, capture_output=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        pred_path = result_dir / f"{input_copy.stem}_bin_seg.png"
        mask = _read_binary_prediction(pred_path, target_shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LWNet HRF inference for VesselMe.")
    parser.add_argument("--input", required=True, help="输入眼底图片路径")
    parser.add_argument("--output", required=True, help="输出 mask.npy 路径")
    parser.add_argument("--source-dir", required=True, help="LWNet 官方源码目录")
    parser.add_argument("--model-path", required=True, help="big_wnet_hrf_av_1024 权重目录")
    parser.add_argument("--im-size", type=int, default=1024, help="LWNet 输入边长")
    parser.add_argument("--threshold", type=float, default=0.4196, help="二值化阈值")
    parser.add_argument("--device", default="cpu", help="LWNet 推理设备")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_lwnet(
        Path(args.input),
        Path(args.output),
        source_dir=Path(args.source_dir),
        model_path=Path(args.model_path),
        im_size=args.im_size,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
