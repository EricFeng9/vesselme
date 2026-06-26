from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from vesselme.ai_backends.fr_unet.model import FRUNet
except ModuleNotFoundError:
    from model import FRUNet


def choose_device(requested: str) -> torch.device:
    """按用户请求选择推理设备；设备不可用时直接报错，避免悄悄换设备造成速度预期错误。"""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device(requested)


def load_model(weight_path: Path, device: torch.device) -> FRUNet:
    """加载官方 checkpoint；兼容 DataParallel 产生的 module. 前缀。"""

    if not weight_path.exists():
        raise FileNotFoundError(f"FR-UNet weight not found: {weight_path}")
    # 官方 checkpoint 含有旧训练配置对象；PyTorch 2.6+ 默认 weights_only=True 会拒绝加载。
    # 权重来源由安装脚本固定到 FR-UNet 官方仓库，因此这里显式按完整 checkpoint 读取。
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    cleaned = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model = FRUNet()
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def read_gray_image(image_path: Path) -> np.ndarray:
    """读取眼底图并转为 0~1 的单通道灰度图，保持原始空间分辨率。"""

    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def _window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """生成覆盖完整边界的滑窗起点，最后一个窗口强制贴到图像末端。"""

    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def predict_probability(
    model: FRUNet,
    gray: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    """滑窗推理并在重叠区域做概率平均，输出与原图同尺寸的概率图。"""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > patch_size:
        raise ValueError("stride must be <= patch_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    height, width = gray.shape
    padded_h = max(height, patch_size)
    padded_w = max(width, patch_size)
    padded = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded[:height, :width] = gray

    y_positions = _window_positions(padded_h, patch_size, stride)
    x_positions = _window_positions(padded_w, patch_size, stride)
    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []

    def flush_batch() -> None:
        if not patches:
            return
        batch = np.stack(patches, axis=0)[:, None, :, :]
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        for prob, (y, x) in zip(probs, coords):
            prob_sum[y:y + patch_size, x:x + patch_size] += prob.astype(np.float32)
            count[y:y + patch_size, x:x + patch_size] += 1.0
        patches.clear()
        coords.clear()

    for y in y_positions:
        for x in x_positions:
            patches.append(padded[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
            if len(patches) >= batch_size:
                flush_batch()
    flush_batch()

    if np.any(count == 0):
        raise RuntimeError("Sliding-window stitching failed: uncovered pixels exist")
    probability = prob_sum / count
    return probability[:height, :width]


def probability_to_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    """把概率图转成 VesselMe 标准二值 mask：背景 0，血管 255。"""

    mask = np.zeros(probability.shape, dtype=np.uint8)
    mask[probability >= threshold] = 255
    return mask


def write_mask_npy(mask: np.ndarray, output_path: Path) -> None:
    """保存裸 mask.npy，用于服务层读取后写入内存标签。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)


def write_label_tar(
    mask: np.ndarray,
    output_path: Path,
    image_name: str,
    label_name: str,
    display_color: tuple[int, int, int],
) -> None:
    """直接写出 VesselMe 可自动加载的 .tar 标签包。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {
        "image_name": image_name,
        "label_name": label_name,
        "display_color": [int(v) for v in display_color],
        "visible": True,
        "locked": False,
        "mask_encoding": "uint8_binary",
        "foreground_value": 255,
        "background_value": 0,
        "mask_file": "mask.npy",
        "created_at": now,
        "updated_at": now,
        "tool_version": "fr-unet-auto-segmentation",
    }

    mask_buf = io.BytesIO()
    np.save(mask_buf, mask)
    mask_data = mask_buf.getvalue()
    meta_data = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")

    with tarfile.open(output_path, mode="w") as tf:
        mask_info = tarfile.TarInfo("mask.npy")
        mask_info.size = len(mask_data)
        tf.addfile(mask_info, io.BytesIO(mask_data))

        meta_info = tarfile.TarInfo("meta.json")
        meta_info.size = len(meta_data)
        tf.addfile(meta_info, io.BytesIO(meta_data))


def run_inference(args: argparse.Namespace) -> None:
    """完整单图推理流程：读图、加载权重、滑窗推理、保存结果。"""

    image_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weight_path = Path(args.weights).expanduser().resolve()
    device = choose_device(args.device)

    gray = read_gray_image(image_path)
    model = load_model(weight_path, device)
    probability = predict_probability(
        model,
        gray,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    mask = probability_to_mask(probability, args.threshold)

    if args.output_format == "tar":
        write_label_tar(mask, output_path, image_path.name, args.label_name, tuple(args.display_color))
    else:
        write_mask_npy(mask, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FR-UNet single-image inference for VesselMe.")
    parser.add_argument("--input", required=True, help="输入眼底图像路径")
    parser.add_argument("--output", required=True, help="输出 mask.npy 或 VesselMe .tar 路径")
    parser.add_argument("--weights", required=True, help="FR-UNet checkpoint-epoch40.pth 路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="推理设备")
    parser.add_argument("--patch-size", type=int, default=512, help="滑窗 patch 边长")
    parser.add_argument("--stride", type=int, default=256, help="滑窗步长")
    parser.add_argument("--batch-size", type=int, default=1, help="推理 batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率阈值")
    parser.add_argument("--output-format", choices=["npy", "tar"], default="npy", help="输出格式")
    parser.add_argument("--label-name", default="auto_vessel", help=".tar 输出时写入的标签名")
    parser.add_argument(
        "--display-color",
        type=int,
        nargs=3,
        default=(255, 255, 255),
        metavar=("R", "G", "B"),
        help=".tar 输出时写入的显示颜色",
    )
    return parser


def main() -> None:
    run_inference(build_parser().parse_args())


if __name__ == "__main__":
    main()
