from __future__ import annotations

import io
import json
import re
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np

from vesselme.data.models import LabelData, normalize_mask


TOOL_VERSION = "v0.1"
PATTERN = re.compile(r"^(?P<image_stem>.+)_\[(?P<label_name>.+)]\.tar$")


class TarLabelError(RuntimeError):
    pass


def build_tar_name(image_stem: str, label_name: str) -> str:
    return f"{image_stem}_[{label_name}].tar"


def parse_tar_label_filename(path: Path) -> tuple[str, str] | None:
    m = PATTERN.match(path.name)
    if not m:
        return None
    return m.group("image_stem"), m.group("label_name")


def read_label_tar(tar_path: Path, expected_image_name: str | None = None) -> LabelData:
    if not tar_path.exists():
        raise TarLabelError(f"Label tar not found: {tar_path}")

    with tarfile.open(tar_path, mode="r") as tf:
        mask_member = tf.getmember("mask.npy") if "mask.npy" in tf.getnames() else None
        meta_member = tf.getmember("meta.json") if "meta.json" in tf.getnames() else None
        if mask_member is None or meta_member is None:
            raise TarLabelError("Invalid tar content: required mask.npy and meta.json")

        mask_bytes = tf.extractfile(mask_member)
        meta_bytes = tf.extractfile(meta_member)
        if mask_bytes is None or meta_bytes is None:
            raise TarLabelError("Failed to extract files from tar")

        mask = np.load(io.BytesIO(mask_bytes.read()))
        mask = normalize_mask(mask)
        try:
            meta = json.loads(meta_bytes.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TarLabelError("meta.json is not valid JSON") from exc

    image_name = meta.get("image_name")
    if not image_name:
        raise TarLabelError("meta.json missing image_name")
    if expected_image_name and image_name != expected_image_name:
        raise TarLabelError(
            f"image_name mismatch: expected {expected_image_name}, got {image_name}"
        )

    label_name = meta.get("label_name")
    if not label_name:
        raise TarLabelError("meta.json missing label_name")

    color = tuple(meta.get("display_color", [255, 255, 255]))
    if len(color) != 3:
        color = (255, 255, 255)

    return LabelData(
        image_name=image_name,
        label_name=label_name,
        display_color=(int(color[0]), int(color[1]), int(color[2])),
        mask=mask,
        visible=bool(meta.get("visible", True)),
        locked=bool(meta.get("locked", False)),
        created_at=meta.get("created_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
        updated_at=meta.get("updated_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
        tar_path=tar_path,
        dirty=False,
        imported_only=False,
    )


def write_label_tar(label: LabelData, target_path: Path) -> None:
    if label.mask is None:
        raise TarLabelError("Cannot save label with empty mask")
    mask = normalize_mask(label.mask)

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {
        "image_name": label.image_name,
        "label_name": label.label_name,
        "display_color": [int(v) for v in label.display_color],
        "visible": bool(label.visible),
        "locked": bool(label.locked),
        "mask_encoding": "uint8_binary",
        "foreground_value": 255,
        "background_value": 0,
        "mask_file": "mask.npy",
        "created_at": label.created_at,
        "updated_at": now,
        "tool_version": TOOL_VERSION,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)

    mask_buf = io.BytesIO()
    np.save(mask_buf, mask)
    mask_data = mask_buf.getvalue()
    meta_data = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")

    with tarfile.open(target_path, mode="w") as tf:
        mask_info = tarfile.TarInfo(name="mask.npy")
        mask_info.size = len(mask_data)
        tf.addfile(mask_info, io.BytesIO(mask_data))

        meta_info = tarfile.TarInfo(name="meta.json")
        meta_info.size = len(meta_data)
        tf.addfile(meta_info, io.BytesIO(meta_data))

    label.updated_at = now
    label.tar_path = target_path
    label.dirty = False
    label.imported_only = False


def export_stroke_on_black(mask: np.ndarray, color: tuple[int, int, int], output_path: Path) -> None:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    fg = mask > 0
    rgb[fg, 0] = color[0]
    rgb[fg, 1] = color[1]
    rgb[fg, 2] = color[2]

    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def export_coco_rle_placeholder(mask: np.ndarray) -> dict:
    # Reserved extension point for future COCO/RLE export support.
    del mask
    raise NotImplementedError("COCO/RLE export is reserved for a future version.")
