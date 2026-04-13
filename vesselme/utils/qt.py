from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def rgb_ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    h, w, ch = arr.shape
    if ch != 3:
        raise ValueError("Expected RGB image")
    data = np.ascontiguousarray(arr)
    qimg = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
