from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _base_pixmap(size: int = 18) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return pm, p


def eye_icon(visible: bool, size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    pen = QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Outline eye shape to match lock/delete icon line style.
    eye = QPainterPath()
    eye.moveTo(size * 0.16, size * 0.5)
    eye.quadTo(size * 0.5, size * 0.22, size * 0.84, size * 0.5)
    eye.quadTo(size * 0.5, size * 0.78, size * 0.16, size * 0.5)
    p.drawPath(eye)
    p.drawEllipse(QPointF(size * 0.5, size * 0.5), size * 0.085, size * 0.085)

    if not visible:
        slash_pen = QPen(color, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(slash_pen)
        p.drawLine(int(size * 0.2), int(size * 0.8), int(size * 0.8), int(size * 0.2))
    p.end()
    return QIcon(pm)


def lock_icon(locked: bool, size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    p.setPen(QPen(color, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)

    body_w = size * 0.5
    body_h = size * 0.36
    body_x = (size - body_w) / 2
    body_y = size * 0.48
    p.drawRoundedRect(body_x, body_y, body_w, body_h, 2.2, 2.2)

    shackle = QPainterPath()
    if locked:
        shackle.moveTo(size * 0.32, size * 0.48)
        shackle.quadTo(size * 0.32, size * 0.28, size * 0.5, size * 0.28)
        shackle.quadTo(size * 0.68, size * 0.28, size * 0.68, size * 0.48)
    else:
        shackle.moveTo(size * 0.36, size * 0.48)
        shackle.quadTo(size * 0.36, size * 0.28, size * 0.54, size * 0.28)
        shackle.quadTo(size * 0.7, size * 0.28, size * 0.7, size * 0.36)
    p.drawPath(shackle)
    p.end()
    return QIcon(pm)


def rename_icon(size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    # Clean pencil outline.
    p.drawLine(int(size * 0.24), int(size * 0.76), int(size * 0.68), int(size * 0.32))
    p.drawLine(int(size * 0.68), int(size * 0.32), int(size * 0.78), int(size * 0.42))
    p.drawLine(int(size * 0.78), int(size * 0.42), int(size * 0.34), int(size * 0.86))
    p.drawLine(int(size * 0.26), int(size * 0.84), int(size * 0.34), int(size * 0.86))
    p.drawLine(int(size * 0.22), int(size * 0.88), int(size * 0.28), int(size * 0.82))
    p.end()
    return QIcon(pm)


def delete_icon(size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    p.setPen(QPen(color, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(int(size * 0.3), int(size * 0.34), int(size * 0.4), int(size * 0.48))
    p.drawLine(int(size * 0.24), int(size * 0.3), int(size * 0.76), int(size * 0.3))
    p.drawLine(int(size * 0.42), int(size * 0.2), int(size * 0.58), int(size * 0.2))
    p.drawLine(int(size * 0.42), int(size * 0.44), int(size * 0.42), int(size * 0.74))
    p.drawLine(int(size * 0.5), int(size * 0.44), int(size * 0.5), int(size * 0.74))
    p.drawLine(int(size * 0.58), int(size * 0.44), int(size * 0.58), int(size * 0.74))
    p.end()
    return QIcon(pm)


def brush_icon(size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    p.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    # Minimal nib-pen style.
    path = QPainterPath()
    path.moveTo(size * 0.24, size * 0.76)
    path.lineTo(size * 0.6, size * 0.4)
    path.lineTo(size * 0.75, size * 0.55)
    path.lineTo(size * 0.39, size * 0.9)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(int(size * 0.52), int(size * 0.47), int(size * 0.68), int(size * 0.63))
    p.end()
    return QIcon(pm)


def eraser_icon(size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    p.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    # Parallelogram eraser body + baseline.
    p.drawLine(int(size * 0.2), int(size * 0.72), int(size * 0.46), int(size * 0.36))
    p.drawLine(int(size * 0.46), int(size * 0.36), int(size * 0.78), int(size * 0.62))
    p.drawLine(int(size * 0.78), int(size * 0.62), int(size * 0.52), int(size * 0.88))
    p.drawLine(int(size * 0.52), int(size * 0.88), int(size * 0.2), int(size * 0.72))
    p.drawLine(int(size * 0.52), int(size * 0.88), int(size * 0.84), int(size * 0.88))
    p.end()
    return QIcon(pm)


def clear_icon(size: int = 18, color: QColor | None = None) -> QIcon:
    color = color or QColor("#334e68")
    pm, p = _base_pixmap(size)
    p.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    # Bucket-style clear icon.
    p.drawLine(int(size * 0.25), int(size * 0.42), int(size * 0.58), int(size * 0.24))
    p.drawLine(int(size * 0.58), int(size * 0.24), int(size * 0.78), int(size * 0.44))
    p.drawLine(int(size * 0.78), int(size * 0.44), int(size * 0.46), int(size * 0.74))
    p.drawLine(int(size * 0.46), int(size * 0.74), int(size * 0.25), int(size * 0.42))
    p.drawLine(int(size * 0.18), int(size * 0.82), int(size * 0.82), int(size * 0.82))
    p.drawLine(int(size * 0.64), int(size * 0.64), int(size * 0.72), int(size * 0.72))
    p.end()
    return QIcon(pm)
