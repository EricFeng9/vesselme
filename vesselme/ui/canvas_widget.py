from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from vesselme.core.history import HistoryStack, StrokePatch
from vesselme.utils.qt import rgb_ndarray_to_qpixmap


@dataclass
class StrokeState:
    active: bool = False
    mode: str = "brush"
    min_x: int = 10**9
    min_y: int = 10**9
    max_x: int = -1
    max_y: int = -1

    def reset(self) -> None:
        self.active = False
        self.mode = "brush"
        self.min_x = 10**9
        self.min_y = 10**9
        self.max_x = -1
        self.max_y = -1


class CanvasWidget(QWidget):
    zoomChanged = Signal(float)
    brushChanged = Signal(float)
    message = Signal(str)
    dirtyChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.image_rgb: np.ndarray | None = None
        self.image_pixmap: QPixmap | None = None
        self.mask: np.ndarray | None = None
        self.overlay_color = (255, 255, 255)
        self.overlay_alpha = 204
        self.overlay_rgba: np.ndarray | None = None
        self.overlay_qimage: QImage | None = None

        self.overlay_visible = True
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)

        self.brush_size = 12.0
        self.current_tool = "brush"
        self.editable = True

        self.history = HistoryStack(capacity=100)
        self.stroke_state = StrokeState()
        self.stroke_before: np.ndarray | None = None

        self.last_image_pos: tuple[int, int] | None = None
        self.last_image_pos_float: tuple[float, float] | None = None
        self.ctrl_resize_active = False
        self.ctrl_resize_anchor_x = 0
        self.ctrl_resize_base = self.brush_size
        self.stroke_last_point: tuple[float, float] | None = None

        self.pan_active = False
        self.drag_anchor = QPoint()
        self._sync_cursor()

    def has_content(self) -> bool:
        return self.image_rgb is not None

    def set_scene(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        overlay_color: tuple[int, int, int],
        preserve_view: bool = False,
    ) -> None:
        keep_view = (
            preserve_view
            and self.image_rgb is not None
            and self.image_rgb.shape[:2] == image_rgb.shape[:2]
        )
        self.image_rgb = image_rgb
        self.image_pixmap = rgb_ndarray_to_qpixmap(image_rgb)
        self.mask = mask
        self.overlay_color = overlay_color
        self._rebuild_overlay_cache()
        self.history.clear()
        if not keep_view:
            self.scale = 1.0
            self.offset = QPointF(0.0, 0.0)
            self.fit_to_window()
        self.update()

    def set_image_preview(self, image_rgb: np.ndarray) -> None:
        self.image_rgb = image_rgb
        self.image_pixmap = rgb_ndarray_to_qpixmap(image_rgb)
        self.mask = None
        self.overlay_rgba = None
        self.overlay_qimage = None
        self.history.clear()
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.fit_to_window()
        self.update()

    def set_overlay_color(self, color: tuple[int, int, int]) -> None:
        self.overlay_color = color
        self._rebuild_overlay_cache()
        self.update()

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        self._sync_cursor()

    def set_brush_size(self, size: float) -> None:
        value = max(0.5, min(50.0, round(float(size) * 2.0) / 2.0))
        if abs(value - self.brush_size) < 1e-6:
            return
        self.brush_size = value
        self.brushChanged.emit(self.brush_size)
        self.update()

    def set_editable(self, editable: bool) -> None:
        self.editable = editable
        self._sync_cursor()

    def set_overlay_visible(self, visible: bool) -> None:
        self.overlay_visible = visible
        self.update()

    def set_overlay_opacity(self, percent: float) -> None:
        p = max(0.0, min(100.0, float(percent)))
        self.overlay_alpha = int(round(p * 255.0 / 100.0))
        self._rebuild_overlay_cache()
        self.update()

    def get_overlay_opacity(self) -> float:
        return (self.overlay_alpha / 255.0) * 100.0

    def fit_to_window(self) -> None:
        if self.image_rgb is None:
            return
        ih, iw = self.image_rgb.shape[:2]
        if iw <= 0 or ih <= 0:
            return
        sx = max(self.width() - 20, 1) / iw
        sy = max(self.height() - 20, 1) / ih
        self.scale = min(sx, sy)
        self._center_image()
        self.zoomChanged.emit(self.scale)

    def actual_size(self) -> None:
        self.scale = 1.0
        self._center_image()
        self.zoomChanged.emit(self.scale)
        self.update()

    def undo(self) -> bool:
        if self.mask is None or not self.editable:
            return False
        ok = self.history.undo(self.mask)
        if ok:
            self._rebuild_overlay_cache()
            self.update()
            self.dirtyChanged.emit()
        return ok

    def redo(self) -> bool:
        if self.mask is None or not self.editable:
            return False
        ok = self.history.redo(self.mask)
        if ok:
            self._rebuild_overlay_cache()
            self.update()
            self.dirtyChanged.emit()
        return ok

    def clear_mask(self) -> bool:
        if self.mask is None or not self.editable:
            return False
        before = self.mask.copy()
        self.mask.fill(0)
        if self.overlay_rgba is not None:
            self.overlay_rgba.fill(0)
        patch = StrokePatch(0, self.mask.shape[0], 0, self.mask.shape[1], before=before, after=self.mask.copy())
        self.history.push(patch)
        self.update()
        self.dirtyChanged.emit()
        return True

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        if self.image_rgb is None:
            return
        factor = max(0.05, min(10.0, factor))
        old_scale = self.scale
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)

        img_before = self._widget_to_image_float(anchor)
        self.scale = factor
        img_after = self._widget_to_image_float(anchor)
        dx = (img_after.x() - img_before.x()) * self.scale
        dy = (img_after.y() - img_before.y()) * self.scale
        self.offset = QPointF(self.offset.x() + dx, self.offset.y() + dy)

        if abs(old_scale - self.scale) > 1e-5:
            self.zoomChanged.emit(self.scale)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.image_rgb is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            # Fine-grained brush stepping in vessel annotation range.
            base_step = 0.5 if self.brush_size <= 10.0 else 1.0
            step = base_step if delta > 0 else -base_step
            self.set_brush_size(self.brush_size + step)
            self.message.emit(f"Brush size: {self.brush_size}")
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1.1 if delta > 0 else 0.9
        self.zoom_by(self.scale * step, anchor=event.position())

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(18, 25, 34))

        if self.image_pixmap is None or self.image_rgb is None:
            p.setPen(QColor(212, 223, 235))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open Folder to start annotation\n\nB: Brush  E: Eraser  A: Toggle Overlay",
            )
            return

        ih, iw = self.image_rgb.shape[:2]
        target = QRectF(self.offset.x(), self.offset.y(), iw * self.scale, ih * self.scale)
        p.drawPixmap(target, self.image_pixmap, QRectF(0, 0, iw, ih))

        if self.mask is not None and self.overlay_visible:
            if self.overlay_qimage is None:
                self._rebuild_overlay_cache()
            if self.overlay_qimage is not None:
                p.drawImage(target, self.overlay_qimage, QRectF(0, 0, iw, ih))

        if self.last_image_pos_float is not None:
            cx, cy = self.last_image_pos_float
            wx, wy = self._image_to_widget_float(cx, cy)
            pen = QPen(QColor(255, 255, 255, 220))
            pen.setWidth(1)
            p.setPen(pen)
            p.drawEllipse(QPointF(wx, wy), self.brush_size * self.scale / 2.0, self.brush_size * self.scale / 2.0)

    def _rebuild_overlay_cache(self) -> None:
        if self.mask is None:
            self.overlay_rgba = None
            self.overlay_qimage = None
            return

        h, w = self.mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        fg = self.mask > 0
        rgba[fg, 0] = self.overlay_color[0]
        rgba[fg, 1] = self.overlay_color[1]
        rgba[fg, 2] = self.overlay_color[2]
        rgba[fg, 3] = self.overlay_alpha
        self.overlay_rgba = rgba
        self.overlay_qimage = QImage(
            self.overlay_rgba.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_RGBA8888,
        )

    def _update_overlay_region(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.mask is None or self.overlay_rgba is None:
            return
        if x1 < x0 or y1 < y0:
            return
        region_mask = self.mask[y0 : y1 + 1, x0 : x1 + 1] > 0
        region_rgba = self.overlay_rgba[y0 : y1 + 1, x0 : x1 + 1]
        region_rgba[:, :, :] = 0
        region_rgba[region_mask, 0] = self.overlay_color[0]
        region_rgba[region_mask, 1] = self.overlay_color[1]
        region_rgba[region_mask, 2] = self.overlay_color[2]
        region_rgba[region_mask, 3] = self.overlay_alpha

    def _widget_to_image(self, pos: QPointF) -> tuple[int, int] | None:
        if self.image_rgb is None:
            return None
        ix = int((pos.x() - self.offset.x()) / self.scale)
        iy = int((pos.y() - self.offset.y()) / self.scale)
        h, w = self.image_rgb.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def _widget_to_image_float(self, pos: QPointF) -> QPointF:
        return QPointF((pos.x() - self.offset.x()) / self.scale, (pos.y() - self.offset.y()) / self.scale)

    def _widget_to_image_float_clamped(self, pos: QPointF) -> tuple[float, float] | None:
        if self.image_rgb is None:
            return None
        fx = (pos.x() - self.offset.x()) / self.scale
        fy = (pos.y() - self.offset.y()) / self.scale
        h, w = self.image_rgb.shape[:2]
        if fx < 0 or fy < 0 or fx >= w or fy >= h:
            return None
        return fx, fy

    def _image_to_widget(self, ix: int, iy: int) -> tuple[float, float]:
        return self.offset.x() + ix * self.scale, self.offset.y() + iy * self.scale

    def _image_to_widget_float(self, ix: float, iy: float) -> tuple[float, float]:
        return self.offset.x() + ix * self.scale, self.offset.y() + iy * self.scale

    def _center_image(self) -> None:
        if self.image_rgb is None:
            return
        ih, iw = self.image_rgb.shape[:2]
        x = (self.width() - iw * self.scale) / 2.0
        y = (self.height() - ih * self.scale) / 2.0
        self.offset = QPointF(x, y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.has_content():
            return

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._is_space_pressed()
        ):
            self.pan_active = True
            self.drag_anchor = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.ctrl_resize_active = True
            self.ctrl_resize_anchor_x = event.pos().x()
            self.ctrl_resize_base = self.brush_size
            return

        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return

        if self.mask is None:
            self.message.emit("No label selected. Create or select a label first.")
            return

        if not self.editable:
            self.message.emit("Current label is locked.")
            return

        image_pos = self._widget_to_image_float_clamped(event.position())
        if image_pos is None:
            return

        self.stroke_state.active = True
        self.stroke_state.mode = (
            "eraser"
            if event.button() == Qt.MouseButton.RightButton
            or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            or self.current_tool == "eraser"
            else "brush"
        )
        self.stroke_before = self.mask.copy() if self.mask is not None else None
        self.stroke_last_point = image_pos
        self._apply_stroke_segment(image_pos, image_pos, self.stroke_state.mode)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.has_content():
            return

        pos = event.position()
        image_pos = self._widget_to_image_float_clamped(pos)
        self.last_image_pos_float = image_pos
        if image_pos is not None:
            self.last_image_pos = (int(image_pos[0]), int(image_pos[1]))
        else:
            self.last_image_pos = None

        if self.ctrl_resize_active:
            delta = event.pos().x() - self.ctrl_resize_anchor_x
            new_size = self.ctrl_resize_base + (delta / 10.0)
            if new_size != self.brush_size:
                self.set_brush_size(new_size)
            self.update()
            return

        if self.pan_active:
            delta = event.pos() - self.drag_anchor
            self.offset = QPointF(self.offset.x() + delta.x(), self.offset.y() + delta.y())
            self.drag_anchor = event.pos()
            self.update()
            return

        if self.stroke_state.active and image_pos is not None and self.stroke_last_point is not None:
            self._apply_stroke_segment(self.stroke_last_point, image_pos, self.stroke_state.mode)
            self.stroke_last_point = image_pos
            self.update()
            return

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.ctrl_resize_active and event.button() == Qt.MouseButton.LeftButton:
            self.ctrl_resize_active = False
            self._sync_cursor()
            return
        if self.pan_active and (
            event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton
        ):
            self.pan_active = False
            self._sync_cursor()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self.stroke_state.active:
            self._finish_stroke()

    def leaveEvent(self, event) -> None:
        del event
        self.last_image_pos = None
        self.last_image_pos_float = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.image_rgb is not None and self.scale <= 0.2:
            self.fit_to_window()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_BracketLeft:
            self.set_brush_size(self.brush_size - 0.5)
            return
        if event.key() == Qt.Key.Key_BracketRight:
            self.set_brush_size(self.brush_size + 0.5)
            return
        super().keyPressEvent(event)

    def _apply_stroke_segment(self, start: tuple[float, float], end: tuple[float, float], mode: str) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = float(np.hypot(dx, dy))
        # Dense sampling keeps thin vessel strokes continuous and centered.
        steps = max(1, int(np.ceil(dist / 0.5)))
        for i in range(steps + 1):
            t = i / steps
            px = start[0] + dx * t
            py = start[1] + dy * t
            self._stamp_brush(px, py, mode)

    def _stamp_brush(self, x: float, y: float, mode: str) -> None:
        if self.mask is None:
            return
        value = 0 if mode == "eraser" else 255
        h, w = self.mask.shape
        radius = max(float(self.brush_size) / 2.0, 0.5)

        x0 = max(0, int(np.floor(x - radius)))
        y0 = max(0, int(np.floor(y - radius)))
        x1 = min(w - 1, int(np.ceil(x + radius)))
        y1 = min(h - 1, int(np.ceil(y + radius)))
        if x1 < x0 or y1 < y0:
            return

        xs = (np.arange(x0, x1 + 1, dtype=np.float32) + 0.5)[None, :]
        ys = (np.arange(y0, y1 + 1, dtype=np.float32) + 0.5)[:, None]
        footprint = (xs - x) ** 2 + (ys - y) ** 2 <= radius * radius
        region = self.mask[y0 : y1 + 1, x0 : x1 + 1]
        region[footprint] = value
        self._update_overlay_region(x0, y0, x1, y1)

        self.stroke_state.min_x = min(self.stroke_state.min_x, x0)
        self.stroke_state.min_y = min(self.stroke_state.min_y, y0)
        self.stroke_state.max_x = max(self.stroke_state.max_x, x1)
        self.stroke_state.max_y = max(self.stroke_state.max_y, y1)

    def _finish_stroke(self) -> None:
        if self.mask is None or self.stroke_before is None:
            self.stroke_state.reset()
            return

        h, w = self.mask.shape
        x0 = max(0, self.stroke_state.min_x)
        y0 = max(0, self.stroke_state.min_y)
        x1 = min(w, self.stroke_state.max_x + 1)
        y1 = min(h, self.stroke_state.max_y + 1)

        if x1 > x0 and y1 > y0:
            before = self.stroke_before[y0:y1, x0:x1].copy()
            after = self.mask[y0:y1, x0:x1].copy()
            if not np.array_equal(before, after):
                self.history.push(StrokePatch(y0=y0, y1=y1, x0=x0, x1=x1, before=before, after=after))
                self.dirtyChanged.emit()

        self.stroke_before = None
        self.stroke_last_point = None
        self.stroke_state.reset()
        self.update()

    def _is_space_pressed(self) -> bool:
        return bool(self.window() and self.window().property("space_pressed"))

    def _sync_cursor(self) -> None:
        if self.pan_active:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if not self.editable:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self.current_tool in {"brush", "eraser"}:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        self.setCursor(Qt.CursorShape.ArrowCursor)
