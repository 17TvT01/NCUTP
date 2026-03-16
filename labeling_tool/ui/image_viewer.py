import cv2
import numpy as np
from typing import List, Dict, Optional
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
from .viewer_painter import draw_viewer_content

class ImageViewer(QLabel):
    """Custom widget for displaying and annotating CT images. Focuses only on inputs/events."""

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.image: Optional[QPixmap] = None
        self.annotations: List[Dict] = []
        self.polygon_points: List[QPoint] = []
        self.base_scale = 1.0
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.draw_mode = "rectangle"

    @property
    def eff_scale(self) -> float:
        return self.base_scale * self.zoom

    def _update_base_scale(self):
        if not self.image: return
        ww, wh = self.width(), self.height()
        iw, ih = self.image.width(), self.image.height()
        if ww > 10 and wh > 10 and iw > 0 and ih > 0:
            self.base_scale = min(ww / iw, wh / ih)

    def _clamp_view(self):
        if not self.image: return
        eff = self.eff_scale
        iw, ih = self.image.width(), self.image.height()
        ww, wh = self.width(), self.height()
        vis_w, vis_h = ww / eff, wh / eff
        if vis_w >= iw:
            self.view_x = (iw - vis_w) / 2.0
        else:
            self.view_x = max(0.0, min(self.view_x, iw - vis_w))
        if vis_h >= ih:
            self.view_y = (ih - vis_h) / 2.0
        else:
            self.view_y = max(0.0, min(self.view_y, ih - vis_h))

    def _widget_to_img(self, wx: float, wy: float):
        eff = self.eff_scale
        return self.view_x + wx / eff, self.view_y + wy / eff

    def set_image(self, np_image: np.ndarray):
        img_normalized = ((np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-8) * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        h, w, ch = img_rgb.shape
        q_image = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image = QPixmap.fromImage(q_image)
        self.zoom = 1.0
        self._update_base_scale()
        self._clamp_view()
        self.update_display()

    def set_annotations(self, annotations: List[Dict]):
        self.annotations = annotations
        self.update_display()

    def set_draw_mode(self, mode: str):
        self.draw_mode = mode
        self.polygon_points = []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_base_scale()
        self._clamp_view()
        self.update_display()

    def wheelEvent(self, event):
        if not self.image: return
        mx, my = float(event.pos().x()), float(event.pos().y())
        ix, iy = self._widget_to_img(mx, my)
        STEP = 1.15
        if event.angleDelta().y() > 0:
            self.zoom = min(self.zoom * STEP, 20.0)
        else:
            self.zoom = max(self.zoom / STEP, 1.0)
        eff = self.eff_scale
        self.view_x = ix - mx / eff
        self.view_y = iy - my / eff
        self._clamp_view()
        self.update_display()
        event.accept()

    def update_display(self):
        if not self.image or self.width() <= 0 or self.height() <= 0: return
        out = draw_viewer_content(
            self.width(), self.height(), self.image, self.view_x, self.view_y, self.eff_scale,
            self.annotations, self.drawing, self.draw_mode, self.start_point, self.current_point, self.polygon_points
        )
        self.setPixmap(out)

    def _get_image_position(self, widget_pos: QPoint) -> Optional[QPoint]:
        if not self.image: return None
        ix, iy = self._widget_to_img(float(widget_pos.x()), float(widget_pos.y()))
        iw, ih = self.image.width(), self.image.height()
        return QPoint(int(max(0.0, min(ix, float(iw - 1)))), int(max(0.0, min(iy, float(ih - 1)))))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image:
            img_pos = self._get_image_position(event.pos())
            if not img_pos: return
            if self.draw_mode == "polygon":
                self.polygon_points.append(img_pos)
                self.drawing = True
                self.current_point = img_pos
                self.update_display()
            else:
                self.start_point = img_pos
                self.drawing = True

    def mouseMoveEvent(self, event):
        if self.image:
            img_pos = self._get_image_position(event.pos())
            if not img_pos: return
            self.current_point = img_pos
            if self.drawing:
                if self.draw_mode == "freehand":
                    self.polygon_points.append(self.current_point)
                self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            if self.draw_mode == "polygon": return
            self.drawing = False
            sp, cp = self.start_point, self.current_point
            if self.draw_mode == "rectangle" or self.draw_mode == "ellipse":
                x, y = min(sp.x(), cp.x()), min(sp.y(), cp.y())
                w, h = abs(cp.x() - sp.x()), abs(cp.y() - sp.y())
                if w > 5 and h > 5:
                    self.annotations.append({"shape": self.draw_mode, "bbox": (x, y, w, h)})
            elif self.draw_mode == "circle":
                r = int(np.sqrt((cp.x() - sp.x()) ** 2 + (cp.y() - sp.y()) ** 2))
                if r > 5:
                    self.annotations.append({"shape": "circle", "circle": (sp.x(), sp.y(), r)})
            elif self.draw_mode == "freehand":
                if len(self.polygon_points) > 5:
                    self.annotations.append({"shape": "freehand", "points": [(p.x(), p.y()) for p in self.polygon_points]})
                self.polygon_points = []
            self.update_display()

    def mouseDoubleClickEvent(self, event):
        if self.draw_mode == "polygon" and len(self.polygon_points) > 2:
            self.annotations.append({"shape": "polygon", "points": [(p.x(), p.y()) for p in self.polygon_points]})
            self.polygon_points, self.drawing = [], False
            self.update_display()

    def contextMenuEvent(self, event):
        self.mouseDoubleClickEvent(event)
