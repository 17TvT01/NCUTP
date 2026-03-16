import numpy as np
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF
from typing import List, Dict, Tuple, Optional

def draw_viewer_content(
    ww: int, wh: int,
    image: Optional[QPixmap],
    view_x: float, view_y: float,
    eff: float,
    annotations: List[Dict],
    drawing: bool,
    draw_mode: str,
    start_point: Optional[QPointF],
    current_point: Optional[QPointF],
    polygon_points: List[QPointF]
) -> QPixmap:
    """Hàm độc lập để kết xuất (render) hình ảnh và các nhãn đánh dấu lên màn hình"""
    out = QPixmap(ww, wh)
    out.fill(QColor(0, 0, 0))
    if not image:
        return out
        
    painter = QPainter(out)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)

    iw, ih = image.width(), image.height()
    src_x1 = max(0.0, view_x)
    src_y1 = max(0.0, view_y)
    src_x2 = min(float(iw), view_x + ww / eff)
    src_y2 = min(float(ih), view_y + wh / eff)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        dx1 = (src_x1 - view_x) * eff
        dy1 = (src_y1 - view_y) * eff
        dx2 = (src_x2 - view_x) * eff
        dy2 = (src_y2 - view_y) * eff
        painter.drawPixmap(
            QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1),
            image,
            QRectF(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1),
        )

    def wpt(ix_: float, iy_: float) -> QPointF:
        wx = (ix_ - view_x) * eff
        wy = (iy_ - view_y) * eff
        return QPointF(wx, wy)

    # --- Vẽ các nhãn đã lưu ---
    painter.setPen(QPen(QColor(0, 255, 0), 2))
    for ann in annotations:
        shape = ann.get("shape", "rectangle")
        if shape == "rectangle" and ann.get("bbox"):
            x, y, bw, bh = ann["bbox"]
            p = wpt(x, y)
            painter.drawRect(QRectF(p.x(), p.y(), bw * eff, bh * eff))
        elif shape == "circle" and ann.get("circle"):
            cx, cy, r = ann["circle"]
            painter.drawEllipse(wpt(cx, cy), r * eff, r * eff)
        elif shape == "ellipse" and ann.get("bbox"):
            x, y, bw, bh = ann["bbox"]
            p = wpt(x, y)
            painter.drawEllipse(QRectF(p.x(), p.y(), bw * eff, bh * eff))
        elif shape in ("polygon", "freehand") and ann.get("points"):
            pts = QPolygonF([wpt(p[0], p[1]) for p in ann["points"]])
            if shape == "polygon":
                painter.drawPolygon(pts)
            else:
                painter.drawPolyline(pts)

    # --- Vẽ hình khối đang nằm trong quá trình phác đường ---
    if (drawing or polygon_points) and start_point and current_point:
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        sp, cp = start_point, current_point
        wsp = wpt(sp.x(), sp.y())
        wcp = wpt(cp.x(), cp.y())
        
        if draw_mode == "rectangle":
            painter.drawRect(QRectF(
                min(wsp.x(), wcp.x()), min(wsp.y(), wcp.y()),
                abs(wcp.x() - wsp.x()), abs(wcp.y() - wsp.y()),
            ))
        elif draw_mode == "circle":
            r_img = np.sqrt((cp.x() - sp.x()) ** 2 + (cp.y() - sp.y()) ** 2)
            painter.drawEllipse(wsp, r_img * eff, r_img * eff)
        elif draw_mode == "ellipse":
            painter.drawEllipse(QRectF(
                min(wsp.x(), wcp.x()), min(wsp.y(), wcp.y()),
                abs(wcp.x() - wsp.x()), abs(wcp.y() - wsp.y()),
            ))
        elif draw_mode == "polygon" and polygon_points:
            wpts = [wpt(p.x(), p.y()) for p in polygon_points]
            if len(wpts) > 1:
                painter.drawPolyline(QPolygonF(wpts))
            if drawing:
                painter.drawLine(wpts[-1], wcp)
        elif draw_mode == "freehand" and polygon_points:
            painter.drawPolyline(QPolygonF([wpt(p.x(), p.y()) for p in polygon_points]))

    painter.end()
    return out
