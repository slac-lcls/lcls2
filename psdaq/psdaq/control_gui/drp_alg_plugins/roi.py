"""ROI selection plugin over a detector geometry.

Intent:
    {"rois": [[r0, c0, r1, c1], ...]}
    - Return a list of half-open ROIs

ROIs are drawn anywhere inside the detector bounds.

They are allowed to cross between different segments of the detector.

The mapping of pixels and geometry is done by the DRP algorithm adapter.
"""

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from drpalgos.geometry import DetectorGeometry, Rect
from psdaq.control_gui.drp_alg_plugins import DrpAlgParamPlugin


def _qrect(r: Rect) -> QRectF:
    """Detector-frame (r0, c0, r1, c1) to a scene rectangle (x = col, y = row)."""
    return QRectF(r[1], r[0], r[3] - r[1], r[2] - r[0])


class _RoiView(QGraphicsView):
    def __init__(self, plugin: "RoiPlugin"):
        super().__init__(plugin.scene)
        self.plugin: "RoiPlugin" = plugin
        self._start: Optional[QPointF] = None
        self._band: Optional[QGraphicsRectItem] = None

    def mousePressEvent(self, e) -> None:
        bounds: Optional[QRectF] = self.plugin.bounds_rect()
        pos: QPointF = self.mapToScene(e.pos())
        if bounds is None or not bounds.contains(pos):
            return
        self._start = pos
        self._band = self.scene().addRect(
            QRectF(pos, pos), QPen(Qt.red, 1, Qt.DashLine)
        )

    def mouseMoveEvent(self, e) -> None:
        bounds: Optional[QRectF] = self.plugin.bounds_rect()
        if self._band is not None and bounds is not None:
            rect = QRectF(self._start, self.mapToScene(e.pos())).normalized()
            self._band.setRect(rect.intersected(bounds))

    def mouseReleaseEvent(self, e) -> None:
        if self._band is None:
            return
        rect: QRectF = self._band.rect()
        self.scene().removeItem(self._band)
        self._band = None
        self._start = None
        self.plugin.add_roi(rect)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)


class RoiPlugin(DrpAlgParamPlugin):
    intent_type = "roi"

    def __init__(self, options, parent=None):
        super().__init__(options, parent)
        self.max_rois: int = int(options.get("max_rois", 8))
        self.rois: List[Rect] = []
        self.geom: Optional[DetectorGeometry] = None

        self.scene: QGraphicsScene = QGraphicsScene(self)
        self.view: _RoiView = _RoiView(self)
        self.lst_rois: QListWidget = QListWidget()
        self.btn_remove: QPushButton = QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.on_remove)

        side: QVBoxLayout = QVBoxLayout()
        side.addWidget(self.lst_rois)
        side.addWidget(self.btn_remove)
        layout: QHBoxLayout = QHBoxLayout(self)
        layout.addWidget(self.view, 3)
        layout.addLayout(side, 1)
        self.setMinimumHeight(250)

    ### Detector Geometry ###
    #########################

    def set_geometry(self, geom: Optional[DetectorGeometry]) -> None:
        self.geom = geom
        self.redraw()

    def bounds_rect(self) -> Optional[QRectF]:
        if self.geom is None or not self.geom.segments:
            return None
        return _qrect(self.geom.bounds())

    #############################
    ### End Detector Geometry ###

    ### ROI Setup and Management ###
    ################################

    def add_roi(self, rect: QRectF) -> None:
        if len(self.rois) >= self.max_rois:
            return
        roi: Rect = (
            round(rect.top()),
            round(rect.left()),
            round(rect.bottom()),
            round(rect.right()),
        )
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            return
        self.rois.append(roi)
        self.redraw()
        self.valuesChanged.emit()

    def on_remove(self) -> None:
        rows: List[int] = sorted(
            (self.lst_rois.row(item) for item in self.lst_rois.selectedItems()),
            reverse=True,
        )
        if not rows:
            return
        for row in rows:
            del self.rois[row]
        self.redraw()
        self.valuesChanged.emit()

    def fit(self) -> None:
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def redraw(self) -> None:
        self.scene.clear()
        if self.geom is not None:
            for s in self.geom.segments:
                rect: QRectF = _qrect(self.geom.panel_rect(s.segment))
                self.scene.addRect(
                    rect, QPen(Qt.darkGray), QBrush(QColor(235, 235, 235))
                )
                label = self.scene.addSimpleText(str(s.segment))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                label.setPos(rect.topLeft() + QPointF(4, 4))
        self.lst_rois.clear()
        for i, (r0, c0, r1, c1) in enumerate(self.rois):
            self.scene.addRect(
                _qrect((r0, c0, r1, c1)), QPen(Qt.red, 1), QBrush(QColor(255, 0, 0, 50))
            )
            self.lst_rois.addItem(f"ROI {i}: rows [{r0}:{r1}) cols [{c0}:{c1})")
        self.fit()

    ####################################
    ### End ROI Setup and Management ###

    ### Intent IO ###
    #################

    def set_intent(self, intent: Dict[str, Any]) -> None:
        self.rois = [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in intent.get("rois", [])
        ]
        self.redraw()

    def intent(self) -> Dict[str, Any]:
        return {"rois": [list(r) for r in self.rois]}
