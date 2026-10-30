"""
PhantomTouch Custom UI Widgets.
Includes high-performance CameraViewWidget with aspect scaling and HUD overlay,
and modern StatusCard indicators.
"""

from typing import Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, QRect, QSize, Signal
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QFont,
    QPen,
    QBrush,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
)

from phantomtouch.core.state_manager import AppState


class CameraViewWidget(QWidget):
    """
    Renders live webcam video frames with aspect ratio preservation
    and non-intrusive status HUD overlays.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0b0c10; border-radius: 12px;")

        self._pixmap: Optional[QPixmap] = None
        self._state_text: str = "CAMERA INACTIVE"
        self._state_color: QColor = QColor("#64748b")
        self._fps_text: str = "0.0 FPS"
        self._gesture_msg: str = "Start camera to begin"
        self._is_active: bool = False

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        state: AppState,
        fps: float,
        gesture_msg: str,
    ) -> None:
        """Receive a new frame from worker and request repaint."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w

        # Convert OpenCV BGR to RGB for Qt
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_img)
        self._is_active = True

        self._fps_text = f"{fps:.1f} FPS"
        self._gesture_msg = gesture_msg or ""

        # Map state to display text and accent color
        if state == AppState.INVISIBLE:
            self._state_text = "INVISIBILITY ACTIVE"
            self._state_color = QColor("#10b981")  # Emerald
        elif state == AppState.GESTURE_DETECTED:
            self._state_text = "GESTURE DETECTED"
            self._state_color = QColor("#6366f1")  # Indigo
        elif state == AppState.INVISIBILITY_ACTIVATING:
            self._state_text = "ACTIVATING INVISIBILITY"
            self._state_color = QColor("#8b5cf6")  # Purple
        elif state == AppState.INVISIBILITY_DEACTIVATING:
            self._state_text = "HOLDING EFFECT"
            self._state_color = QColor("#f59e0b")  # Amber
        elif state == AppState.BACKGROUND_REQUIRED:
            self._state_text = "BACKGROUND REQUIRED"
            self._state_color = QColor("#ef4444")  # Red
        elif state == AppState.NORMAL:
            self._state_text = "MONITORING FOR GESTURE"
            self._state_color = QColor("#3b82f6")  # Blue
        elif state == AppState.CAMERA_ERROR:
            self._state_text = "CAMERA ERROR"
            self._state_color = QColor("#dc2626")
        else:
            self._state_text = "INITIALIZING"
            self._state_color = QColor("#64748b")

        self.update()

    def set_inactive(self, message: str = "Camera Stopped") -> None:
        """Clear active frame and show placeholder message."""
        self._is_active = False
        self._pixmap = None
        self._state_text = "OFFLINE"
        self._state_color = QColor("#475569")
        self._gesture_msg = message
        self.update()

    def paintEvent(self, event) -> None:
        """Render frame scaled with aspect ratio and draw overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()

        # Draw dark background
        painter.fillRect(rect, QColor("#0b0c10"))

        if self._is_active and self._pixmap and not self._pixmap.isNull():
            # Scale pixmap preserving aspect ratio
            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            # Center in widget
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw HUD Overlay
            self._draw_hud(painter, rect, x, y, scaled.width(), scaled.height())
        else:
            # Draw offline placeholder
            self._draw_placeholder(painter, rect)

    def _draw_hud(self, painter: QPainter, rect: QRect, vid_x: int, vid_y: int, vid_w: int, vid_h: int) -> None:
        """Draw HUD tags on top of video."""
        # State Pill Tag (Top Left)
        tag_margin = 16
        tag_x = vid_x + tag_margin
        tag_y = vid_y + tag_margin

        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        state_metrics = painter.fontMetrics()
        text_w = state_metrics.horizontalAdvance(self._state_text)
        pill_rect = QRect(tag_x, tag_y, text_w + 24, 28)

        # Pill background
        path = QPainterPath()
        path.addRoundedRect(pill_rect, 14, 14)
        painter.fillPath(path, QBrush(QColor(15, 23, 42, 210)))

        # Dot indicator
        painter.setBrush(QBrush(self._state_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(tag_x + 8, tag_y + 9, 10, 10)

        # State text
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(tag_x + 24, tag_y + 19, self._state_text)

        # FPS Tag (Top Right)
        fps_metrics = painter.fontMetrics()
        fps_w = fps_metrics.horizontalAdvance(self._fps_text)
        fps_rect = QRect(vid_x + vid_w - fps_w - tag_margin - 20, tag_y, fps_w + 20, 28)
        fps_path = QPainterPath()
        fps_path.addRoundedRect(fps_rect, 14, 14)
        painter.fillPath(fps_path, QBrush(QColor(15, 23, 42, 210)))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(fps_rect.x() + 10, tag_y + 19, self._fps_text)

        # Guidance Message Banner (Bottom Center)
        if self._gesture_msg:
            sub_font = QFont("Segoe UI", 11, QFont.DemiBold)
            painter.setFont(sub_font)
            metrics = painter.fontMetrics()
            msg_w = metrics.horizontalAdvance(self._gesture_msg)
            msg_x = vid_x + (vid_w - msg_w - 32) // 2
            msg_y = vid_y + vid_h - 48
            banner_rect = QRect(msg_x, msg_y, msg_w + 32, 34)

            banner_path = QPainterPath()
            banner_path.addRoundedRect(banner_rect, 17, 17)
            painter.fillPath(banner_path, QBrush(QColor(15, 23, 42, 220)))

            painter.setPen(QColor("#f1f5f9"))
            painter.drawText(msg_x + 16, msg_y + 22, self._gesture_msg)

    def _draw_placeholder(self, painter: QPainter, rect: QRect) -> None:
        """Render stylish placeholder when camera is not running."""
        painter.setPen(QColor("#475569"))
        font_title = QFont("Segoe UI", 16, QFont.Bold)
        painter.setFont(font_title)
        painter.drawText(rect, Qt.AlignCenter, "PhantomTouch Ready\n\nClick 'Start Camera' to begin")


class StatusCard(QFrame):
    """A card showing an icon, title, current status value, and dynamic color."""

    def __init__(self, title: str, initial_value: str = "Unknown", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._title_label = QLabel(title.upper())
        self._title_label.setProperty("class", "section-header")

        self._value_layout = QHBoxLayout()
        self._value_layout.setSpacing(8)

        self._indicator = QLabel()
        self._indicator.setFixedSize(10, 10)
        self._indicator.setStyleSheet("background-color: #64748b; border-radius: 5px;")

        self._value_label = QLabel(initial_value)
        self._value_label.setProperty("class", "status-value")

        self._value_layout.addWidget(self._indicator)
        self._value_layout.addWidget(self._value_label)
        self._value_layout.addStretch()

        self._desc_label = QLabel("")
        self._desc_label.setProperty("class", "subtext")

        layout.addWidget(self._title_label)
        layout.addLayout(self._value_layout)
        layout.addWidget(self._desc_label)

    def set_status(self, text: str, color_hex: str = "#64748b", description: str = "") -> None:
        """Update text, indicator color, and optional description."""
        self._value_label.setText(text)
        self._indicator.setStyleSheet(f"background-color: {color_hex}; border-radius: 5px;")
        if description:
            self._desc_label.setText(description)
            self._desc_label.setVisible(True)
        else:
            self._desc_label.setVisible(False)
