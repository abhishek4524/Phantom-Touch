"""
PhantomTouch Desktop Application Main Window.
Modern dark UI with live video preview, control panels, interactive calibration,
and real-time gesture telemetry.
"""

from typing import Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QGroupBox,
    QFrame,
    QMessageBox,
    QScrollArea,
)

from phantomtouch.camera.webcam_manager import WebcamManager
from phantomtouch.camera.frame_worker import FrameWorker
from phantomtouch.core.state_manager import StateManager, AppState
from phantomtouch.core.background_manager import BackgroundManager
from phantomtouch.core.compositor import VideoCompositor
from phantomtouch.virtual_camera.virtual_camera_manager import VirtualCameraManager
from phantomtouch.vision.gesture_detector import GestureResult
from phantomtouch.ui.styles import DARK_THEME_QSS
from phantomtouch.ui.widgets import CameraViewWidget, StatusCard
from phantomtouch.utils.config import AppConfig
from phantomtouch.utils.model_manager import ModelManager
from phantomtouch.utils.logger import get_logger

logger = get_logger("MainWindow")


class WorkerSignals(QObject):
    """Bridge for safely dispatching worker callbacks to Qt main thread."""
    frame_ready = Signal(object, object, object, float)  # frame, gesture_result, state, fps
    error_occurred = Signal(str)


class MainWindow(QMainWindow):
    """Primary application window for PhantomTouch."""

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig.load()

        self.setWindowTitle("PhantomTouch — Gesture-Controlled Invisibility")
        self.resize(1150, 720)
        self.setMinimumSize(960, 600)
        self.setStyleSheet(DARK_THEME_QSS)

        # Core subsystems
        self.state_mgr = StateManager(
            activation_frames=self.config.gesture_activation_frames,
            hold_frames=self.config.gesture_hold_frames,
            on_state_change=self._on_state_changed,
        )
        self.background_mgr = BackgroundManager(self.config.background_path)
        self.compositor = VideoCompositor()
        self.webcam_mgr = WebcamManager(
            camera_index=self.config.camera_index,
            width=self.config.frame_width,
            height=self.config.frame_height,
        )
        self.vcam_mgr = VirtualCameraManager(
            width=self.config.frame_width,
            height=self.config.frame_height,
            fps=self.config.target_fps,
        )

        # Worker & signals
        self.signals = WorkerSignals()
        self.signals.frame_ready.connect(self._on_frame_received)
        self.signals.error_occurred.connect(self._on_error_received)

        self.worker = FrameWorker(
            config=self.config,
            webcam=self.webcam_mgr,
            background_mgr=self.background_mgr,
            state_mgr=self.state_mgr,
            compositor=self.compositor,
            on_frame_ready=self._emit_frame,
            on_error=self._emit_error,
        )

        self._init_ui()
        self._check_models_at_startup()

    def _init_ui(self) -> None:
        """Construct user interface layout."""
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # LEFT SIDE: Video Preview & Quick Controls
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # Header Title
        header_layout = QHBoxLayout()
        app_title = QLabel("PhantomTouch")
        app_title.setProperty("class", "header-title")
        app_sub = QLabel("Gesture-Controlled Real-Time Webcam Invisibility")
        app_sub.setProperty("class", "subtext")
        header_layout.addWidget(app_title)
        header_layout.addSpacing(12)
        header_layout.addWidget(app_sub)
        header_layout.addStretch()

        left_layout.addLayout(header_layout)

        # Live Camera View Widget
        self.camera_view = CameraViewWidget()
        left_layout.addWidget(self.camera_view, stretch=1)

        # Bottom Action Bar (Camera selection & Background capture)
        action_bar = QFrame()
        action_bar.setProperty("class", "card")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(10)

        # Camera selection dropdown
        cam_label = QLabel("Camera:")
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(140)
        self._populate_cameras()
        self.cam_combo.currentIndexChanged.connect(self._on_camera_selected)

        # Start / Stop Buttons
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setProperty("class", "btn-primary")
        self.btn_start.clicked.connect(self.start_pipeline)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)

        # Background Capture Actions
        self.btn_capture_bg = QPushButton("Capture Background")
        self.btn_capture_bg.setProperty("class", "btn-primary")
        self.btn_capture_bg.clicked.connect(self._on_capture_bg_clicked)

        self.btn_recal_bg = QPushButton("Clear Background")
        self.btn_recal_bg.clicked.connect(self._on_clear_bg_clicked)

        action_layout.addWidget(cam_label)
        action_layout.addWidget(self.cam_combo)
        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(self.btn_stop)
        action_layout.addSpacing(16)
        action_layout.addWidget(self.btn_capture_bg)
        action_layout.addWidget(self.btn_recal_bg)
        action_layout.addStretch()

        left_layout.addWidget(action_bar)
        main_layout.addLayout(left_layout, stretch=7)

        # RIGHT SIDE: Telemetry & Controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)

        panel_content = QWidget()
        right_layout = QVBoxLayout(panel_content)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(12)

        # Status Cards
        self.card_state = StatusCard("Operating Mode", "CAMERA_INITIALIZING")
        self.card_camera = StatusCard("Camera Status", "Disconnected")
        self.card_bg = StatusCard("Background", "Not Captured")
        self.card_gesture = StatusCard("Gesture Detection", "Monitoring")

        right_layout.addWidget(self.card_state)
        right_layout.addWidget(self.card_camera)
        right_layout.addWidget(self.card_bg)
        right_layout.addWidget(self.card_gesture)

        # Controls & Settings Group
        controls_group = QFrame()
        controls_group.setProperty("class", "card")
        ctrl_layout = QVBoxLayout(controls_group)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_layout.setSpacing(10)

        ctrl_title = QLabel("PIPELINE CONTROLS")
        ctrl_title.setProperty("class", "section-header")
        ctrl_layout.addWidget(ctrl_title)

        # Manual Invisibility Toggle
        self.chk_manual = QCheckBox("Manual Invisibility Override")
        self.chk_manual.toggled.connect(self._on_manual_toggled)
        ctrl_layout.addWidget(self.chk_manual)

        # Mirror Preview
        self.chk_mirror = QCheckBox("Mirror Camera Preview")
        self.chk_mirror.setChecked(self.config.mirror_preview)
        self.chk_mirror.toggled.connect(self._on_mirror_toggled)
        ctrl_layout.addWidget(self.chk_mirror)

        # Virtual Camera Output
        self.chk_vcam = QCheckBox("Virtual Camera Output (Zoom/OBS)")
        self.chk_vcam.setChecked(False)
        self.chk_vcam.toggled.connect(self._on_vcam_toggled)
        ctrl_layout.addWidget(self.chk_vcam)

        ctrl_layout.addSpacing(8)

        # Gesture Sensitivity Slider
        sens_label = QLabel("Forehead Distance Tolerance:")
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setRange(20, 70)  # 0.20 to 0.70
        self.slider_sens.setValue(int(self.config.forehead_distance_threshold * 100))
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        ctrl_layout.addWidget(sens_label)
        ctrl_layout.addWidget(self.slider_sens)

        # Segmentation Threshold Slider
        seg_label = QLabel("Person Segmentation Threshold:")
        self.slider_seg = QSlider(Qt.Horizontal)
        self.slider_seg.setRange(20, 80)  # 0.20 to 0.80
        self.slider_seg.setValue(int(self.config.seg_threshold * 100))
        self.slider_seg.valueChanged.connect(self._on_seg_changed)
        ctrl_layout.addWidget(seg_label)
        ctrl_layout.addWidget(self.slider_seg)

        # Edge Feathering Slider
        feather_label = QLabel("Edge Feathering (Smoothness):")
        self.slider_feather = QSlider(Qt.Horizontal)
        self.slider_feather.setRange(1, 25)  # pixels
        self.slider_feather.setValue(self.config.seg_feather_radius)
        self.slider_feather.valueChanged.connect(self._on_feather_changed)
        ctrl_layout.addWidget(feather_label)
        ctrl_layout.addWidget(self.slider_feather)

        right_layout.addWidget(controls_group)
        right_layout.addStretch()

        scroll_area.setWidget(panel_content)
        main_layout.addWidget(scroll_area, stretch=3)

        self._update_status_display()

    def _populate_cameras(self) -> None:
        """Scan available camera devices."""
        cams = self.webcam_mgr.list_available_cameras()
        self.cam_combo.clear()
        if not cams:
            self.cam_combo.addItem("Camera 0 (Default)", 0)
        else:
            for idx in cams:
                self.cam_combo.addItem(f"Camera {idx}", idx)

    def _check_models_at_startup(self) -> None:
        """Verify models exist and guide user if missing."""
        valid, missing = ModelManager.check_models()
        if not valid:
            missing_text = "\n".join(f"- {m}" for m in missing)
            QMessageBox.warning(
                self,
                "MediaPipe Models Required",
                f"The following required vision models are missing or corrupted:\n\n{missing_text}\n\n"
                "Please run:\n  python scripts/download_models.py\n\nor click OK to download automatically."
            )

    def start_pipeline(self) -> None:
        """Start camera capture and vision worker."""
        # Initialize models if needed
        if not self.worker.initialize_models():
            QMessageBox.critical(self, "Initialization Error", "Failed to initialize vision models.")
            return

        if self.worker.start():
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.cam_combo.setEnabled(False)
            self.card_camera.set_status("Active", "#10b981", f"Device {self.webcam_mgr.camera_index}")
        else:
            QMessageBox.critical(self, "Camera Error", "Unable to open webcam.")

    def stop_pipeline(self) -> None:
        """Stop worker and camera."""
        self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cam_combo.setEnabled(True)
        self.camera_view.set_inactive("Camera Stopped")
        self.card_camera.set_status("Disconnected", "#64748b")
        self._update_status_display()

    def _emit_frame(self, frame, gesture_res, state, fps):
        self.signals.frame_ready.emit(frame, gesture_res, state, fps)

    def _emit_error(self, err_msg):
        self.signals.error_occurred.emit(err_msg)

    def _on_frame_received(
        self,
        frame: np.ndarray,
        gesture_res: Optional[GestureResult],
        state: AppState,
        fps: float,
    ) -> None:
        """Receive frame from worker and update UI."""
        msg = gesture_res.message if gesture_res else ""
        self.camera_view.update_frame(frame, state, fps, msg)

        # Stream to virtual camera if active
        if self.vcam_mgr.is_active:
            self.vcam_mgr.send_frame(frame)

        # Update telemetry
        self._update_telemetry(gesture_res)

    def _on_error_received(self, error_message: str) -> None:
        QMessageBox.warning(self, "PhantomTouch Warning", error_message)

    def _on_state_changed(self, old_state: AppState, new_state: AppState) -> None:
        self._update_status_display()

    def _update_telemetry(self, res: Optional[GestureResult]) -> None:
        if res is None:
            return

        if res.is_valid:
            self.card_gesture.set_status("Valid Trigger!", "#10b981", "Invisibility Active")
        elif not res.right_hand_found:
            self.card_gesture.set_status("Searching Hand", "#64748b", "Raise right hand")
        elif not (res.index_extended and res.middle_extended):
            self.card_gesture.set_status("Extend 2 Fingers", "#f59e0b", "Index + middle required")
        elif not (res.ring_folded and res.pinky_folded):
            self.card_gesture.set_status("Fold 2 Fingers", "#f59e0b", "Fold ring + pinky")
        elif not res.forehead_touching:
            self.card_gesture.set_status("Touch Forehead", "#3b82f6", f"Dist: {res.index_forehead_dist:.2f}")
        else:
            self.card_gesture.set_status("Aligning", "#6366f1", res.message)

    def _update_status_display(self) -> None:
        state = self.state_mgr.current_state

        # Operating Mode Card
        if state == AppState.INVISIBLE:
            self.card_state.set_status("INVISIBLE", "#10b981", "User removed from video")
        elif state == AppState.GESTURE_DETECTED:
            self.card_state.set_status("GESTURE DETECTED", "#6366f1", "Hold gesture to vanish")
        elif state == AppState.INVISIBILITY_ACTIVATING:
            self.card_state.set_status("ACTIVATING", "#8b5cf6", "Fading to background")
        elif state == AppState.INVISIBILITY_DEACTIVATING:
            self.card_state.set_status("HOLDING", "#f59e0b", "Debounce hold")
        elif state == AppState.BACKGROUND_REQUIRED:
            self.card_state.set_status("BG REQUIRED", "#ef4444", "Capture empty background")
        elif state == AppState.NORMAL:
            self.card_state.set_status("NORMAL", "#3b82f6", "Person visible")
        else:
            self.card_state.set_status(state.value, "#64748b")

        # Background Card
        if self.background_mgr.has_background:
            self.card_bg.set_status("Ready", "#10b981", "Clean background saved")
            self.btn_capture_bg.setText("Recalibrate Background")
        else:
            self.card_bg.set_status("Required", "#ef4444", "Step 1: Capture empty scene")
            self.btn_capture_bg.setText("Capture Background")

    def _on_capture_bg_clicked(self) -> None:
        if not self.webcam_mgr.is_running:
            QMessageBox.information(self, "Camera Needed", "Please start the camera first before capturing.")
            return

        self.worker.trigger_background_capture()

    def _on_clear_bg_clicked(self) -> None:
        self.background_mgr.clear()
        self.state_mgr.set_background_status(False)
        self._update_status_display()

    def _on_camera_selected(self, index: int) -> None:
        device_idx = self.cam_combo.currentData()
        if device_idx is not None and device_idx != self.webcam_mgr.camera_index:
            self.webcam_mgr.set_camera_index(device_idx)

    def _on_manual_toggled(self, checked: bool) -> None:
        self.state_mgr.set_manual_override(checked)
        self._update_status_display()

    def _on_mirror_toggled(self, checked: bool) -> None:
        self.config.mirror_preview = checked

    def _on_sens_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.config.forehead_distance_threshold = threshold
        self.worker.gesture_detector.forehead_dist_threshold = threshold

    def _on_seg_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.config.seg_threshold = threshold
        self.worker.mask_processor.threshold = threshold

    def _on_feather_changed(self, value: int) -> None:
        radius = value if value % 2 == 1 else value + 1
        self.config.seg_feather_radius = radius
        self.worker.mask_processor.feather_radius = radius

    def _on_vcam_toggled(self, checked: bool) -> None:
        if checked:
            ok, msg = self.vcam_mgr.start()
            if not ok:
                self.chk_vcam.setChecked(False)
                QMessageBox.information(self, "Virtual Camera", msg)
        else:
            self.vcam_mgr.stop()

    def closeEvent(self, event) -> None:
        """Ensure clean shutdown of threads and devices."""
        logger.info("Closing application...")
        self.stop_pipeline()
        self.vcam_mgr.stop()
        self.config.save()
        event.accept()
