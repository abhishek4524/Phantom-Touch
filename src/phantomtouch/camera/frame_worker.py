"""
PhantomTouch Frame Worker.
Runs computer vision inference and compositing asynchronously off the UI thread.
"""

import threading
import time
from typing import Optional, Callable
import cv2
import numpy as np

from phantomtouch.camera.webcam_manager import WebcamManager
from phantomtouch.vision.person_segmenter import PersonSegmenter
from phantomtouch.vision.hand_detector import HandDetector
from phantomtouch.vision.face_detector import FaceDetector, ForeheadROI
from phantomtouch.vision.gesture_detector import GestureDetector, GestureResult
from phantomtouch.vision.mask_processor import MaskProcessor
from phantomtouch.core.state_manager import StateManager, AppState
from phantomtouch.core.background_manager import BackgroundManager
from phantomtouch.core.compositor import VideoCompositor
from phantomtouch.utils.config import AppConfig
from phantomtouch.utils.logger import get_logger

logger = get_logger("FrameWorker")


class FrameWorker:
    """
    Coordinates asynchronous frame capture, multi-model vision inference,
    state management, and compositing in a dedicated worker thread.
    """

    def __init__(
        self,
        config: AppConfig,
        webcam: WebcamManager,
        background_mgr: BackgroundManager,
        state_mgr: StateManager,
        compositor: VideoCompositor,
        on_frame_ready: Optional[Callable[[np.ndarray, GestureResult, AppState, float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.webcam = webcam
        self.background_mgr = background_mgr
        self.state_mgr = state_mgr
        self.compositor = compositor

        self.on_frame_ready = on_frame_ready
        self.on_error = on_error

        # Vision modules
        self.segmenter = PersonSegmenter()
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        self.gesture_detector = GestureDetector(
            forehead_dist_threshold=config.forehead_distance_threshold,
            finger_straight_ratio=config.finger_straight_threshold,
            require_right_hand=True,
        )
        self.mask_processor = MaskProcessor(
            threshold=config.seg_threshold,
            feather_radius=config.seg_feather_radius,
            temporal_alpha=config.seg_temporal_alpha,
            remove_small_artifacts=config.remove_small_artifacts,
            min_artifact_area=config.artifact_min_area,
        )

        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Performance and FPS calculation
        self._fps_start = time.time()
        self._fps_count = 0
        self._fps = 0.0

        # Background capture sequence state
        self._bg_countdown_start: Optional[float] = None
        self._bg_capturing = False
        self._bg_countdown_duration = config.bg_countdown_seconds

    def initialize_models(self) -> bool:
        """Initialize all three vision models."""
        logger.info("Initializing MediaPipe vision models...")
        seg_ok = self.segmenter.initialize()
        hand_ok = self.hand_detector.initialize()
        face_ok = self.face_detector.initialize()

        if not (seg_ok and hand_ok and face_ok):
            failed = []
            if not seg_ok:
                failed.append("Person Segmenter")
            if not hand_ok:
                failed.append("Hand Detector")
            if not face_ok:
                failed.append("Face Detector")
            msg = f"Failed to initialize models: {', '.join(failed)}"
            logger.error(msg)
            if self.on_error:
                self.on_error(msg)
            return False

        logger.info("All vision models initialized.")
        return True

    def start(self) -> bool:
        """Start worker thread and camera."""
        if self._is_running:
            return True

        if not self.webcam.start():
            self.state_mgr.set_camera_error(True)
            if self.on_error:
                self.on_error("Could not open webcam.")
            return False

        self.state_mgr.camera_initialized()
        self.state_mgr.set_background_status(self.background_mgr.has_background)

        self._is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PhantomTouchFrameWorker")
        self._thread.start()
        logger.info("FrameWorker loop started.")
        return True

    def stop(self) -> None:
        """Stop worker and release camera and model resources."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

        self.webcam.stop()
        self.segmenter.close()
        self.hand_detector.close()
        self.face_detector.close()
        logger.info("FrameWorker stopped.")

    def trigger_background_capture(self) -> None:
        """Initiate background capture countdown."""
        self._bg_countdown_start = time.time()
        self._bg_capturing = False
        logger.info("Background capture countdown initiated.")

    def _run_loop(self) -> None:
        """Main processing loop."""
        while self._is_running:
            loop_start = time.time()

            ret, raw_frame, _ = self.webcam.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            # Update FPS tracking
            self._fps_count += 1
            if time.time() - self._fps_start >= 1.0:
                self._fps = self._fps_count / (time.time() - self._fps_start)
                self._fps_count = 0
                self._fps_start = time.time()

            # Handle background capture sequence
            if self._bg_countdown_start is not None:
                elapsed = time.time() - self._bg_countdown_start
                remaining = self._bg_countdown_duration - elapsed
                if remaining > 0:
                    # Still in countdown - show countdown overlay
                    preview = raw_frame.copy()
                    text = f"LEAVE FRAME! Capturing in: {remaining:.1f}s"
                    cv2.putText(preview, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if self.config.mirror_preview:
                        preview = cv2.flip(preview, 1)
                    if self.on_frame_ready:
                        self.on_frame_ready(preview, None, self.state_mgr.current_state, self._fps)
                    time.sleep(0.02)
                    continue
                else:
                    # Countdown finished: start accumulating clean background frames
                    if not self._bg_capturing:
                        self.background_mgr.start_capture_sequence()
                        self._bg_capturing = True

                    count = self.background_mgr.add_capture_frame(raw_frame)
                    preview = raw_frame.copy()
                    cv2.putText(preview, f"Capturing background: {count}/{self.config.bg_capture_frames}",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    if self.config.mirror_preview:
                        preview = cv2.flip(preview, 1)
                    if self.on_frame_ready:
                        self.on_frame_ready(preview, None, self.state_mgr.current_state, self._fps)

                    if count >= self.config.bg_capture_frames:
                        success, bg = self.background_mgr.finish_capture()
                        self._bg_countdown_start = None
                        self._bg_capturing = False
                        self.state_mgr.set_background_status(success)
                        logger.info(f"Background capture completed: success={success}")
                    time.sleep(0.01)
                    continue

            # CRITICAL ARCHITECTURAL RULE:
            # GESTURE AND FACE DETECTION ALWAYS RUN ON THE ORIGINAL RAW WEBCAM FRAME!
            # Never run gesture recognition on composited output.
            hands = self.hand_detector.detect(raw_frame)
            forehead_roi = self.face_detector.detect_forehead(raw_frame)
            gesture_res = self.gesture_detector.evaluate(hands, forehead_roi)

            # Update State Manager with detection outcome
            self.state_mgr.update(gesture_res.is_valid)

            # Run person segmentation
            # We segment when invisibility is active or transitioning
            raw_mask = None
            processed_mask = None
            if self.state_mgr.is_invisible or self.compositor.blend_factor > 0.0:
                raw_mask = self.segmenter.segment(raw_frame)
                if raw_mask is not None:
                    processed_mask = self.mask_processor.process(raw_mask, target_shape=raw_frame.shape)

            # Alpha composite with empty background
            composited = self.compositor.composite(
                original_frame=raw_frame,
                background_frame=self.background_mgr.background_image,
                person_mask=processed_mask,
                invisibility_active=self.state_mgr.is_invisible,
            )

            # Preview mirroring if configured
            display_frame = cv2.flip(composited, 1) if self.config.mirror_preview else composited

            # Emit frame to UI
            if self.on_frame_ready:
                self.on_frame_ready(display_frame, gesture_res, self.state_mgr.current_state, self._fps)

            # Regulate frame rate to prevent unnecessary CPU consumption
            target_interval = 1.0 / max(10, self.config.target_fps)
            loop_duration = time.time() - loop_start
            if loop_duration < target_interval:
                time.sleep(target_interval - loop_duration)
