"""
PhantomTouch Webcam Manager.
Handles camera enumeration, frame acquisition, bounded buffer to avoid stale frames,
resolution configuration, and graceful release.
"""

import threading
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from phantomtouch.utils.logger import get_logger

logger = get_logger("WebcamManager")


class WebcamManager:
    """Manages webcam capture in a dedicated reader thread."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._frame_count: int = 0
        self._actual_fps: float = 0.0

        # FPS calculation tracking
        self._fps_start_time = time.time()
        self._fps_frame_counter = 0

    @staticmethod
    def list_available_cameras(max_tested: int = 6) -> List[int]:
        """
        Probe camera indices up to max_tested using cv2.CAP_DSHOW on Windows.
        Returns list of accessible indices.
        """
        available = []
        for idx in range(max_tested):
            # Prefer cv2.CAP_DSHOW on Windows for fast non-blocking probe
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
            else:
                # Also try default backend if DSHOW is not available
                cap_def = cv2.VideoCapture(idx)
                if cap_def.isOpened():
                    ret, _ = cap_def.read()
                    if ret:
                        available.append(idx)
                    cap_def.release()
        return available

    def start(self) -> bool:
        """Start the webcam capture thread. Returns True if successfully opened."""
        if self._is_running:
            return True

        logger.info(f"Opening webcam index {self.camera_index} ({self.width}x{self.height})...")
        # Try DirectShow first on Windows, then fallback
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.info("DirectShow open failed, trying default backend...")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera index {self.camera_index}")
            self.cap = None
            return False

        # Set resolution and preferred buffer size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Webcam opened successfully: {actual_w}x{actual_h}")

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="WebcamCaptureThread")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop capture and release resources."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            self.cap = None

        with self._lock:
            self._latest_frame = None

        logger.info("Webcam stopped and released.")

    def _capture_loop(self) -> None:
        """Continuously grab the latest frame to keep the driver buffer fresh."""
        while self._is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Brief sleep to avoid 100% CPU spin on read failure
                time.sleep(0.01)
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._frame_timestamp = now
                self._frame_count += 1
                self._fps_frame_counter += 1

                # Update FPS every ~1.0 second
                elapsed = now - self._fps_start_time
                if elapsed >= 1.0:
                    self._actual_fps = self._fps_frame_counter / elapsed
                    self._fps_frame_counter = 0
                    self._fps_start_time = now

            # Minor yield to allow reader thread scheduling
            time.sleep(0.002)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieve the most recent frame without buffering lag.
        Returns: (success, frame_bgr, timestamp)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            # Return a copy or reference of the current frame
            return True, self._latest_frame.copy(), self._frame_timestamp

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def actual_fps(self) -> float:
        with self._lock:
            return self._actual_fps

    def set_camera_index(self, index: int) -> bool:
        """Switch to another camera index."""
        was_running = self._is_running
        if was_running:
            self.stop()
        self.camera_index = index
        if was_running:
            return self.start()
        return True
