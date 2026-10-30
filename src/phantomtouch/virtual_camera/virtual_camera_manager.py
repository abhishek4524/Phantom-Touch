"""
PhantomTouch Virtual Camera Manager.
Provides optional virtual camera output via pyvirtualcam for Zoom, Google Meet, OBS.
Gracefully handles missing backends (e.g. OBS Virtual Camera driver).
"""

from typing import Optional, Tuple
import numpy as np
import cv2

from phantomtouch.utils.logger import get_logger

logger = get_logger("VirtualCameraManager")


class VirtualCameraManager:
    """Manages virtual camera streaming without crashing if driver is missing."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._cam = None
        self._is_active = False
        self._available_backend: Optional[str] = None
        self._error_message: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def is_pyvirtualcam_installed(self) -> bool:
        """Check if pyvirtualcam library is available in environment."""
        try:
            import pyvirtualcam  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self) -> Tuple[bool, str]:
        """
        Attempt to initialize virtual camera output.
        Returns: (success, status_or_error_message)
        """
        if self._is_active:
            return True, "Virtual camera already active."

        try:
            import pyvirtualcam

            # Attempt creation with default or OBS backend
            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self._is_active = True
            device_name = self._cam.device
            logger.info(f"Virtual camera started successfully: {device_name}")
            return True, f"Virtual camera active: {device_name}"

        except ImportError:
            msg = "pyvirtualcam is not installed. Virtual camera output unavailable."
            logger.warning(msg)
            self._error_message = msg
            return False, msg

        except Exception as e:
            msg = (
                f"Virtual camera driver unavailable: {e}. "
                "To use PhantomTouch in Zoom/Meet, install OBS Studio with Virtual Camera enabled."
            )
            logger.warning(msg)
            self._error_message = msg
            self._cam = None
            self._is_active = False
            return False, msg

    def send_frame(self, frame_bgr: np.ndarray) -> bool:
        """Send a BGR frame to the virtual camera if active."""
        if not self._is_active or self._cam is None or frame_bgr is None:
            return False

        try:
            # Ensure correct resolution
            h, w = frame_bgr.shape[:2]
            if w != self.width or h != self.height:
                frame_to_send = cv2.resize(frame_bgr, (self.width, self.height))
            else:
                frame_to_send = frame_bgr

            self._cam.send(frame_to_send)
            self._cam.sleep_until_next_frame()
            return True
        except Exception as e:
            logger.warning(f"Error sending frame to virtual camera: {e}")
            return False

    def stop(self) -> None:
        """Close virtual camera."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        self._is_active = False
        logger.info("Virtual camera stopped.")
