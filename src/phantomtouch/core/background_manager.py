"""
PhantomTouch Background Capture and Storage Manager.
Captures empty scene frames, computes temporal median background,
saves and verifies local background images.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from phantomtouch.utils.config import DEFAULT_BACKGROUND_PATH
from phantomtouch.utils.logger import get_logger

logger = get_logger("BackgroundManager")


class BackgroundManager:
    """Manages clean background calibration, persistence, and retrieval."""

    def __init__(self, background_path: Path = DEFAULT_BACKGROUND_PATH):
        self.background_path = Path(background_path)
        self._background_image: Optional[np.ndarray] = None
        self._captured_frames: List[np.ndarray] = []
        self._is_capturing: bool = False

        # Attempt to load existing background if available
        self.load_background()

    @property
    def has_background(self) -> bool:
        """True if a valid, non-empty background image is loaded."""
        return self._background_image is not None and self._background_image.size > 0

    @property
    def background_image(self) -> Optional[np.ndarray]:
        """Return the loaded background image (BGR)."""
        return self._background_image

    def load_background(self) -> bool:
        """Load background image from file if present and valid."""
        if not self.background_path.exists():
            logger.info("No saved background image found.")
            self._background_image = None
            return False

        try:
            img = cv2.imread(str(self.background_path))
            if img is None or img.size == 0:
                logger.warning(f"Background image at {self.background_path} is invalid or empty.")
                self._background_image = None
                return False

            self._background_image = img
            logger.info(f"Loaded existing background: {img.shape[1]}x{img.shape[0]}")
            return True
        except Exception as e:
            logger.error(f"Error loading background image: {e}")
            self._background_image = None
            return False

    def save_background(self, image: np.ndarray) -> bool:
        """Save given background image to disk."""
        if image is None or image.size == 0:
            logger.error("Cannot save empty background image.")
            return False

        try:
            self.background_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(self.background_path), image)
            if success:
                self._background_image = image.copy()
                logger.info(f"Background saved successfully to {self.background_path}")
                return True
            else:
                logger.error(f"Failed to write image to {self.background_path}")
                return False
        except Exception as e:
            logger.error(f"Error saving background image: {e}")
            return False

    def start_capture_sequence(self) -> None:
        """Prepare internal buffer for a multi-frame capture session."""
        self._captured_frames.clear()
        self._is_capturing = True

    def add_capture_frame(self, frame: np.ndarray) -> int:
        """Add a frame to the calibration sequence. Returns total collected so far."""
        if not self._is_capturing or frame is None:
            return len(self._captured_frames)

        self._captured_frames.append(frame.copy())
        return len(self._captured_frames)

    def finish_capture(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Compute temporal median from accumulated frames and save as new background.
        Returns: (success, background_image)
        """
        self._is_capturing = False
        if len(self._captured_frames) < 3:
            logger.warning(f"Insufficient frames for background calibration: {len(self._captured_frames)}")
            return False, None

        try:
            logger.info(f"Computing robust median background from {len(self._captured_frames)} frames...")
            # Stack frames along 4th dimension: (N, H, W, C)
            stack = np.array(self._captured_frames, dtype=np.uint8)
            # Compute median along time axis
            median_bg = np.median(stack, axis=0).astype(np.uint8)

            # Validate generated image (not pure black or degenerate)
            mean_val = float(np.mean(median_bg))
            if mean_val < 1.0:
                logger.warning("Captured background is completely black or degenerate.")
                return False, None

            # Persist
            saved = self.save_background(median_bg)
            self._captured_frames.clear()
            return saved, median_bg

        except Exception as e:
            logger.error(f"Failed to compute median background: {e}")
            self._captured_frames.clear()
            return False, None

    def clear(self) -> None:
        """Remove background and delete saved file if requested."""
        self._background_image = None
        if self.background_path.exists():
            try:
                self.background_path.unlink()
            except Exception:
                pass
        logger.info("Background cleared.")
