"""
PhantomTouch Person Segmenter.
Uses Google MediaPipe Tasks Vision ImageSegmenter API.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from phantomtouch.utils.config import SELFIE_SEGMENTER_MODEL
from phantomtouch.utils.logger import get_logger

logger = get_logger("PersonSegmenter")


class PersonSegmenter:
    """Wrapper around MediaPipe ImageSegmenter task."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or SELFIE_SEGMENTER_MODEL)
        self._segmenter: Optional[vision.ImageSegmenter] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize the MediaPipe ImageSegmenter."""
        if not self.model_path.exists():
            logger.error(f"Segmenter model not found at: {self.model_path}")
            return False

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe Person Segmenter initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Person Segmenter: {e}")
            self._segmenter = None
            self._is_initialized = False
            return False

    def segment(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run person segmentation on a BGR frame.
        Returns: 2D float32 numpy array (H, W) in range [0.0, 1.0], or None on error.
        """
        if not self._is_initialized or self._segmenter is None:
            return None

        if frame_bgr is None or frame_bgr.size == 0:
            return None

        try:
            # Convert OpenCV BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = self._segmenter.segment(mp_image)
            if not result.confidence_masks:
                return None

            # selfie_segmenter returns 1 confidence mask where values are person confidence
            raw_mask = result.confidence_masks[0].numpy_view()
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.squeeze(axis=-1)

            return raw_mask.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error during segmentation inference: {e}")
            return None

    def close(self) -> None:
        """Release underlying segmenter resources."""
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
            self._segmenter = None
        self._is_initialized = False
