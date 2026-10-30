"""
PhantomTouch Face Detector.
Detects faces and computes dynamic forehead region of interest and scale invariants
using MediaPipe Tasks Vision FaceLandmarker API.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from phantomtouch.utils.config import FACE_LANDMARKER_MODEL
from phantomtouch.utils.logger import get_logger

logger = get_logger("FaceDetector")


@dataclass
class ForeheadROI:
    """Dynamic forehead region of interest normalized to [0.0, 1.0]."""
    center_x: float
    center_y: float
    radius: float
    face_scale: float  # Inter-eye or face reference scale
    glabella: Tuple[float, float]  # Point between eyebrows (LM 9/151)
    hairline: Tuple[float, float]  # Top of forehead (LM 10)


class FaceDetector:
    """Wrapper around MediaPipe FaceLandmarker task."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or FACE_LANDMARKER_MODEL)
        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize the MediaPipe FaceLandmarker."""
        if not self.model_path.exists():
            logger.error(f"Face landmarker model not found at: {self.model_path}")
            return False

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe Face Landmarker initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Face Landmarker: {e}")
            self._landmarker = None
            self._is_initialized = False
            return False

    def detect_forehead(self, frame_bgr: np.ndarray) -> Optional[ForeheadROI]:
        """
        Detect the face in frame_bgr and compute forehead ROI.
        Returns ForeheadROI if a face is detected, or None.
        """
        if not self._is_initialized or self._landmarker is None:
            return None

        if frame_bgr is None or frame_bgr.size == 0:
            return None

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None

            lms = result.face_landmarks[0]

            # Key landmark indices:
            # 10: Top hairline
            # 9 or 151: Glabella (between eyebrows)
            # 33: Left eye outer corner
            # 263: Right eye outer corner
            lm10 = lms[10]
            lm9 = lms[9]
            lm33 = lms[33]
            lm263 = lms[263]

            # Reference face scale: inter-eye distance in normalized coords
            dx_eye = lm263.x - lm33.x
            dy_eye = lm263.y - lm33.y
            inter_eye_dist = float(np.sqrt(dx_eye**2 + dy_eye**2))

            # Guard against division by zero or extreme angles
            face_scale = max(0.05, inter_eye_dist)

            # Forehead center is midpoint between glabella (9) and top hairline (10)
            center_x = (lm9.x + lm10.x) / 2.0
            center_y = (lm9.y + lm10.y) / 2.0

            # Dynamic radius: 50% of inter-eye distance
            radius = face_scale * 0.50

            return ForeheadROI(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                face_scale=face_scale,
                glabella=(lm9.x, lm9.y),
                hairline=(lm10.x, lm10.y),
            )

        except Exception as e:
            logger.warning(f"Error during face/forehead detection: {e}")
            return None

    def close(self) -> None:
        """Release landmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self._is_initialized = False
