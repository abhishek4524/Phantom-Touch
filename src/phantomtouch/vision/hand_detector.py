"""
PhantomTouch Hand Detector.
Detects hands and 21 3D landmarks using MediaPipe Tasks Vision HandLandmarker API.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from phantomtouch.utils.config import HAND_LANDMARKER_MODEL
from phantomtouch.utils.logger import get_logger

logger = get_logger("HandDetector")


@dataclass
class HandLandmarks:
    """Represents detected hand landmarks and handedness."""
    handedness: str  # "Right" or "Left"
    score: float
    # Normalized landmarks (x, y, z in [0.0, 1.0])
    landmarks: List[Tuple[float, float, float]]

    # Convenience fingertip normalized coordinates
    @property
    def wrist(self) -> Tuple[float, float, float]:
        return self.landmarks[0]

    @property
    def index_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[8]

    @property
    def middle_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[12]

    @property
    def ring_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[16]

    @property
    def pinky_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[20]


class HandDetector:
    """Wrapper around MediaPipe HandLandmarker task."""

    def __init__(self, model_path: Optional[Path] = None, max_hands: int = 2):
        self.model_path = Path(model_path or HAND_LANDMARKER_MODEL)
        self.max_hands = max_hands
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize the MediaPipe HandLandmarker."""
        if not self.model_path.exists():
            logger.error(f"Hand landmarker model not found at: {self.model_path}")
            return False

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe Hand Landmarker initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Hand Landmarker: {e}")
            self._landmarker = None
            self._is_initialized = False
            return False

    def detect(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands in a BGR frame.
        Returns list of HandLandmarks.
        """
        if not self._is_initialized or self._landmarker is None:
            return []

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = self._landmarker.detect(mp_image)
            detected_hands: List[HandLandmarks] = []

            if not result.hand_landmarks:
                return []

            for idx, hand_lms in enumerate(result.hand_landmarks):
                # Retrieve handedness (Right vs Left)
                label = "Unknown"
                score = 1.0
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    cat = result.handedness[idx][0]
                    label = cat.category_name
                    score = cat.score

                coords = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                detected_hands.append(HandLandmarks(handedness=label, score=score, landmarks=coords))

            return detected_hands

        except Exception as e:
            logger.warning(f"Error during hand detection: {e}")
            return []

    def close(self) -> None:
        """Release landmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self._is_initialized = False
