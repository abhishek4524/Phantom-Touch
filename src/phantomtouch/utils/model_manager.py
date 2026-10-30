"""
PhantomTouch Model Manager.
Verifies and resolves paths for MediaPipe task models.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from phantomtouch.utils.config import (
    HAND_LANDMARKER_MODEL,
    FACE_LANDMARKER_MODEL,
    SELFIE_SEGMENTER_MODEL,
    MODELS_DIR,
)
from phantomtouch.utils.logger import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """Manages local model availability and verification."""

    REQUIRED_MODELS: Dict[str, Tuple[Path, int]] = {
        "Hand Landmarker": (HAND_LANDMARKER_MODEL, 5_000_000),
        "Face Landmarker": (FACE_LANDMARKER_MODEL, 3_000_000),
        "Person Segmenter": (SELFIE_SEGMENTER_MODEL, 100_000),
    }

    @classmethod
    def check_models(cls) -> Tuple[bool, List[str]]:
        """
        Check if all required models exist and have valid file sizes.
        Returns: (all_valid, list_of_missing_or_corrupted_model_names)
        """
        missing = []
        for name, (path, min_size) in cls.REQUIRED_MODELS.items():
            if not path.exists():
                missing.append(f"{name} (missing: {path.name})")
            elif path.stat().st_size < min_size:
                missing.append(f"{name} (corrupted/incomplete: {path.name}, size={path.stat().st_size} bytes)")

        return (len(missing) == 0, missing)

    @classmethod
    def get_hand_model_path(cls) -> Path:
        return HAND_LANDMARKER_MODEL

    @classmethod
    def get_face_model_path(cls) -> Path:
        return FACE_LANDMARKER_MODEL

    @classmethod
    def get_segmenter_model_path(cls) -> Path:
        return SELFIE_SEGMENTER_MODEL
