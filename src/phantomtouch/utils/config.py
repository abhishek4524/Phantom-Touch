"""
PhantomTouch Configuration Constants and Settings Management.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

# Project Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
SETTINGS_DIR = DATA_DIR / "settings"

DEFAULT_BACKGROUND_PATH = BACKGROUNDS_DIR / "background.png"
SETTINGS_FILE_PATH = SETTINGS_DIR / "settings.json"

# Model filenames
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
SELFIE_SEGMENTER_MODEL = MODELS_DIR / "selfie_segmenter.tflite"


@dataclass
class AppConfig:
    """Runtime application configuration settings."""
    # Camera Settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_preview: bool = True

    # Background Capture Settings
    bg_countdown_seconds: int = 3
    bg_capture_frames: int = 25
    background_path: str = str(DEFAULT_BACKGROUND_PATH)

    # Person Segmentation Settings
    seg_threshold: float = 0.50
    seg_feather_radius: int = 9  # odd integer blur kernel size
    seg_temporal_alpha: float = 0.40  # smoothing factor (0.0=all old, 1.0=all new)
    remove_small_artifacts: bool = True
    artifact_min_area: int = 200

    # Gesture Detection Settings
    # Consecutive valid gesture frames before transitioning to INVISIBLE
    gesture_activation_frames: int = 5
    # Frames to hold state when detection briefly drops (debounce / hysteresis)
    gesture_hold_frames: int = 8
    # Forehead tolerance: normalized distance threshold relative to inter-eye or face width
    forehead_distance_threshold: float = 0.35
    # Finger angle / extension threshold
    finger_straight_threshold: float = 0.85

    # Virtual Camera Settings
    virtual_camera_enabled: bool = False
    virtual_camera_device: str = ""

    def save(self, path: Path = SETTINGS_FILE_PATH) -> None:
        """Persist settings to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=4)
        except Exception:
            pass

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE_PATH) -> "AppConfig":
        """Load settings from JSON file or return defaults."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()
