"""
PhantomTouch Mask Processor.
Processes segmentation masks with thresholding, morphological noise removal,
temporal exponential moving average smoothing, and edge feathering.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from phantomtouch.utils.logger import get_logger

logger = get_logger("MaskProcessor")


class MaskProcessor:
    """Refines raw segmentation masks into clean, temporally stable alpha masks."""

    def __init__(
        self,
        threshold: float = 0.50,
        feather_radius: int = 9,
        temporal_alpha: float = 0.40,
        remove_small_artifacts: bool = True,
        min_artifact_area: int = 200,
    ):
        self.threshold = threshold
        self.feather_radius = feather_radius
        self.temporal_alpha = temporal_alpha
        self.remove_small_artifacts = remove_small_artifacts
        self.min_artifact_area = min_artifact_area

        self._prev_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear temporal history."""
        self._prev_mask = None

    def process(self, raw_mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Refine a float32 confidence mask [0.0, 1.0] to a smoothed float32 alpha mask [0.0, 1.0]
        matching target_shape (H, W).
        """
        target_h, target_w = target_shape[:2]

        if raw_mask is None or raw_mask.size == 0:
            if self._prev_mask is not None and self._prev_mask.shape == (target_h, target_w):
                # Fade out smoothly if mask dropped for 1 frame
                self._prev_mask *= (1.0 - self.temporal_alpha)
                return self._prev_mask
            return np.zeros((target_h, target_w), dtype=np.float32)

        # 1. Resize to target frame resolution if dimensions differ
        if raw_mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = raw_mask.copy()

        # Ensure float32 in range [0.0, 1.0]
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)

        # 2. Thresholding / Sigmoid shaping
        # Create binary mask for morphological cleanup
        binary = (mask >= self.threshold).astype(np.uint8)

        # 3. Artifact cleanup (morphology & contour area filtering)
        if self.remove_small_artifacts:
            # Small open operation to eliminate isolated 1-2 pixel noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Filter out components smaller than min_artifact_area
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            cleaned_binary = np.zeros_like(binary)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_artifact_area:
                    cleaned_binary[labels == i] = 1
            binary = cleaned_binary

        # Modulate original confidence with cleaned binary boundary
        clean_mask = mask * binary.astype(np.float32)

        # 4. Temporal Smoothing: Exponential Moving Average
        if self._prev_mask is not None and self._prev_mask.shape == clean_mask.shape:
            smoothed = self.temporal_alpha * clean_mask + (1.0 - self.temporal_alpha) * self._prev_mask
        else:
            smoothed = clean_mask
        self._prev_mask = smoothed.copy()

        # 5. Edge Feathering / Blur for seamless boundary blending
        if self.feather_radius > 1:
            ksize = self.feather_radius if self.feather_radius % 2 == 1 else self.feather_radius + 1
            feathered = cv2.GaussianBlur(smoothed, (ksize, ksize), 0)
        else:
            feathered = smoothed

        # Clamp strictly between 0.0 and 1.0
        return np.clip(feathered, 0.0, 1.0).astype(np.float32)
