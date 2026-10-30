"""
PhantomTouch Video Compositor.
Blends live camera frames with captured background using the processed person mask.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from phantomtouch.utils.logger import get_logger

logger = get_logger("Compositor")


class VideoCompositor:
    """Performs alpha compositing between live webcam frames and the background."""

    def __init__(self, transition_speed: float = 0.15):
        self.transition_speed = transition_speed  # Smooth fade in/out speed per frame
        self._blend_factor: float = 0.0  # 0.0 = fully visible, 1.0 = fully invisible

    @property
    def blend_factor(self) -> float:
        return self._blend_factor

    def reset_transition(self) -> None:
        self._blend_factor = 0.0

    def composite(
        self,
        original_frame: np.ndarray,
        background_frame: Optional[np.ndarray],
        person_mask: Optional[np.ndarray],
        invisibility_active: bool,
    ) -> np.ndarray:
        """
        Produce composited output.
        - If invisibility_active is False and blend_factor is 0: returns original_frame.
        - If background_frame is None: returns original_frame.
        - Otherwise computes: Original * (1 - Mask * Blend) + Background * (Mask * Blend).
        """
        if original_frame is None or original_frame.size == 0:
            return original_frame

        h, w = original_frame.shape[:2]

        # Update smooth transition blend factor
        if invisibility_active and background_frame is not None:
            self._blend_factor = min(1.0, self._blend_factor + self.transition_speed)
        else:
            self._blend_factor = max(0.0, self._blend_factor - self.transition_speed)

        # Quick bypass if completely inactive
        if self._blend_factor <= 0.001 or background_frame is None or person_mask is None:
            self._blend_factor = 0.0
            return original_frame

        # Resize background if dimensions differ
        if background_frame.shape[:2] != (h, w):
            bg = cv2.resize(background_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bg = background_frame

        # Resize mask if dimensions differ
        if person_mask.shape[:2] != (h, w):
            mask = cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = person_mask

        # Effective alpha mask modulated by transition blend factor
        effective_alpha = mask * self._blend_factor

        # Expand mask to 3 channels (H, W, 1) for broadcasting with (H, W, 3)
        alpha_3ch = np.expand_dims(effective_alpha, axis=-1)

        # Alpha composite: Original * (1 - Alpha) + Background * Alpha
        orig_f = original_frame.astype(np.float32)
        bg_f = bg.astype(np.float32)

        composited = orig_f * (1.0 - alpha_3ch) + bg_f * alpha_3ch
        return np.clip(composited, 0, 255).astype(np.uint8)
