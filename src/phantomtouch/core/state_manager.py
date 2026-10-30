"""
PhantomTouch Gesture and Operating State Manager.
Manages state transitions, hysteresis, activation debouncing, and failure states.
"""

from enum import Enum, auto
from typing import Optional, Callable
from phantomtouch.utils.logger import get_logger

logger = get_logger("StateManager")


class AppState(Enum):
    CAMERA_INITIALIZING = "CAMERA_INITIALIZING"
    BACKGROUND_REQUIRED = "BACKGROUND_REQUIRED"
    NORMAL = "NORMAL"
    GESTURE_DETECTED = "GESTURE_DETECTED"
    INVISIBILITY_ACTIVATING = "INVISIBILITY_ACTIVATING"
    INVISIBLE = "INVISIBLE"
    INVISIBILITY_DEACTIVATING = "INVISIBILITY_DEACTIVATING"
    CAMERA_ERROR = "CAMERA_ERROR"


class StateManager:
    """
    Coordinates application states and smooth transitions for the invisibility effect.
    """

    def __init__(
        self,
        activation_frames: int = 5,
        hold_frames: int = 8,
        on_state_change: Optional[Callable[[AppState, AppState], None]] = None,
    ):
        self.activation_frames = activation_frames
        self.hold_frames = hold_frames
        self.on_state_change = on_state_change

        self._current_state: AppState = AppState.CAMERA_INITIALIZING
        self._consecutive_gesture_frames: int = 0
        self._lost_gesture_frames: int = 0
        self._manual_override: bool = False
        self._has_valid_background: bool = False

    @property
    def current_state(self) -> AppState:
        return self._current_state

    @property
    def is_invisible(self) -> bool:
        """True when the video compositor should output the invisibility effect."""
        return self._current_state in (
            AppState.INVISIBILITY_ACTIVATING,
            AppState.INVISIBLE,
            AppState.INVISIBILITY_DEACTIVATING,
        )

    @property
    def manual_override(self) -> bool:
        return self._manual_override

    def set_manual_override(self, enabled: bool) -> None:
        """Manually toggle invisibility if background is ready."""
        self._manual_override = enabled
        logger.info(f"Manual invisibility override set to: {enabled}")
        if enabled:
            if self._has_valid_background:
                self._transition_to(AppState.INVISIBLE)
            else:
                self._transition_to(AppState.BACKGROUND_REQUIRED)
        else:
            if self._has_valid_background:
                self._transition_to(AppState.NORMAL)

    def set_background_status(self, has_background: bool) -> None:
        """Update background readiness."""
        self._has_valid_background = has_background
        if not has_background and self._current_state not in (AppState.CAMERA_INITIALIZING, AppState.CAMERA_ERROR):
            self._transition_to(AppState.BACKGROUND_REQUIRED)
        elif has_background and self._current_state == AppState.BACKGROUND_REQUIRED:
            self._transition_to(AppState.NORMAL)

    def set_camera_error(self, has_error: bool) -> None:
        """Report camera hardware failure."""
        if has_error:
            self._transition_to(AppState.CAMERA_ERROR)
        elif self._current_state == AppState.CAMERA_ERROR:
            self._transition_to(AppState.CAMERA_INITIALIZING)

    def camera_initialized(self) -> None:
        """Signal that webcam has started successfully."""
        if self._has_valid_background:
            self._transition_to(AppState.NORMAL)
        else:
            self._transition_to(AppState.BACKGROUND_REQUIRED)

    def update(self, gesture_detected: bool) -> AppState:
        """
        Process the detection result from the current frame.
        Applies hysteresis and debouncing.
        """
        if self._manual_override:
            if not self._has_valid_background:
                self._transition_to(AppState.BACKGROUND_REQUIRED)
            return self._current_state

        if self._current_state in (AppState.CAMERA_INITIALIZING, AppState.CAMERA_ERROR):
            return self._current_state

        if not self._has_valid_background:
            self._transition_to(AppState.BACKGROUND_REQUIRED)
            return self._current_state

        # Background is ready; evaluate gesture triggers
        if gesture_detected:
            self._consecutive_gesture_frames += 1
            self._lost_gesture_frames = 0

            if self._current_state == AppState.NORMAL:
                self._transition_to(AppState.GESTURE_DETECTED)

            elif self._current_state == AppState.GESTURE_DETECTED:
                if self._consecutive_gesture_frames >= self.activation_frames:
                    self._transition_to(AppState.INVISIBILITY_ACTIVATING)

            elif self._current_state == AppState.INVISIBILITY_ACTIVATING:
                self._transition_to(AppState.INVISIBLE)

            elif self._current_state == AppState.INVISIBILITY_DEACTIVATING:
                # Gesture returned during deactivation hold
                self._transition_to(AppState.INVISIBLE)

        else:
            # Gesture not detected in this frame
            self._lost_gesture_frames += 1
            self._consecutive_gesture_frames = 0

            if self._current_state == AppState.GESTURE_DETECTED:
                self._transition_to(AppState.NORMAL)

            elif self._current_state == AppState.INVISIBLE:
                # Begin deactivating with debounce hold
                self._transition_to(AppState.INVISIBILITY_DEACTIVATING)

            elif self._current_state == AppState.INVISIBILITY_DEACTIVATING:
                if self._lost_gesture_frames >= self.hold_frames:
                    self._transition_to(AppState.NORMAL)

            elif self._current_state == AppState.INVISIBILITY_ACTIVATING:
                if self._lost_gesture_frames >= 2:
                    self._transition_to(AppState.NORMAL)

        return self._current_state

    def _transition_to(self, new_state: AppState) -> None:
        if self._current_state != new_state:
            old_state = self._current_state
            self._current_state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def reset(self) -> None:
        """Reset internal frame counters."""
        self._consecutive_gesture_frames = 0
        self._lost_gesture_frames = 0
