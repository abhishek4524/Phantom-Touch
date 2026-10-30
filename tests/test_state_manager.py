"""
Automated unit tests for StateManager and debounce logic.
"""

import pytest

from phantomtouch.core.state_manager import StateManager, AppState


class TestStateManager:
    def test_initial_state_requires_background(self):
        sm = StateManager(activation_frames=3, hold_frames=4)
        assert sm.current_state == AppState.CAMERA_INITIALIZING

        # Camera initialized without background
        sm.camera_initialized()
        assert sm.current_state == AppState.BACKGROUND_REQUIRED
        assert not sm.is_invisible

    def test_missing_background_blocks_activation(self):
        """Even if gesture is detected, system refuses to activate without background."""
        sm = StateManager(activation_frames=2, hold_frames=2)
        sm.camera_initialized()
        assert sm.current_state == AppState.BACKGROUND_REQUIRED

        # Feed 10 consecutive positive gestures
        for _ in range(10):
            sm.update(gesture_detected=True)
            assert sm.current_state == AppState.BACKGROUND_REQUIRED
            assert not sm.is_invisible

    def test_successful_activation_sequence(self):
        """Verify smooth progression: NORMAL -> GESTURE_DETECTED -> ACTIVATING -> INVISIBLE."""
        sm = StateManager(activation_frames=3, hold_frames=4)
        sm.set_background_status(True)
        sm.camera_initialized()
        assert sm.current_state == AppState.NORMAL

        # Frame 1: Gesture detected
        sm.update(gesture_detected=True)
        assert sm.current_state == AppState.GESTURE_DETECTED

        # Frame 2: Gesture held
        sm.update(gesture_detected=True)
        assert sm.current_state == AppState.GESTURE_DETECTED

        # Frame 3: Reached activation_frames (3)
        sm.update(gesture_detected=True)
        assert sm.current_state == AppState.INVISIBILITY_ACTIVATING
        assert sm.is_invisible

        # Frame 4: Activated fully into INVISIBLE
        sm.update(gesture_detected=True)
        assert sm.current_state == AppState.INVISIBLE
        assert sm.is_invisible

    def test_hysteresis_and_debounce_hold(self):
        """Short detection interruptions must NOT immediately deactivate invisibility."""
        sm = StateManager(activation_frames=2, hold_frames=4)
        sm.set_background_status(True)
        sm.camera_initialized()

        # Activate
        sm.update(True)
        sm.update(True)
        sm.update(True)
        assert sm.current_state == AppState.INVISIBLE

        # Frame drop 1: gesture briefly lost
        sm.update(False)
        assert sm.current_state == AppState.INVISIBILITY_DEACTIVATING
        assert sm.is_invisible, "Must hold invisibility during debounce window"

        # Frame drop 2: still lost
        sm.update(False)
        assert sm.current_state == AppState.INVISIBILITY_DEACTIVATING
        assert sm.is_invisible

        # Gesture recovers before hold_frames (4) expires!
        sm.update(True)
        assert sm.current_state == AppState.INVISIBLE
        assert sm.is_invisible

    def test_deactivation_after_full_hold_expires(self):
        """When gesture is removed permanently, state returns to NORMAL after hold_frames."""
        sm = StateManager(activation_frames=2, hold_frames=3)
        sm.set_background_status(True)
        sm.camera_initialized()

        # Activate
        sm.update(True)
        sm.update(True)
        sm.update(True)
        assert sm.current_state == AppState.INVISIBLE

        # Gesture dropped for 1, 2, 3 frames
        sm.update(False)  # Frame 1 lost: DEACTIVATING
        sm.update(False)  # Frame 2 lost: DEACTIVATING
        sm.update(False)  # Frame 3 lost: Reached hold_frames -> NORMAL
        assert sm.current_state == AppState.NORMAL
        assert not sm.is_invisible

    def test_manual_override(self):
        sm = StateManager()
        sm.set_background_status(True)
        sm.camera_initialized()
        assert sm.current_state == AppState.NORMAL

        sm.set_manual_override(True)
        assert sm.current_state == AppState.INVISIBLE
        assert sm.is_invisible

        sm.set_manual_override(False)
        assert sm.current_state == AppState.NORMAL
        assert not sm.is_invisible
