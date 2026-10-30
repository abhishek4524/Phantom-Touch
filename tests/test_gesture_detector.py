"""
Automated unit tests for GestureDetector and ModelManager.
"""

import pytest
from pathlib import Path
from typing import List, Tuple, Optional

from phantomtouch.vision.gesture_detector import GestureDetector, GestureResult
from phantomtouch.vision.hand_detector import HandLandmarks
from phantomtouch.vision.face_detector import ForeheadROI
from phantomtouch.utils.model_manager import ModelManager


def create_mock_hand(
    handedness: str = "Right",
    index_tip_pos: Optional[Tuple[float, float]] = None,
    middle_tip_pos: Optional[Tuple[float, float]] = None,
    offset_y: float = 0.0,
    index_extended: bool = True,
    middle_extended: bool = True,
    ring_folded: bool = True,
    pinky_folded: bool = True,
) -> HandLandmarks:
    """Helper to generate mock 21 hand landmarks for geometric unit testing."""
    lms: List[Tuple[float, float, float]] = [(0.5, 0.8 + offset_y, 0.0)] * 21

    # Wrist
    lms[0] = (0.5, 0.90 + offset_y, 0.0)

    # Index finger: MCP (5), PIP (6), DIP (7), TIP (8)
    lms[5] = (0.45, 0.70 + offset_y, 0.0)
    lms[6] = (0.45, 0.50 + offset_y, 0.0)
    lms[7] = (0.45, 0.35 + offset_y, 0.0)
    if index_extended:
        tip = index_tip_pos if index_tip_pos is not None else (0.45, 0.20 + offset_y)
        lms[8] = (tip[0], tip[1], 0.0)
    else:
        lms[8] = (0.45, 0.65 + offset_y, 0.0)

    # Middle finger: MCP (9), PIP (10), DIP (11), TIP (12)
    lms[9] = (0.50, 0.68 + offset_y, 0.0)
    lms[10] = (0.50, 0.48 + offset_y, 0.0)
    lms[11] = (0.50, 0.33 + offset_y, 0.0)
    if middle_extended:
        tip = middle_tip_pos if middle_tip_pos is not None else (0.50, 0.20 + offset_y)
        lms[12] = (tip[0], tip[1], 0.0)
    else:
        lms[12] = (0.50, 0.64 + offset_y, 0.0)

    # Ring finger: MCP (13), PIP (14), DIP (15), TIP (16)
    lms[13] = (0.55, 0.70 + offset_y, 0.0)
    lms[14] = (0.55, 0.52 + offset_y, 0.0)
    lms[15] = (0.55, 0.58 + offset_y, 0.0)
    if ring_folded:
        lms[16] = (0.55, 0.66 + offset_y, 0.0)
    else:
        lms[16] = (0.55, 0.30 + offset_y, 0.0)

    # Pinky finger: MCP (17), PIP (18), DIP (19), TIP (20)
    lms[17] = (0.60, 0.72 + offset_y, 0.0)
    lms[18] = (0.60, 0.56 + offset_y, 0.0)
    lms[19] = (0.60, 0.62 + offset_y, 0.0)
    if pinky_folded:
        lms[20] = (0.60, 0.68 + offset_y, 0.0)
    else:
        lms[20] = (0.60, 0.35 + offset_y, 0.0)

    return HandLandmarks(handedness=handedness, score=0.95, landmarks=lms)


def create_mock_forehead(
    center: Tuple[float, float] = (0.5, 0.2),
    face_scale: float = 0.25,
) -> ForeheadROI:
    """Helper to generate mock ForeheadROI."""
    return ForeheadROI(
        center_x=center[0],
        center_y=center[1],
        radius=face_scale * 0.5,
        face_scale=face_scale,
        glabella=(center[0], center[1] + 0.06),
        hairline=(center[0], center[1] - 0.06),
    )


class TestGestureDetector:
    def test_no_hands_detected(self):
        detector = GestureDetector()
        forehead = create_mock_forehead()
        res = detector.evaluate(hands=[], forehead=forehead)
        assert not res.is_valid
        assert "No hands detected" in res.message

    def test_no_forehead_detected(self):
        detector = GestureDetector()
        hand = create_mock_hand()
        res = detector.evaluate(hands=[hand], forehead=None)
        assert not res.is_valid
        assert "Face not detected" in res.message

    def test_valid_two_finger_forehead_gesture(self):
        """Right hand, index+middle extended, ring+pinky folded, at forehead center."""
        detector = GestureDetector()
        forehead = create_mock_forehead(center=(0.5, 0.2))
        hand = create_mock_hand(
            handedness="Right",
            index_tip_pos=(0.5, 0.2),
            middle_tip_pos=(0.5, 0.2),
            index_extended=True,
            middle_extended=True,
            ring_folded=True,
            pinky_folded=True,
        )

        res = detector.evaluate(hands=[hand], forehead=forehead)
        assert res.is_valid
        assert res.right_hand_found
        assert res.index_extended
        assert res.middle_extended
        assert res.ring_folded
        assert res.pinky_folded
        assert res.forehead_touching

    def test_fingers_far_from_forehead(self):
        """Right hand with two extended fingers, but down at chest level (y=0.6)."""
        detector = GestureDetector()
        forehead = create_mock_forehead(center=(0.5, 0.2))
        hand = create_mock_hand(
            handedness="Right",
            offset_y=0.40,  # Entire hand lowered to chest level
            index_extended=True,
            middle_extended=True,
            ring_folded=True,
            pinky_folded=True,
        )

        res = detector.evaluate(hands=[hand], forehead=forehead)
        assert not res.is_valid
        assert not res.forehead_touching
        assert "closer to forehead" in res.message

    def test_left_hand_rejected_when_right_required(self):
        """User uses Left hand instead of Right hand."""
        detector = GestureDetector(require_right_hand=True)
        forehead = create_mock_forehead(center=(0.5, 0.2))
        hand = create_mock_hand(
            handedness="Left",
            index_tip_pos=(0.5, 0.2),
            middle_tip_pos=(0.5, 0.2),
        )

        res = detector.evaluate(hands=[hand], forehead=forehead)
        assert not res.is_valid
        assert not res.right_hand_found
        assert "Wrong hand" in res.message

    def test_ring_finger_extended_rejected(self):
        """User extends 3 fingers (index, middle, ring). Must be rejected."""
        detector = GestureDetector()
        forehead = create_mock_forehead(center=(0.5, 0.2))
        hand = create_mock_hand(
            handedness="Right",
            index_tip_pos=(0.5, 0.2),
            middle_tip_pos=(0.5, 0.2),
            ring_folded=False,  # Ring extended!
        )

        res = detector.evaluate(hands=[hand], forehead=forehead)
        assert not res.is_valid
        assert not res.ring_folded


class TestModelManager:
    def test_check_models_validation(self):
        """ModelManager must report status of required model files accurately."""
        valid, missing = ModelManager.check_models()
        # All models were downloaded earlier
        assert valid, f"Expected models to be ready, but got missing: {missing}"
        assert len(missing) == 0

        # Paths should be valid
        assert ModelManager.get_hand_model_path().exists()
        assert ModelManager.get_face_model_path().exists()
        assert ModelManager.get_segmenter_model_path().exists()
