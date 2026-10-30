"""
PhantomTouch Gesture Detector.
Recognizes the two-finger forehead invisibility gesture using 3D hand landmarks
and face forehead ROI.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from phantomtouch.vision.hand_detector import HandLandmarks
from phantomtouch.vision.face_detector import ForeheadROI
from phantomtouch.utils.logger import get_logger

logger = get_logger("GestureDetector")


@dataclass
class GestureResult:
    """Detailed diagnostics of gesture evaluation."""
    is_valid: bool
    right_hand_found: bool
    index_extended: bool
    middle_extended: bool
    ring_folded: bool
    pinky_folded: bool
    forehead_touching: bool
    index_forehead_dist: float
    middle_forehead_dist: float
    message: str


class GestureDetector:
    """
    Evaluates hand landmarks and forehead ROI to determine whether
    the invisibility trigger gesture is active.
    """

    def __init__(
        self,
        forehead_dist_threshold: float = 0.45,
        finger_straight_ratio: float = 1.15,
        require_right_hand: bool = True,
    ):
        self.forehead_dist_threshold = forehead_dist_threshold
        self.finger_straight_ratio = finger_straight_ratio
        self.require_right_hand = require_right_hand

    @staticmethod
    def _dist_2d(p1: Tuple[float, float, ...], p2: Tuple[float, float, ...]) -> float:
        """Euclidean distance in normalized (x, y) coordinates."""
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def is_finger_extended(
        self,
        landmarks: List[Tuple[float, float, float]],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
        wrist_idx: int = 0,
    ) -> bool:
        """
        Robust geometric check for finger extension.
        A finger is extended if its tip is significantly farther from wrist and MCP
        than the intermediate PIP joint is.
        """
        wrist = landmarks[wrist_idx]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]

        d_wrist_tip = self._dist_2d(wrist, tip)
        d_wrist_pip = self._dist_2d(wrist, pip)

        d_mcp_tip = self._dist_2d(mcp, tip)
        d_mcp_pip = self._dist_2d(mcp, pip)

        # Tip must be further from wrist and MCP than PIP
        is_extended_from_wrist = d_wrist_tip > d_wrist_pip * 1.05
        is_extended_from_mcp = d_mcp_tip > d_mcp_pip * self.finger_straight_ratio

        return bool(is_extended_from_wrist and is_extended_from_mcp)

    def is_finger_folded(
        self,
        landmarks: List[Tuple[float, float, float]],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
        wrist_idx: int = 0,
    ) -> bool:
        """
        Check if a finger is curled/folded into palm.
        Tip is closer to wrist/MCP than when extended.
        """
        wrist = landmarks[wrist_idx]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]

        d_wrist_tip = self._dist_2d(wrist, tip)
        d_wrist_pip = self._dist_2d(wrist, pip)
        d_mcp_tip = self._dist_2d(mcp, tip)
        d_mcp_pip = self._dist_2d(mcp, pip)

        # Tip is curled inward towards palm/wrist
        return bool(d_wrist_tip <= d_wrist_pip * 1.15 or d_mcp_tip <= d_mcp_pip * 1.05)

    def _dist_to_segment(
        self,
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
    ) -> float:
        """Distance from a point to a 2D line segment."""
        px, py = point[:2]
        x1, y1 = seg_start[:2]
        x2, y2 = seg_end[:2]

        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx**2 + dy**2
        if length_sq < 1e-6:
            return float(np.hypot(px - x1, py - y1))

        # Project point onto segment, clamped to [0.0, 1.0]
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    def evaluate(
        self,
        hands: List[HandLandmarks],
        forehead: Optional[ForeheadROI],
    ) -> GestureResult:
        """
        Evaluate candidate hands against forehead ROI.
        """
        if not hands:
            return GestureResult(
                is_valid=False,
                right_hand_found=False,
                index_extended=False,
                middle_extended=False,
                ring_folded=False,
                pinky_folded=False,
                forehead_touching=False,
                index_forehead_dist=999.0,
                middle_forehead_dist=999.0,
                message="No hands detected",
            )

        if forehead is None:
            return GestureResult(
                is_valid=False,
                right_hand_found=False,
                index_extended=False,
                middle_extended=False,
                ring_folded=False,
                pinky_folded=False,
                forehead_touching=False,
                index_forehead_dist=999.0,
                middle_forehead_dist=999.0,
                message="Face not detected (forehead required)",
            )

        # Test each detected hand (prioritize Right hand if required)
        best_candidate: Optional[GestureResult] = None

        for hand in hands:
            is_right = (hand.handedness.lower() == "right")
            if self.require_right_hand and not is_right:
                # If only one hand detected and require_right_hand is set, we still check
                # but flag right_hand_found according to handedness score
                pass

            lms = hand.landmarks
            if len(lms) < 21:
                continue

            # 1. Finger extensions:
            # Index (MCP 5, PIP 6, DIP 7, TIP 8)
            index_ext = self.is_finger_extended(lms, mcp_idx=5, pip_idx=6, tip_idx=8)

            # Middle (MCP 9, PIP 10, DIP 11, TIP 12)
            middle_ext = self.is_finger_extended(lms, mcp_idx=9, pip_idx=10, tip_idx=12)

            # Ring (MCP 13, PIP 14, DIP 15, TIP 16) - should be folded
            ring_fold = self.is_finger_folded(lms, mcp_idx=13, pip_idx=14, tip_idx=16)

            # Pinky (MCP 17, PIP 18, DIP 19, TIP 20) - should be folded
            pinky_fold = self.is_finger_folded(lms, mcp_idx=17, pip_idx=18, tip_idx=20)

            # 2. Forehead Proximity:
            # Distance from index and middle tips to forehead midline (glabella to hairline)
            # or forehead center
            idx_tip = (hand.index_tip[0], hand.index_tip[1])
            mid_tip = (hand.middle_tip[0], hand.middle_tip[1])

            dist_idx_seg = self._dist_to_segment(idx_tip, forehead.glabella, forehead.hairline)
            dist_mid_seg = self._dist_to_segment(mid_tip, forehead.glabella, forehead.hairline)

            # Also check distance to forehead center circle
            dist_idx_center = self._dist_2d(idx_tip, (forehead.center_x, forehead.center_y))
            dist_mid_center = self._dist_2d(mid_tip, (forehead.center_x, forehead.center_y))

            min_idx_dist = min(dist_idx_seg, dist_idx_center)
            min_mid_dist = min(dist_mid_seg, dist_mid_center)

            # Normalize by face scale
            norm_idx_dist = min_idx_dist / forehead.face_scale
            norm_mid_dist = min_mid_dist / forehead.face_scale

            touching = (
                norm_idx_dist <= self.forehead_dist_threshold and
                norm_mid_dist <= self.forehead_dist_threshold
            )

            # Overall validity check
            hand_ok = (not self.require_right_hand) or is_right
            gesture_valid = (
                hand_ok and
                index_ext and
                middle_ext and
                ring_fold and
                pinky_fold and
                touching
            )

            # Formulate user diagnostic message
            if gesture_valid:
                msg = "Gesture valid! (Invisibility ready)"
            elif not hand_ok:
                msg = f"Wrong hand detected ({hand.handedness}, right hand required)"
            elif not (index_ext and middle_ext):
                msg = "Extend both index and middle fingers"
            elif not (ring_fold and pinky_fold):
                msg = "Fold ring and little fingers into palm"
            elif not touching:
                msg = "Move fingers closer to forehead"
            else:
                msg = "Adjust hand gesture"

            candidate = GestureResult(
                is_valid=gesture_valid,
                right_hand_found=is_right,
                index_extended=index_ext,
                middle_extended=middle_ext,
                ring_folded=ring_fold,
                pinky_folded=pinky_fold,
                forehead_touching=touching,
                index_forehead_dist=norm_idx_dist,
                middle_forehead_dist=norm_mid_dist,
                message=msg,
            )

            if gesture_valid:
                return candidate
            if best_candidate is None:
                best_candidate = candidate

        return best_candidate or GestureResult(
            is_valid=False,
            right_hand_found=False,
            index_extended=False,
            middle_extended=False,
            ring_folded=False,
            pinky_folded=False,
            forehead_touching=False,
            index_forehead_dist=999.0,
            middle_forehead_dist=999.0,
            message="No valid gesture",
        )
