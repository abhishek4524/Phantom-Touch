"""
Automated unit tests for VideoCompositor and MaskProcessor.
"""

import pytest
import numpy as np
import cv2

from phantomtouch.core.compositor import VideoCompositor
from phantomtouch.vision.mask_processor import MaskProcessor


class TestVideoCompositor:
    def test_normal_mode_returns_original_frame(self):
        """When invisibility is inactive, compositor must return the original frame."""
        compositor = VideoCompositor(transition_speed=1.0)
        orig = np.full((100, 100, 3), 120, dtype=np.uint8)
        bg = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.float32)

        result = compositor.composite(
            original_frame=orig,
            background_frame=bg,
            person_mask=mask,
            invisibility_active=False,
        )
        assert np.array_equal(result, orig), "Expected original frame when invisibility is inactive"

    def test_missing_background_returns_original(self):
        """When background is None, compositor must safely return original frame."""
        compositor = VideoCompositor(transition_speed=1.0)
        orig = np.full((100, 100, 3), 120, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.float32)

        result = compositor.composite(
            original_frame=orig,
            background_frame=None,
            person_mask=mask,
            invisibility_active=True,
        )
        assert np.array_equal(result, orig), "Expected original frame when background is missing"

    def test_full_invisibility_compositing(self):
        """With 100% person mask and full transition, person area becomes background."""
        # Using transition_speed=1.0 for instantaneous activation
        compositor = VideoCompositor(transition_speed=1.0)
        orig = np.full((100, 100, 3), 50, dtype=np.uint8)
        bg = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Person in left half (mask=1.0), empty background in right half (mask=0.0)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[:, :50] = 1.0

        result = compositor.composite(
            original_frame=orig,
            background_frame=bg,
            person_mask=mask,
            invisibility_active=True,
        )

        # Left half should match background (200)
        assert np.all(result[:, :50] == 200), "Left half (person) should be replaced by background"
        # Right half should remain original (50)
        assert np.all(result[:, 50:] == 50), "Right half (already background) should remain original"

    def test_feathered_alpha_blending(self):
        """Test linear alpha blending: Orig * (1 - Mask) + Bg * Mask."""
        compositor = VideoCompositor(transition_speed=1.0)
        orig = np.full((10, 10, 3), 100, dtype=np.uint8)
        bg = np.full((10, 10, 3), 200, dtype=np.uint8)
        # 50% transparency mask
        mask = np.full((10, 10), 0.5, dtype=np.float32)

        result = compositor.composite(
            original_frame=orig,
            background_frame=bg,
            person_mask=mask,
            invisibility_active=True,
        )
        # 100 * 0.5 + 200 * 0.5 = 150
        assert np.allclose(result, 150, atol=1), "Expected 50% blend to equal ~150"

    def test_resolution_alignment(self):
        """Test automatic resizing when background and mask dimensions differ from frame."""
        compositor = VideoCompositor(transition_speed=1.0)
        orig = np.zeros((200, 300, 3), dtype=np.uint8)
        bg = np.full((100, 150, 3), 180, dtype=np.uint8)  # Different shape
        mask = np.ones((50, 75), dtype=np.float32)  # Different shape

        result = compositor.composite(
            original_frame=orig,
            background_frame=bg,
            person_mask=mask,
            invisibility_active=True,
        )
        assert result.shape == orig.shape, "Output shape must match original frame"
        assert np.all(result == 180), "Mismatched background should be resized and applied"


class TestMaskProcessor:
    def test_mask_thresholding(self):
        processor = MaskProcessor(threshold=0.5, feather_radius=0, temporal_alpha=1.0, remove_small_artifacts=False)
        raw = np.array([[0.2, 0.4], [0.6, 0.9]], dtype=np.float32)
        out = processor.process(raw, target_shape=(2, 2))
        assert out[0, 0] == 0.0
        assert out[0, 1] == 0.0
        assert out[1, 0] == 0.6
        assert out[1, 1] == 0.9

    def test_temporal_smoothing(self):
        processor = MaskProcessor(threshold=0.5, feather_radius=0, temporal_alpha=0.5, remove_small_artifacts=False)
        # Frame 1: all 1.0
        f1 = np.ones((10, 10), dtype=np.float32)
        out1 = processor.process(f1, target_shape=(10, 10))
        assert np.allclose(out1, 1.0)

        # Frame 2: all 0.0 -> smoothed = 0.5 * 0 + 0.5 * 1.0 = 0.5
        f2 = np.zeros((10, 10), dtype=np.float32)
        out2 = processor.process(f2, target_shape=(10, 10))
        assert np.allclose(out2, 0.5)
