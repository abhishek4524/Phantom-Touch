"""
Automated unit tests for BackgroundManager.
"""

import pytest
import numpy as np
from pathlib import Path

from phantomtouch.core.background_manager import BackgroundManager


class TestBackgroundManager:
    def test_missing_background_initial_state(self, tmp_path):
        bg_path = tmp_path / "nonexistent.png"
        mgr = BackgroundManager(bg_path)
        assert not mgr.has_background
        assert mgr.background_image is None

    def test_save_and_load_background(self, tmp_path):
        bg_path = tmp_path / "test_bg.png"
        mgr = BackgroundManager(bg_path)

        dummy_img = np.full((120, 160, 3), 150, dtype=np.uint8)
        assert mgr.save_background(dummy_img)
        assert mgr.has_background

        # Create fresh manager pointing to same path
        mgr2 = BackgroundManager(bg_path)
        assert mgr2.has_background
        assert mgr2.background_image.shape == (120, 160, 3)
        assert np.all(mgr2.background_image == 150)

    def test_multi_frame_median_combination(self, tmp_path):
        """Test temporal median computation from multiple frames."""
        bg_path = tmp_path / "median_bg.png"
        mgr = BackgroundManager(bg_path)

        mgr.start_capture_sequence()
        # Create 5 frames where most pixels are 100, but frame 2 has a transient spike (250)
        for i in range(5):
            val = 250 if i == 2 else 100
            frame = np.full((50, 50, 3), val, dtype=np.uint8)
            mgr.add_capture_frame(frame)

        success, bg = mgr.finish_capture()
        assert success
        assert bg is not None
        # Median of [100, 100, 250, 100, 100] is 100 (robust to spike!)
        assert np.all(bg == 100)

    def test_insufficient_frames_rejected(self, tmp_path):
        bg_path = tmp_path / "short_bg.png"
        mgr = BackgroundManager(bg_path)
        mgr.start_capture_sequence()
        mgr.add_capture_frame(np.full((50, 50, 3), 100, dtype=np.uint8))
        # Only 1 frame - should fail
        success, bg = mgr.finish_capture()
        assert not success
        assert bg is None

    def test_clear_background(self, tmp_path):
        bg_path = tmp_path / "clear_bg.png"
        mgr = BackgroundManager(bg_path)
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        mgr.save_background(img)
        assert mgr.has_background
        assert bg_path.exists()

        mgr.clear()
        assert not mgr.has_background
        assert not bg_path.exists()
