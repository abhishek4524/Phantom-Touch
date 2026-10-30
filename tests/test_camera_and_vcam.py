"""
Automated unit tests for WebcamManager and VirtualCameraManager.
"""

import pytest
import numpy as np

from phantomtouch.camera.webcam_manager import WebcamManager
from phantomtouch.virtual_camera.virtual_camera_manager import VirtualCameraManager


class TestWebcamManager:
    def test_initialization(self):
        wm = WebcamManager(camera_index=0, width=640, height=480)
        assert wm.camera_index == 0
        assert wm.width == 640
        assert wm.height == 480
        assert not wm.is_running
        assert wm.actual_fps == 0.0

    def test_read_when_not_started(self):
        wm = WebcamManager()
        ret, frame, ts = wm.read()
        assert not ret
        assert frame is None
        assert ts == 0.0

    def test_set_camera_index_offline(self):
        wm = WebcamManager(camera_index=0)
        assert wm.set_camera_index(2)
        assert wm.camera_index == 2
        assert not wm.is_running


class TestVirtualCameraManager:
    def test_initialization(self):
        vcam = VirtualCameraManager(width=640, height=480, fps=30)
        assert not vcam.is_active
        assert vcam.width == 640
        assert vcam.height == 480

    def test_send_frame_inactive_returns_false(self):
        vcam = VirtualCameraManager()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        assert not vcam.send_frame(dummy)

    def test_graceful_missing_driver_handling(self):
        """When hardware OBS virtual camera driver is missing, start() must return False with message without crashing."""
        vcam = VirtualCameraManager()
        ok, msg = vcam.start()
        # Either succeeds if OBS driver is present on host, or returns False with helpful instructions
        if not ok:
            assert "Virtual camera" in msg or "pyvirtualcam" in msg
            assert not vcam.is_active
            assert vcam.error_message is not None
        vcam.stop()
        assert not vcam.is_active
