"""
PhantomTouch Desktop Application Entrypoint.
Initializes QApplication with high-DPI scaling and shows MainWindow.
"""

import sys
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from phantomtouch.app.main_window import MainWindow
from phantomtouch.utils.config import AppConfig
from phantomtouch.utils.logger import get_logger

logger = get_logger("Application")


def run() -> int:
    """Run the PhantomTouch desktop application."""
    # Suppress verbose TF/MediaPipe logs on stderr
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "2"

    logger.info("Starting PhantomTouch Desktop Application...")

    app = QApplication(sys.argv)
    app.setApplicationName("PhantomTouch")
    app.setOrganizationName("PhantomTouch")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
