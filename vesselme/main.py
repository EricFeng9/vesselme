from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from vesselme.ui.main_window import MainWindow


def run() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
