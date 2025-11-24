"""
SSVEP BCI Main Window (Phase 2)

PyQt6-based GUI for the SSVEP Brain-Computer Interface.
This is a placeholder for Phase 2 implementation.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import SSVEPConfig


class MainWindow(QMainWindow):
    """Main window for SSVEP BCI application.

    Phase 2 placeholder - will implement:
    - Left panel: EEG time-series plot
    - Center panel: Classification status with correlation bars
    - Right panel: Controls and metrics
    """

    def __init__(self):
        super().__init__()

        self.config = SSVEPConfig()
        self.setWindowTitle("SSVEP BCI - Phase 2 Coming Soon")
        self.setMinimumSize(1200, 800)

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Placeholder message
        label = QLabel(
            "<h1>SSVEP BCI</h1>"
            "<h2>Phase 2: GUI Under Development</h2>"
            "<p>For now, use the CLI test:</p>"
            "<pre>python main.py --cli</pre>"
            "<p>Or directly:</p>"
            "<pre>python cli_test.py</pre>"
            "<br>"
            "<p>Target frequencies:</p>"
            "<ul>"
            "<li>8.57 Hz (Arduino Pin D2)</li>"
            "<li>10.0 Hz (Arduino Pin D3)</li>"
            "<li>12.0 Hz (Arduino Pin D4)</li>"
            "<li>15.0 Hz (Arduino Pin D5)</li>"
            "</ul>"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.cli_btn = QPushButton("Open CLI Test Info")
        self.cli_btn.clicked.connect(self._show_cli_info)
        btn_layout.addWidget(self.cli_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # Status bar
        self.statusBar().showMessage("Ready - Use CLI for Phase 1 testing")

    def _show_cli_info(self):
        """Show CLI usage information."""
        QMessageBox.information(
            self,
            "CLI Test Usage",
            "Run the CLI test from command line:\n\n"
            "Synthetic data (no hardware):\n"
            "  python cli_test.py\n"
            "  python cli_test.py --target 12.0\n\n"
            "With real Cyton:\n"
            "  python cli_test.py --port COM3\n\n"
            "Options:\n"
            "  --duration N    Run for N seconds\n"
            "  --confidence X  Set confidence threshold\n"
            "  --margin X      Set margin threshold\n"
            "  --no-log        Don't save logs"
        )


# Test
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
