"""
Control Panel Widget

Controls for the SSVEP BCI system:
- Start/Stop buttons
- Arduino connection and control
- Port selection
- Calibration mode trigger
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import SSVEPConfig


class ControlPanel(QWidget):
    """Control panel for BCI system operation."""

    # Signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    arduino_connect_clicked = pyqtSignal(str)  # port
    arduino_disconnect_clicked = pyqtSignal()
    cyton_connect_clicked = pyqtSignal(str)  # port
    cyton_disconnect_clicked = pyqtSignal()
    settings_changed = pyqtSignal(dict)

    def __init__(self, config: SSVEPConfig = None):
        super().__init__()

        self.config = config or SSVEPConfig()
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Controls")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Main control buttons
        control_group = QGroupBox("BCI Control")
        control_layout = QVBoxLayout(control_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("START")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #0f9d58;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #0d8c4d;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #db4437;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #c33c30;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_layout.addWidget(self.stop_btn)

        control_layout.addLayout(btn_layout)

        # Calibration button
        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.setStyleSheet("""
            QPushButton {
                background: #4285f4;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #3b78dc;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.calibrate_btn.clicked.connect(self.calibrate_clicked.emit)
        control_layout.addWidget(self.calibrate_btn)

        layout.addWidget(control_group)

        # Arduino connection
        arduino_group = QGroupBox("Arduino")
        arduino_layout = QVBoxLayout(arduino_group)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.arduino_port = QComboBox()
        self.arduino_port.setEditable(True)
        self.arduino_port.setMinimumWidth(100)
        port_row.addWidget(self.arduino_port)

        self.arduino_refresh_btn = QPushButton("Refresh")
        self.arduino_refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.arduino_refresh_btn)
        arduino_layout.addLayout(port_row)

        conn_row = QHBoxLayout()
        self.arduino_connect_btn = QPushButton("Connect")
        self.arduino_connect_btn.clicked.connect(self._on_arduino_connect)
        conn_row.addWidget(self.arduino_connect_btn)

        self.arduino_disconnect_btn = QPushButton("Disconnect")
        self.arduino_disconnect_btn.setEnabled(False)
        self.arduino_disconnect_btn.clicked.connect(self.arduino_disconnect_clicked.emit)
        conn_row.addWidget(self.arduino_disconnect_btn)
        arduino_layout.addLayout(conn_row)

        # Arduino status
        self.arduino_status = QLabel("Not connected")
        self.arduino_status.setStyleSheet("color: #666; font-style: italic;")
        arduino_layout.addWidget(self.arduino_status)

        layout.addWidget(arduino_group)

        # Cyton connection
        cyton_group = QGroupBox("OpenBCI Cyton")
        cyton_layout = QVBoxLayout(cyton_group)

        port_row2 = QHBoxLayout()
        port_row2.addWidget(QLabel("Port:"))
        self.cyton_port = QComboBox()
        self.cyton_port.setEditable(True)
        self.cyton_port.setMinimumWidth(100)
        port_row2.addWidget(self.cyton_port)
        cyton_layout.addLayout(port_row2)

        # Synthetic mode checkbox
        self.synthetic_check = QCheckBox("Use Synthetic Data")
        self.synthetic_check.setChecked(False)
        cyton_layout.addWidget(self.synthetic_check)

        conn_row2 = QHBoxLayout()
        self.cyton_connect_btn = QPushButton("Connect")
        self.cyton_connect_btn.clicked.connect(self._on_cyton_connect)
        conn_row2.addWidget(self.cyton_connect_btn)

        self.cyton_disconnect_btn = QPushButton("Disconnect")
        self.cyton_disconnect_btn.setEnabled(False)
        self.cyton_disconnect_btn.clicked.connect(self.cyton_disconnect_clicked.emit)
        conn_row2.addWidget(self.cyton_disconnect_btn)
        cyton_layout.addLayout(conn_row2)

        # Cyton status
        self.cyton_status = QLabel("Not connected")
        self.cyton_status.setStyleSheet("color: #666; font-style: italic;")
        cyton_layout.addWidget(self.cyton_status)

        layout.addWidget(cyton_group)

        # Settings
        settings_group = QGroupBox("Thresholds")
        settings_layout = QVBoxLayout(settings_group)

        # Confidence threshold
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.config.confidence_threshold)
        self.confidence_spin.valueChanged.connect(self._on_settings_changed)
        conf_row.addWidget(self.confidence_spin)
        settings_layout.addLayout(conf_row)

        # Margin threshold
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("Margin:"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 0.5)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setValue(self.config.margin_threshold)
        self.margin_spin.valueChanged.connect(self._on_settings_changed)
        margin_row.addWidget(self.margin_spin)
        settings_layout.addLayout(margin_row)

        layout.addWidget(settings_group)

        # Session stats
        stats_group = QGroupBox("Session")
        stats_layout = QVBoxLayout(stats_group)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Duration:"))
        self.duration_label = QLabel("00:00")
        self.duration_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        duration_row.addWidget(self.duration_label)
        duration_row.addStretch()
        stats_layout.addLayout(duration_row)

        accuracy_row = QHBoxLayout()
        accuracy_row.addWidget(QLabel("Accuracy:"))
        self.accuracy_label = QLabel("--%")
        self.accuracy_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        accuracy_row.addWidget(self.accuracy_label)
        accuracy_row.addStretch()
        stats_layout.addLayout(accuracy_row)

        layout.addWidget(stats_group)

        layout.addStretch()

        # Initial port refresh
        self._refresh_ports()

    def _refresh_ports(self):
        """Refresh available serial ports."""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())

            # Update Arduino combo
            current_arduino = self.arduino_port.currentText()
            self.arduino_port.clear()
            for port in ports:
                self.arduino_port.addItem(port.device)
            if current_arduino:
                idx = self.arduino_port.findText(current_arduino)
                if idx >= 0:
                    self.arduino_port.setCurrentIndex(idx)

            # Update Cyton combo
            current_cyton = self.cyton_port.currentText()
            self.cyton_port.clear()
            for port in ports:
                self.cyton_port.addItem(port.device)
            if current_cyton:
                idx = self.cyton_port.findText(current_cyton)
                if idx >= 0:
                    self.cyton_port.setCurrentIndex(idx)

        except Exception as e:
            print(f"Error refreshing ports: {e}")

    def _on_arduino_connect(self):
        """Handle Arduino connect button click."""
        port = self.arduino_port.currentText()
        if port:
            self.arduino_connect_clicked.emit(port)

    def _on_cyton_connect(self):
        """Handle Cyton connect button click."""
        if self.synthetic_check.isChecked():
            self.cyton_connect_clicked.emit("synthetic")
        else:
            port = self.cyton_port.currentText()
            if port:
                self.cyton_connect_clicked.emit(port)

    def _on_settings_changed(self):
        """Emit settings changed signal."""
        settings = {
            'confidence_threshold': self.confidence_spin.value(),
            'margin_threshold': self.margin_spin.value(),
        }
        self.settings_changed.emit(settings)

    def set_running(self, running: bool):
        """Update UI for running/stopped state."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.calibrate_btn.setEnabled(not running)

    def set_arduino_connected(self, connected: bool, port: str = ""):
        """Update Arduino connection status."""
        self.arduino_connect_btn.setEnabled(not connected)
        self.arduino_disconnect_btn.setEnabled(connected)
        self.arduino_port.setEnabled(not connected)
        self.arduino_refresh_btn.setEnabled(not connected)

        if connected:
            self.arduino_status.setText(f"Connected on {port}")
            self.arduino_status.setStyleSheet("color: #0f9d58; font-weight: bold;")
        else:
            self.arduino_status.setText("Not connected")
            self.arduino_status.setStyleSheet("color: #666; font-style: italic;")

    def set_cyton_connected(self, connected: bool, mode: str = ""):
        """Update Cyton connection status."""
        self.cyton_connect_btn.setEnabled(not connected)
        self.cyton_disconnect_btn.setEnabled(connected)
        self.cyton_port.setEnabled(not connected)
        self.synthetic_check.setEnabled(not connected)

        if connected:
            self.cyton_status.setText(f"Connected ({mode})")
            self.cyton_status.setStyleSheet("color: #0f9d58; font-weight: bold;")
        else:
            self.cyton_status.setText("Not connected")
            self.cyton_status.setStyleSheet("color: #666; font-style: italic;")

    def update_session_stats(self, duration_seconds: int, accuracy: float = None):
        """Update session statistics display.

        Args:
            duration_seconds: Session duration in seconds
            accuracy: Classification accuracy (0-100) or None
        """
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        self.duration_label.setText(f"{minutes:02d}:{seconds:02d}")

        if accuracy is not None:
            self.accuracy_label.setText(f"{accuracy:.1f}%")
        else:
            self.accuracy_label.setText("--%")

    def is_synthetic_mode(self) -> bool:
        """Check if synthetic mode is selected."""
        return self.synthetic_check.isChecked()

    def get_settings(self) -> dict:
        """Get current settings values."""
        return {
            'confidence_threshold': self.confidence_spin.value(),
            'margin_threshold': self.margin_spin.value(),
        }


# Test
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    widget = ControlPanel()
    widget.resize(300, 600)
    widget.show()

    # Connect signals for testing
    widget.start_clicked.connect(lambda: print("Start clicked"))
    widget.stop_clicked.connect(lambda: print("Stop clicked"))
    widget.calibrate_clicked.connect(lambda: print("Calibrate clicked"))
    widget.arduino_connect_clicked.connect(lambda p: print(f"Arduino connect: {p}"))
    widget.cyton_connect_clicked.connect(lambda p: print(f"Cyton connect: {p}"))
    widget.settings_changed.connect(lambda s: print(f"Settings: {s}"))

    sys.exit(app.exec())
