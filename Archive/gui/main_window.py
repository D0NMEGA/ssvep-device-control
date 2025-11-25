"""
SSVEP BCI Main Window

PyQt6-based GUI for the SSVEP Brain-Computer Interface.

Layout:
- Left panel: EEG signal plot (real-time scrolling)
- Center panel: Classification results (correlation bars)
- Right panel: Controls (start/stop, Arduino, calibration)
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import SSVEPConfig, create_config
from models.eeg_buffer import EEGBuffer
from models.cca_decoder import SSVEPDecoder
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController, BCIFeedbackController
from gui.widgets import SignalPlotWidget, ClassificationPanel, ControlPanel


class MainWindow(QMainWindow):
    """Main window for SSVEP BCI application."""

    def __init__(self):
        super().__init__()

        self.config = SSVEPConfig()
        self.setWindowTitle("SSVEP BCI - Real-time Classification")
        self.setMinimumSize(1400, 800)

        # BCI components (initialized on connect)
        self.buffer = None
        self.decoder = None
        self.eeg_driver = None
        self.arduino = None
        self.feedback = None

        # State
        self.is_running = False
        self.session_start_time = None
        self.window_count = 0
        self.correct_predictions = 0
        self.synthetic_target = 10.0

        # UI update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._process_step)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI components."""
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout with splitter
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: EEG signal plot
        self.signal_plot = SignalPlotWidget(self.config)
        self.signal_plot.setMinimumWidth(500)
        splitter.addWidget(self.signal_plot)

        # Center panel: Classification
        self.classification_panel = ClassificationPanel(self.config)
        self.classification_panel.setMinimumWidth(300)
        self.classification_panel.setMaximumWidth(400)
        splitter.addWidget(self.classification_panel)

        # Right panel: Controls
        self.control_panel = ControlPanel(self.config)
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(350)
        splitter.addWidget(self.control_panel)

        # Set splitter sizes
        splitter.setSizes([600, 350, 300])

        main_layout.addWidget(splitter)

        # Connect control panel signals
        self.control_panel.start_clicked.connect(self._on_start)
        self.control_panel.stop_clicked.connect(self._on_stop)
        self.control_panel.calibrate_clicked.connect(self._on_calibrate)
        self.control_panel.arduino_connect_clicked.connect(self._on_arduino_connect)
        self.control_panel.arduino_disconnect_clicked.connect(self._on_arduino_disconnect)
        self.control_panel.cyton_connect_clicked.connect(self._on_cyton_connect)
        self.control_panel.cyton_disconnect_clicked.connect(self._on_cyton_disconnect)
        self.control_panel.settings_changed.connect(self._on_settings_changed)

        # Status bar
        self.statusBar().showMessage("Ready - Connect to Arduino and Cyton to start")

    def _on_arduino_connect(self, port: str):
        """Handle Arduino connection."""
        try:
            self.arduino = ArduinoController(port)
            if self.arduino.connect():
                self.feedback = BCIFeedbackController(self.arduino)
                self.control_panel.set_arduino_connected(True, port)
                self.statusBar().showMessage(f"Arduino connected on {port}")
            else:
                QMessageBox.warning(self, "Connection Failed",
                                    f"Failed to connect to Arduino on {port}")
                self.arduino = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Arduino connection error: {e}")
            self.arduino = None

    def _on_arduino_disconnect(self):
        """Handle Arduino disconnection."""
        if self.arduino:
            self.arduino.disconnect()
            self.arduino = None
            self.feedback = None
        self.control_panel.set_arduino_connected(False)
        self.statusBar().showMessage("Arduino disconnected")

    def _on_cyton_connect(self, port: str):
        """Handle Cyton/EEG connection."""
        try:
            # Update config with settings
            settings = self.control_panel.get_settings()
            self.config = create_config(
                confidence_threshold=settings['confidence_threshold'],
                margin_threshold=settings['margin_threshold']
            )

            # Create components
            self.buffer = EEGBuffer(self.config)
            self.decoder = SSVEPDecoder(self.config)

            if port == "synthetic":
                self.eeg_driver = SyntheticSSVEPDriver(
                    self.config,
                    target_frequency=self.synthetic_target
                )
                mode = "Synthetic"
            else:
                self.eeg_driver = BrainFlowDriver(self.config)
                self.eeg_driver.config.serial_port = port
                mode = port

            if self.eeg_driver.connect():
                self.control_panel.set_cyton_connected(True, mode)
                self.statusBar().showMessage(f"EEG connected ({mode})")
            else:
                QMessageBox.warning(self, "Connection Failed",
                                    f"Failed to connect to EEG on {port}")
                self.eeg_driver = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"EEG connection error: {e}")
            self.eeg_driver = None

    def _on_cyton_disconnect(self):
        """Handle Cyton/EEG disconnection."""
        if self.is_running:
            self._on_stop()

        if self.eeg_driver:
            self.eeg_driver.disconnect()
            self.eeg_driver = None

        self.control_panel.set_cyton_connected(False)
        self.statusBar().showMessage("EEG disconnected")

    def _on_settings_changed(self, settings: dict):
        """Handle settings change."""
        self.config = create_config(
            confidence_threshold=settings['confidence_threshold'],
            margin_threshold=settings['margin_threshold']
        )

        # Update decoder if exists
        if self.decoder:
            self.decoder.config = self.config

        # Update classification panel
        self.classification_panel.config = self.config

    def _on_start(self):
        """Start BCI acquisition and classification."""
        if not self.eeg_driver:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to EEG device first")
            return

        try:
            # Start Arduino stimulation if connected
            if self.arduino:
                self.arduino.start_stimulation()

            # Start EEG streaming
            if not self.eeg_driver.start_stream():
                QMessageBox.warning(self, "Stream Error",
                                    "Failed to start EEG stream")
                return

            # Reset components
            self.buffer.reset()
            self.decoder.reset()
            self.signal_plot.reset()
            self.classification_panel.reset()

            # Reset counters
            self.window_count = 0
            self.correct_predictions = 0
            self.session_start_time = time.time()

            # Start processing
            self.is_running = True
            self.update_timer.start(20)  # 50 Hz update rate

            self.control_panel.set_running(True)
            self.statusBar().showMessage("BCI Running - Look at a flickering LED")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
            self._on_stop()

    def _on_stop(self):
        """Stop BCI acquisition."""
        self.is_running = False
        self.update_timer.stop()

        if self.eeg_driver:
            self.eeg_driver.stop_stream()

        if self.arduino:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()

        self.control_panel.set_running(False)
        self.statusBar().showMessage("BCI Stopped")

    def _on_calibrate(self):
        """Open calibration dialog."""
        QMessageBox.information(
            self,
            "Calibration",
            "Calibration mode coming soon!\n\n"
            "This will collect training data for each frequency\n"
            "and create personalized templates for improved accuracy."
        )

    def _process_step(self):
        """Process one step of the BCI loop."""
        if not self.is_running or not self.eeg_driver:
            return

        # Get EEG data
        data = self.eeg_driver.get_data()

        if data is not None and data.shape[1] > 0:
            # Update signal plot
            self.signal_plot.update_data(data)

            # Add to buffer
            self.buffer.append(data)

        # Process available windows
        while self.buffer.ready():
            window = self.buffer.get_window()
            if window is None:
                break

            # Classify
            result = self.decoder.step(window)
            self.window_count += 1

            # Send feedback if Arduino connected
            if result.committed_prediction is not None and self.feedback:
                self.feedback.on_prediction(result.committed_prediction)

                # Track accuracy for synthetic mode
                if self.control_panel.is_synthetic_mode():
                    if result.committed_prediction == self.synthetic_target:
                        self.correct_predictions += 1

            # Update classification panel
            self.classification_panel.update_result(
                correlations=result.correlations,
                detected_freq=result.committed_prediction,
                margin=result.margin,
                window_count=self.window_count,
                latency_ms=result.processing_time_ms
            )

        # Update session stats
        if self.session_start_time:
            duration = int(time.time() - self.session_start_time)
            accuracy = None
            if self.control_panel.is_synthetic_mode() and self.window_count > 0:
                accuracy = (self.correct_predictions / self.window_count) * 100
            self.control_panel.update_session_stats(duration, accuracy)

    def closeEvent(self, event):
        """Handle window close."""
        if self.is_running:
            self._on_stop()

        if self.arduino:
            self.arduino.disconnect()

        if self.eeg_driver:
            self.eeg_driver.disconnect()

        event.accept()


# Test
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
