#!/usr/bin/env python3
"""
SSVEP BCI - Complete PyQt6 GUI

Professional, integrated interface for calibration, BCI operation, 
real-time monitoring, and hardware troubleshooting.
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QMessageBox,
    QDialog, QProgressBar, QPlainTextEdit, QSplitter, QStatusBar,
    QGroupBox, QFormLayout, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtCharts import QChart, QChartView, QLineSeries
from PyQt6.QtCore import QPointF
from ssvep_bci.utils.config import SSVEPConfig, create_config
from ssvep_bci.models.eeg_buffer import EEGBuffer
from ssvep_bci.models.cca_decoder import SSVEPDecoder
from ssvep_bci.models.template_cca import TemplateCCADecoder, TemplateCCAConfig
from ssvep_bci.models.calibration import CalibrationCollector, CalibrationData
from ssvep_bci.drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from ssvep_bci.drivers.arduino_controller import ArduinoController, BCIFeedbackController
from ssvep_bci.drivers.data_logger import DataLogger



class EEGDisplayWidget(QWidget):
    """Displays 8-channel EEG in stacked vertical format (like clinical EEG)."""
    
    def __init__(self, config: SSVEPConfig = None):
        super().__init__()
        self.config = config or SSVEPConfig()
        self.setMinimumHeight(600)
        
        # Display parameters
        self.time_window_s = 5.0  # 5 second display window
        self.uv_per_div = 100.0   # µV per vertical division
        self.n_divs = 5            # Vertical divisions per channel
        
        # Channel buffers (store last 5 seconds)
        max_samples = int(self.time_window_s * self.config.fs)
        self.buffers = [deque(maxlen=max_samples) for _ in range(8)]
        
        # Channel labels
        self.channel_names = [
            "Pz", "P3", "P4", "PO3", "PO4", "O1", "Oz", "O2"
        ]
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create matplotlib-like plot using PyQt6 (simplified version)
        # For now, we'll create a text-based representation
        self.display_text = QPlainTextEdit()
        self.display_text.setReadOnly(True)
        self.display_text.setFont(QFont("Courier", 8))
        layout.addWidget(self.display_text)
    
    def update_eeg_data(self, data: np.ndarray):
        """Update with new EEG data.
        
        Args:
            data: Shape (8, n_samples)
        """
        for ch in range(min(8, data.shape[0])):
            for sample in data[ch]:
                self.buffers[ch].append(float(sample))
        
        self._redraw()
    
    def _redraw(self):
        """Redraw the EEG display."""
        lines = []
        lines.append("═" * 100)
        lines.append("EEG Display - 5 Second Window (100 µV/div)")
        lines.append("═" * 100)
        
        for ch in range(8):
            buf = list(self.buffers[ch])
            if not buf:
                continue
            
            # Normalize to µV/div
            normalized = [v / self.uv_per_div for v in buf]
            
            # Create simple ASCII plot
            baseline = 20
            plot_line = list(" " * 100)
            
            # Map samples to horizontal position
            step = max(1, len(normalized) // 100)
            for i, val in enumerate(normalized[::step]):
                x = min(i, 99)
                # Clamp to ±5 divs
                y = int(baseline - np.clip(val, -5, 5))
                if 0 <= y < len(plot_line):
                    plot_line[y] = "█"
            
            ch_line = f"{self.channel_names[ch]:3} | {''.join(plot_line)}"
            lines.append(ch_line)
        
        lines.append("═" * 100)
        self.display_text.setPlainText("\n".join(lines))
    
    def reset(self):
        """Clear all buffers."""
        for buf in self.buffers:
            buf.clear()
        self.display_text.setPlainText("")


class StartupDialog(QDialog):
    """Auto-running hardware troubleshooting dialog on startup."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSVEP BCI - Startup Diagnostics")
        self.setMinimumSize(600, 500)
        self.results = {}
        
        self.initUI()
        self.run_diagnostics()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("Hardware Diagnostics")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        button_layout = QHBoxLayout()
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(self.continue_btn)
        layout.addLayout(button_layout)
    
    def run_diagnostics(self):
        """Run hardware checks in background."""
        def diagnose():
            messages = []
            
            # Check Arduino
            messages.append("="*60)
            messages.append("CHECKING ARDUINO...")
            messages.append("="*60)
            
            arduino_port = ArduinoController.find_arduino()
            if arduino_port:
                messages.append(f"✓ Arduino found on {arduino_port}")
                self.results['arduino_port'] = arduino_port
                self.results['arduino_ok'] = True
            else:
                messages.append("⚠ Arduino not detected (optional - can continue)")
                self.results['arduino_ok'] = False
            
            self.progress.setValue(33)
            
            # Check Cyton
            messages.append("\n" + "="*60)
            messages.append("CHECKING CYTON EEG...")
            messages.append("="*60)
            
            cyton_port = BrainFlowDriver.auto_detect_cyton()
            if cyton_port:
                messages.append(f"✓ Cyton found on {cyton_port}")
                self.results['cyton_port'] = cyton_port
                self.results['cyton_ok'] = True
            else:
                messages.append("⚠ Cyton not detected (can use synthetic mode)")
                messages.append("Available ports:")
                for port, desc in BrainFlowDriver.list_ports():
                    messages.append(f"  {port}: {desc}")
                self.results['cyton_ok'] = False
            
            self.progress.setValue(66)
            
            # Check calibration data
            messages.append("\n" + "="*60)
            messages.append("CHECKING CALIBRATION DATA...")
            messages.append("="*60)
            
            cal_dir = Path("calibration")
            if cal_dir.exists():
                subjects = [d.name for d in cal_dir.iterdir() if d.is_dir()]
                messages.append(f"✓ Found {len(subjects)} subject(s)")
                for subj in subjects:
                    sessions = len(list((cal_dir / subj).glob("session_*.npz")))
                    messages.append(f"  {subj}: {sessions} session(s)")
                self.results['subjects'] = subjects
            else:
                messages.append("⚠ No calibration data yet (start with calibration mode)")
                self.results['subjects'] = []
            
            self.progress.setValue(100)
            
            messages.append("\n" + "="*60)
            messages.append("DIAGNOSTICS COMPLETE")
            messages.append("="*60)
            messages.append("\nYou can now proceed to the main interface.")
            
            self.log_text.setPlainText("\n".join(messages))
            self.continue_btn.setEnabled(True)
        
        thread = threading.Thread(target=diagnose, daemon=True)
        thread.start()


class SubjectManagerDialog(QDialog):
    """Subject selection and management."""
    
    def __init__(self, parent=None, subjects: List[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Subject Manager")
        self.setMinimumSize(400, 300)
        
        self.subjects = subjects or []
        self.selected_subject = None
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Current subject display
        current_label = QLabel("Current Subject:")
        current_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(current_label)
        
        self.current_subject_display = QLabel("None selected")
        self.current_subject_display.setFont(QFont("Courier", 11))
        self.current_subject_display.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.current_subject_display)
        
        # Subject selection
        select_label = QLabel("Select Subject:")
        select_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(select_label)
        
        self.subject_combo = QComboBox()
        self.subject_combo.addItem("--- New Subject ---")
        for subj in self.subjects:
            self.subject_combo.addItem(subj)
        self.subject_combo.currentTextChanged.connect(self._on_subject_changed)
        layout.addWidget(self.subject_combo)
        
        # Subject info
        info_label = QLabel("Subject Info:")
        layout.addWidget(info_label)
        
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        layout.addWidget(self.info_text)
        
        # New subject input
        layout.addSpacing(10)
        new_subj_label = QLabel("Or enter new subject name:")
        layout.addWidget(new_subj_label)
        
        self.new_subject_input = QComboBox()
        self.new_subject_input.setEditable(True)
        self.new_subject_input.setPlaceholderText("Enter name or select from list")
        layout.addWidget(self.new_subject_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Select")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn.clicked.connect(self._on_select)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
    
    def _on_subject_changed(self, subject: str):
        """Update info when subject selection changes."""
        if subject == "--- New Subject ---":
            self.info_text.setPlainText("Create a new subject for calibration")
        else:
            cal_dir = Path("calibration") / subject
            if cal_dir.exists():
                sessions = list(cal_dir.glob("session_*.npz"))
                info = f"Calibration Sessions: {len(sessions)}\n"
                info += "\nRecent sessions:\n"
                for sess in sorted(sessions)[-3:]:
                    info += f"  • {sess.stem}\n"
                self.info_text.setPlainText(info)
            else:
                self.info_text.setPlainText(f"Subject: {subject}")
    
    def _on_select(self):
        """Confirm selection."""
        combo_text = self.subject_combo.currentText()
        new_text = self.new_subject_input.currentText().strip()
        
        if new_text:
            self.selected_subject = new_text
        elif combo_text != "--- New Subject ---":
            self.selected_subject = combo_text
        
        if self.selected_subject:
            self.current_subject_display.setText(f"Selected: {self.selected_subject}")
            self.accept()
        else:
            QMessageBox.warning(self, "Invalid", "Please select or enter a subject name")


class DashboardTab(QWidget):
    """Mode selection dashboard."""
    
    mode_selected = pyqtSignal(str)  # Emits mode name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("SSVEP BCI - Select Operating Mode")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Modes grid
        modes_layout = QHBoxLayout()
        
        modes = [
            ("Calibration", "Collect calibration data\nfor personalized templates", "📊"),
            ("Standard BCI", "Run BCI with sine/cosine\nreference signals", "🎯"),
            ("Calibrated BCI", "Run BCI with personalized\ncalibration templates", "⭐"),
            ("Real-time Monitor", "Monitor signal quality and\noptimize performance", "📈"),
        ]
        
        for mode, desc, icon in modes:
            btn = QPushButton(f"{icon}\n{mode}\n\n{desc}")
            btn.setMinimumSize(200, 150)
            btn.setFont(QFont("Arial", 10))
            btn.clicked.connect(lambda checked, m=mode: self.mode_selected.emit(m))
            modes_layout.addWidget(btn)
        
        layout.addLayout(modes_layout)
        
        # Bottom section
        bottom_group = QGroupBox("Quick Setup")
        bottom_layout = QFormLayout()
        
        # Subject selector
        subj_layout = QHBoxLayout()
        self.subject_label = QLabel("None")
        self.subject_label.setFont(QFont("Courier", 10))
        self.change_subject_btn = QPushButton("Change Subject")
        subj_layout.addWidget(self.subject_label)
        subj_layout.addWidget(self.change_subject_btn)
        bottom_layout.addRow("Subject:", subj_layout)
        
        bottom_group.setLayout(bottom_layout)
        layout.addWidget(bottom_group)
        
        layout.addStretch()


class CalibrationTab(QWidget):
    """Calibration mode interface."""
    
    def __init__(self, parent=None, config: SSVEPConfig = None):
        super().__init__(parent)
        self.config = config or SSVEPConfig()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Settings
        settings_group = QGroupBox("Calibration Settings")
        settings_layout = QFormLayout()
        
        self.trials_spin = QSpinBox()
        self.trials_spin.setValue(5)
        self.trials_spin.setRange(1, 20)
        settings_layout.addRow("Trials per frequency:", self.trials_spin)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setValue(4.0)
        self.duration_spin.setRange(1.0, 10.0)
        settings_layout.addRow("Trial duration (s):", self.duration_spin)
        
        self.rest_spin = QDoubleSpinBox()
        self.rest_spin.setValue(2.0)
        self.rest_spin.setRange(0.5, 5.0)
        settings_layout.addRow("Rest duration (s):", self.rest_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Status display
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        # Controls
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Calibration")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)


class StandardBCITab(QWidget):
    """Standard BCI mode (sine/cosine references)."""
    
    def __init__(self, parent=None, config: SSVEPConfig = None):
        super().__init__(parent)
        self.config = config or SSVEPConfig()
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout(self)
        
        # Left: EEG display
        self.eeg_display = EEGDisplayWidget(self.config)
        layout.addWidget(self.eeg_display, 3)
        
        # Right: Classification panel
        control_layout = QVBoxLayout()
        
        # Threshold settings
        settings_group = QGroupBox("Classification Settings")
        settings_form = QFormLayout()
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setValue(0.55)
        self.confidence_spin.setRange(0.3, 0.8)
        self.confidence_spin.setSingleStep(0.05)
        settings_form.addRow("Confidence threshold:", self.confidence_spin)
        
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setValue(0.15)
        self.margin_spin.setRange(0.05, 0.3)
        self.margin_spin.setSingleStep(0.05)
        settings_form.addRow("Margin threshold:", self.margin_spin)
        
        settings_group.setLayout(settings_form)
        control_layout.addWidget(settings_group)
        
        # Results display
        results_group = QGroupBox("Classification Results")
        results_layout = QFormLayout()
        
        self.correlations_text = QPlainTextEdit()
        self.correlations_text.setReadOnly(True)
        self.correlations_text.setMaximumHeight(200)
        results_layout.addRow("Correlations:", self.correlations_text)
        
        self.detection_label = QLabel("Waiting...")
        self.detection_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        results_layout.addRow("Detection:", self.detection_label)
        
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)
        
        control_layout.addStretch()
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start BCI")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)
        
        layout.addLayout(control_layout, 1)


class CalibratedBCITab(StandardBCITab):
    """Calibrated BCI mode (with personalized templates)."""
    
    def __init__(self, parent=None, config: SSVEPConfig = None):
        super().__init__(parent, config)


class RealTimeMonitorTab(QWidget):
    """Real-time monitoring and optimization."""
    
    def __init__(self, parent=None, config: SSVEPConfig = None):
        super().__init__(parent)
        self.config = config or SSVEPConfig()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Info
        info = QLabel("Real-time signal monitoring and performance optimization")
        layout.addWidget(info)
        
        # Monitoring display
        self.monitor_table = QTableWidget()
        self.monitor_table.setColumnCount(3)
        self.monitor_table.setHorizontalHeaderLabels(["Frequency", "Avg Correlation", "Count"])
        layout.addWidget(self.monitor_table)
        
        # Status
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)
        
        # Controls
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Monitoring")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SSVEP BCI - Brain-Computer Interface")
        self.setMinimumSize(1600, 900)
        
        self.config = SSVEPConfig()
        self.selected_subject = None
        self.diagnostics = {}
        
        # Hardware components
        self.eeg_driver = None
        self.arduino = None
        self.decoder = None
        self.buffer = None
        
        self.is_running = False
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._process_step)
        
        self._setup_ui()
        self._run_startup_dialog()
    
    def _setup_ui(self):
        """Set up the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        
        # Header with subject info
        header = QHBoxLayout()
        self.subject_label = QLabel("Subject: None")
        self.subject_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.change_subject_btn = QPushButton("Change Subject")
        self.change_subject_btn.clicked.connect(self._on_change_subject)
        header.addWidget(self.subject_label)
        header.addStretch()
        header.addWidget(self.change_subject_btn)
        layout.addLayout(header)
        
        # Tabs
        self.tabs = QTabWidget()
        
        self.dashboard_tab = DashboardTab()
        self.dashboard_tab.mode_selected.connect(self._on_mode_selected)
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        
        self.calibration_tab = CalibrationTab(config=self.config)
        self.tabs.addTab(self.calibration_tab, "Calibration")
        
        self.standard_bci_tab = StandardBCITab(config=self.config)
        self.tabs.addTab(self.standard_bci_tab, "Standard BCI")
        
        self.calibrated_bci_tab = CalibratedBCITab(config=self.config)
        self.tabs.addTab(self.calibrated_bci_tab, "Calibrated BCI")
        
        self.monitor_tab = RealTimeMonitorTab(config=self.config)
        self.tabs.addTab(self.monitor_tab, "Real-time Monitor")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select mode from Dashboard")
    
    def _run_startup_dialog(self):
        """Run startup diagnostics."""
        dialog = StartupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.diagnostics = dialog.results
            # Set default ports if found
            if dialog.results.get('cyton_ok'):
                # Auto-connect to detected hardware
                pass
    
    def _on_change_subject(self):
        """Open subject manager."""
        subjects = self.diagnostics.get('subjects', [])
        dialog = SubjectManagerDialog(self, subjects)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_subject = dialog.selected_subject
            self.subject_label.setText(f"Subject: {self.selected_subject}")
            self.statusBar().showMessage(f"Subject set to: {self.selected_subject}")
    
    def _on_mode_selected(self, mode: str):
        """Handle mode selection from dashboard."""
        mode_map = {
            "Calibration": 1,
            "Standard BCI": 2,
            "Calibrated BCI": 3,
            "Real-time Monitor": 4
        }
        
        tab_index = mode_map.get(mode, 0)
        self.tabs.setCurrentIndex(tab_index)
        self.statusBar().showMessage(f"Mode: {mode}")
    
    def _process_step(self):
        """Process one step of the BCI loop."""
        pass
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.is_running:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "BCI is running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()