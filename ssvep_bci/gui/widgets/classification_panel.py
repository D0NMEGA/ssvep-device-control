"""
Classification Panel Widget

Displays real-time SSVEP classification results:
- Correlation bars for each target frequency
- Current detection status
- Confidence and margin metrics
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import SSVEPConfig


class CorrelationBar(QWidget):
    """Single frequency correlation bar with label."""

    def __init__(self, frequency: float, color: str = "#4285f4"):
        super().__init__()

        self.frequency = frequency
        self.color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Frequency label
        self.freq_label = QLabel(f"{frequency:.2f} Hz")
        self.freq_label.setFixedWidth(70)
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.freq_label)

        # Progress bar for correlation
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%.3f")
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 3px;
                background: #f0f0f0;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress)

        # Correlation value
        self.value_label = QLabel("0.000")
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.value_label)

        # Detection indicator
        self.indicator = QLabel("")
        self.indicator.setFixedWidth(30)
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.indicator)

    def set_correlation(self, value: float, is_detected: bool = False):
        """Update correlation value and detection state.

        Args:
            value: Correlation value (0-1)
            is_detected: Whether this frequency is currently detected
        """
        # Clamp value
        value = max(0, min(1, value))

        self.progress.setValue(int(value * 100))
        self.value_label.setText(f"{value:.3f}")

        if is_detected:
            self.indicator.setText("[*]")
            self.indicator.setStyleSheet("color: #0f9d58; font-weight: bold;")
            self.progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #0f9d58;
                    border-radius: 3px;
                    background: #f0f0f0;
                    height: 20px;
                }}
                QProgressBar::chunk {{
                    background: #0f9d58;
                    border-radius: 2px;
                }}
            """)
        else:
            self.indicator.setText("")
            self.progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background: #f0f0f0;
                    height: 20px;
                }}
                QProgressBar::chunk {{
                    background: {self.color};
                    border-radius: 2px;
                }}
            """)


class ClassificationPanel(QWidget):
    """Panel showing classification results and metrics."""

    def __init__(self, config: SSVEPConfig = None):
        super().__init__()

        self.config = config or SSVEPConfig()

        # Colors for each frequency bar
        self.bar_colors = ["#4285f4", "#db4437", "#f4b400", "#0f9d58"]

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Classification")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Detection status (large display)
        self.status_frame = QFrame()
        self.status_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.status_frame.setStyleSheet("""
            QFrame {
                background: #f5f5f5;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(self.status_frame)

        self.status_label = QLabel("Waiting...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        status_layout.addWidget(self.status_label)

        self.detection_label = QLabel("No detection")
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("color: #666; font-size: 12px;")
        status_layout.addWidget(self.detection_label)

        layout.addWidget(self.status_frame)

        # Correlation bars group
        bars_group = QGroupBox("Correlations")
        bars_layout = QVBoxLayout(bars_group)

        self.bars = {}
        for i, freq in enumerate(self.config.target_frequencies):
            color = self.bar_colors[i % len(self.bar_colors)]
            bar = CorrelationBar(freq, color)
            self.bars[freq] = bar
            bars_layout.addWidget(bar)

        layout.addWidget(bars_group)

        # Metrics group
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        # Margin display
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("Margin:"))
        self.margin_label = QLabel("0.000")
        self.margin_label.setStyleSheet("font-weight: bold;")
        margin_row.addWidget(self.margin_label)
        margin_row.addWidget(QLabel(f"(threshold: {self.config.margin_threshold:.2f})"))
        margin_row.addStretch()
        metrics_layout.addLayout(margin_row)

        # Max correlation display
        max_corr_row = QHBoxLayout()
        max_corr_row.addWidget(QLabel("Max Corr:"))
        self.max_corr_label = QLabel("0.000")
        self.max_corr_label.setStyleSheet("font-weight: bold;")
        max_corr_row.addWidget(self.max_corr_label)
        max_corr_row.addWidget(QLabel(f"(threshold: {self.config.confidence_threshold:.2f})"))
        max_corr_row.addStretch()
        metrics_layout.addLayout(max_corr_row)

        # Window count
        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Windows:"))
        self.window_label = QLabel("0")
        self.window_label.setStyleSheet("font-weight: bold;")
        window_row.addWidget(self.window_label)
        window_row.addStretch()
        metrics_layout.addLayout(window_row)

        # Latency
        latency_row = QHBoxLayout()
        latency_row.addWidget(QLabel("Latency:"))
        self.latency_label = QLabel("-- ms")
        self.latency_label.setStyleSheet("font-weight: bold;")
        latency_row.addWidget(self.latency_label)
        latency_row.addStretch()
        metrics_layout.addLayout(latency_row)

        layout.addWidget(metrics_group)
        layout.addStretch()

    def update_result(
        self,
        correlations: Dict[float, float],
        detected_freq: Optional[float] = None,
        margin: float = 0.0,
        window_count: int = 0,
        latency_ms: float = 0.0
    ):
        """Update the panel with new classification result.

        Args:
            correlations: Dict mapping frequency -> correlation value
            detected_freq: Currently detected frequency (or None)
            margin: Margin between top-2 correlations
            window_count: Number of windows processed
            latency_ms: Processing latency in ms
        """
        # Update bars
        max_corr = 0
        for freq, bar in self.bars.items():
            corr = correlations.get(freq, 0)
            is_detected = (freq == detected_freq)
            bar.set_correlation(corr, is_detected)
            max_corr = max(max_corr, corr)

        # Update status
        if detected_freq is not None:
            self.status_label.setText(f"{detected_freq:.2f} Hz")
            self.status_label.setStyleSheet("color: #0f9d58;")
            self.detection_label.setText("DETECTED")
            self.detection_label.setStyleSheet("color: #0f9d58; font-weight: bold;")
            self.status_frame.setStyleSheet("""
                QFrame {
                    background: #e8f5e9;
                    border: 2px solid #0f9d58;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        else:
            self.status_label.setText("Searching...")
            self.status_label.setStyleSheet("color: #666;")
            self.detection_label.setText("No detection")
            self.detection_label.setStyleSheet("color: #666;")
            self.status_frame.setStyleSheet("""
                QFrame {
                    background: #f5f5f5;
                    border: 2px solid #ccc;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)

        # Update metrics
        self.margin_label.setText(f"{margin:.3f}")
        if margin >= self.config.margin_threshold:
            self.margin_label.setStyleSheet("color: #0f9d58; font-weight: bold;")
        else:
            self.margin_label.setStyleSheet("color: #db4437; font-weight: bold;")

        self.max_corr_label.setText(f"{max_corr:.3f}")
        if max_corr >= self.config.confidence_threshold:
            self.max_corr_label.setStyleSheet("color: #0f9d58; font-weight: bold;")
        else:
            self.max_corr_label.setStyleSheet("color: #db4437; font-weight: bold;")

        self.window_label.setText(str(window_count))
        self.latency_label.setText(f"{latency_ms:.1f} ms")

    def reset(self):
        """Reset to initial state."""
        for bar in self.bars.values():
            bar.set_correlation(0, False)
        self.status_label.setText("Waiting...")
        self.status_label.setStyleSheet("color: #666;")
        self.detection_label.setText("Not running")
        self.margin_label.setText("0.000")
        self.max_corr_label.setText("0.000")
        self.window_label.setText("0")
        self.latency_label.setText("-- ms")


# Test
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    import numpy as np
    import sys

    app = QApplication(sys.argv)

    widget = ClassificationPanel()
    widget.resize(350, 500)
    widget.show()

    # Simulate updates
    window_count = 0

    def update():
        global window_count
        window_count += 1

        # Generate fake correlations
        correlations = {
            15.0: np.random.uniform(0.3, 0.8),
            12.0: np.random.uniform(0.2, 0.6),
            10.0: np.random.uniform(0.2, 0.5),
            8.57: np.random.uniform(0.2, 0.5),
        }

        # Random detection
        detected = 15.0 if np.random.random() > 0.5 else None
        margin = max(correlations.values()) - sorted(correlations.values())[-2]

        widget.update_result(
            correlations=correlations,
            detected_freq=detected,
            margin=margin,
            window_count=window_count,
            latency_ms=np.random.uniform(2, 5)
        )

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(200)

    sys.exit(app.exec())
