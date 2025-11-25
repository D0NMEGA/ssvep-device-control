"""
EEG Signal Plot Widget

Real-time scrolling plot of EEG channels using pyqtgraph.
Shows the last N seconds of data for all channels.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import SSVEPConfig


class SignalPlotWidget(QWidget):
    """Real-time EEG signal visualization widget."""

    def __init__(self, config: SSVEPConfig = None, display_seconds: float = 3.0):
        """Initialize signal plot widget.

        Args:
            config: SSVEPConfig instance
            display_seconds: How many seconds of data to display
        """
        super().__init__()

        self.config = config or SSVEPConfig()
        self.display_seconds = display_seconds
        self.n_samples = int(display_seconds * self.config.fs)
        self.n_channels = len(self.config.eeg_channels)

        # Data buffer
        self.data = np.zeros((self.n_channels, self.n_samples))

        # Channel colors (colorblind-friendly palette)
        self.colors = [
            (66, 133, 244),   # Blue
            (219, 68, 55),    # Red
            (244, 180, 0),    # Yellow
            (15, 157, 88),    # Green
            (171, 71, 188),   # Purple
            (255, 112, 67),   # Orange
            (0, 172, 193),    # Cyan
            (124, 77, 255),   # Violet
        ]

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("EEG Signals")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, background='w', foreground='k')

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Channel')
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # Time axis
        self.time_axis = np.linspace(-self.display_seconds, 0, self.n_samples)

        # Create plot curves for each channel
        self.curves = []
        spacing = 150  # uV spacing between channels

        for i in range(self.n_channels):
            color = self.colors[i % len(self.colors)]
            pen = pg.mkPen(color=color, width=1.5)
            curve = self.plot_widget.plot(
                self.time_axis,
                np.zeros(self.n_samples) + (self.n_channels - 1 - i) * spacing,
                pen=pen,
                name=self.config.electrode_names[i] if i < len(self.config.electrode_names) else f"Ch{i}"
            )
            self.curves.append(curve)

        # Set y-axis range
        self.plot_widget.setYRange(-spacing, self.n_channels * spacing)

        # Add y-axis labels for channels
        y_axis = self.plot_widget.getAxis('left')
        ticks = [(i * spacing, self.config.electrode_names[self.n_channels - 1 - i])
                 for i in range(self.n_channels)]
        y_axis.setTicks([ticks])

        layout.addWidget(self.plot_widget)

        # Legend
        legend_text = " | ".join([
            f'<span style="color: rgb{self.colors[i]};">{name}</span>'
            for i, name in enumerate(self.config.electrode_names[:self.n_channels])
        ])
        legend = QLabel(legend_text)
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(legend)

    def update_data(self, new_data: np.ndarray):
        """Update plot with new EEG data.

        Args:
            new_data: Array of shape (n_channels, n_samples)
        """
        if new_data is None or new_data.size == 0:
            return

        n_new = new_data.shape[1]

        # Shift existing data left and add new data
        if n_new >= self.n_samples:
            # If new data is larger than buffer, just take the last n_samples
            self.data = new_data[:, -self.n_samples:]
        else:
            # Shift left and append
            self.data = np.roll(self.data, -n_new, axis=1)
            self.data[:, -n_new:] = new_data

        # Update curves
        spacing = 150
        for i in range(self.n_channels):
            # Scale data to microvolts (assuming BrainFlow returns uV)
            channel_data = self.data[i, :] + (self.n_channels - 1 - i) * spacing
            self.curves[i].setData(self.time_axis, channel_data)

    def reset(self):
        """Reset the display buffer."""
        self.data = np.zeros((self.n_channels, self.n_samples))
        for curve in self.curves:
            curve.setData(self.time_axis, np.zeros(self.n_samples))


# Test
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    import sys

    app = QApplication(sys.argv)

    widget = SignalPlotWidget()
    widget.resize(600, 400)
    widget.show()

    # Simulate data updates
    def update():
        # Generate fake EEG data
        data = np.random.randn(8, 25) * 20  # 100ms of data at 250Hz
        widget.update_data(data)

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(100)

    sys.exit(app.exec())
