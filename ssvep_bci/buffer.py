"""
EEG Ring Buffer for SSVEP BCI

Maintains a sliding window of EEG data with configurable window size and step size.
Thread-safe for use with streaming acquisition.
"""

import numpy as np
from threading import Lock
from typing import Optional, Tuple

from .config import SSVEPConfig


class EEGBuffer:
    """Ring buffer for EEG data with sliding window support.

    Maintains a rolling window of EEG samples and provides windowed access
    for real-time SSVEP processing. Thread-safe for concurrent append/read.

    Attributes:
        config: SSVEPConfig instance with buffer parameters
        n_channels: Number of EEG channels
        window_samples: Number of samples per window (e.g., 63 for 252ms at 250Hz)
        step_samples: Number of samples to advance between windows (e.g., 12 for 48ms)
    """

    def __init__(self, config: SSVEPConfig = None):
        """Initialize the EEG buffer.

        Args:
            config: SSVEPConfig instance. Uses default if None.
        """
        self.config = config or SSVEPConfig()
        self.n_channels = self.config.n_eeg_channels
        self.window_samples = self.config.window_samples
        self.step_samples = self.config.step_samples

        # Internal buffer - stores more than one window for smooth sliding
        # Keep 2x window size to handle overlapping windows
        self._buffer_size = self.window_samples * 2
        self._buffer = np.zeros((self.n_channels, self._buffer_size), dtype=np.float64)

        # Write position (circular)
        self._write_pos = 0

        # Count of total samples written (for tracking readiness)
        self._total_samples = 0

        # Samples since last window extraction
        self._samples_since_last_window = 0

        # Thread safety
        self._lock = Lock()

        # Track if we've filled at least one full window
        self._window_ready = False

    def append(self, samples: np.ndarray) -> None:
        """Append new EEG samples to the buffer.

        Args:
            samples: New samples with shape (n_channels, n_samples) or (n_samples,)
                     for single-channel. Samples should be in microvolts.
        """
        with self._lock:
            # Handle 1D input (single sample across channels)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)

            # Validate channel count
            if samples.shape[0] != self.n_channels:
                raise ValueError(
                    f"Expected {self.n_channels} channels, got {samples.shape[0]}"
                )

            n_new_samples = samples.shape[1]

            # Write samples to circular buffer
            for i in range(n_new_samples):
                self._buffer[:, self._write_pos] = samples[:, i]
                self._write_pos = (self._write_pos + 1) % self._buffer_size

            self._total_samples += n_new_samples
            self._samples_since_last_window += n_new_samples

            # Check if we have enough samples for a full window
            if self._total_samples >= self.window_samples:
                self._window_ready = True

    def ready(self) -> bool:
        """Check if a new window is ready for processing.

        Returns True when:
        1. At least one full window of data has been collected
        2. At least step_samples new samples have arrived since last extraction
        """
        with self._lock:
            return (
                self._window_ready and
                self._samples_since_last_window >= self.step_samples
            )

    def get_window(self) -> Optional[np.ndarray]:
        """Extract the current window of EEG data.

        Returns:
            Window array with shape (n_channels, window_samples), or None if
            no window is ready. Data is returned in chronological order
            (oldest sample first).
        """
        with self._lock:
            if not self._window_ready:
                return None

            # Calculate start position for the window
            # We want the most recent window_samples samples
            end_pos = self._write_pos
            start_pos = (end_pos - self.window_samples) % self._buffer_size

            # Extract window (handling circular wrap-around)
            if start_pos < end_pos:
                window = self._buffer[:, start_pos:end_pos].copy()
            else:
                # Wrap-around case
                window = np.hstack([
                    self._buffer[:, start_pos:],
                    self._buffer[:, :end_pos]
                ])

            # Reset samples counter (we've "consumed" this window)
            self._samples_since_last_window = 0

            return window

    def get_window_no_consume(self) -> Optional[np.ndarray]:
        """Get current window without marking it as consumed.

        Useful for visualization when you want to peek at the data
        without affecting the processing pipeline.

        Returns:
            Window array with shape (n_channels, window_samples), or None.
        """
        with self._lock:
            if not self._window_ready:
                return None

            end_pos = self._write_pos
            start_pos = (end_pos - self.window_samples) % self._buffer_size

            if start_pos < end_pos:
                window = self._buffer[:, start_pos:end_pos].copy()
            else:
                window = np.hstack([
                    self._buffer[:, start_pos:],
                    self._buffer[:, :end_pos]
                ])

            return window

    def get_extended_window(self, n_samples: int) -> Optional[np.ndarray]:
        """Get an extended window for visualization (e.g., 5 seconds).

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            Array with shape (n_channels, n_samples), or None if not enough data.
        """
        with self._lock:
            if self._total_samples < n_samples:
                return None

            # Calculate how much data we actually have
            available = min(self._total_samples, self._buffer_size)
            n_samples = min(n_samples, available)

            end_pos = self._write_pos
            start_pos = (end_pos - n_samples) % self._buffer_size

            if start_pos < end_pos:
                window = self._buffer[:, start_pos:end_pos].copy()
            else:
                window = np.hstack([
                    self._buffer[:, start_pos:],
                    self._buffer[:, :end_pos]
                ])

            return window

    def reset(self) -> None:
        """Reset the buffer to initial state."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_samples = 0
            self._samples_since_last_window = 0
            self._window_ready = False

    @property
    def total_samples(self) -> int:
        """Total number of samples written to buffer."""
        with self._lock:
            return self._total_samples

    @property
    def samples_pending(self) -> int:
        """Number of samples since last window extraction."""
        with self._lock:
            return self._samples_since_last_window


class ExtendedEEGBuffer(EEGBuffer):
    """Extended buffer that also keeps a longer history for visualization.

    Maintains both the short processing window and a longer rolling buffer
    for plotting 5+ seconds of EEG data.
    """

    def __init__(self, config: SSVEPConfig = None, history_seconds: float = 5.0):
        """Initialize extended buffer.

        Args:
            config: SSVEPConfig instance
            history_seconds: Seconds of history to keep for visualization
        """
        super().__init__(config)

        # Extended history buffer
        self._history_samples = int(history_seconds * self.config.fs)
        self._history_buffer = np.zeros(
            (self.n_channels, self._history_samples), dtype=np.float64
        )
        self._history_write_pos = 0
        self._history_filled = False

    def append(self, samples: np.ndarray) -> None:
        """Append samples to both processing and history buffers."""
        # First, call parent to update processing buffer
        super().append(samples)

        # Handle 1D input
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        # Also append to history buffer
        with self._lock:
            n_new = samples.shape[1]
            for i in range(n_new):
                self._history_buffer[:, self._history_write_pos] = samples[:, i]
                self._history_write_pos = (
                    (self._history_write_pos + 1) % self._history_samples
                )
                if self._history_write_pos == 0:
                    self._history_filled = True

    def get_history(self) -> Tuple[np.ndarray, float]:
        """Get the full history buffer for visualization.

        Returns:
            Tuple of (data array, duration in seconds).
            Data shape is (n_channels, n_samples), chronological order.
        """
        with self._lock:
            if self._history_filled:
                # Full buffer, need to unwrap
                data = np.hstack([
                    self._history_buffer[:, self._history_write_pos:],
                    self._history_buffer[:, :self._history_write_pos]
                ])
                duration = self._history_samples / self.config.fs
            else:
                # Not yet filled, just return what we have
                data = self._history_buffer[:, :self._history_write_pos].copy()
                duration = self._history_write_pos / self.config.fs

            return data, duration

    def reset(self) -> None:
        """Reset both processing and history buffers."""
        super().reset()
        with self._lock:
            self._history_buffer.fill(0)
            self._history_write_pos = 0
            self._history_filled = False


# Unit test
if __name__ == "__main__":
    print("Testing EEGBuffer...")

    config = SSVEPConfig()
    buffer = EEGBuffer(config)

    print(f"Window samples: {buffer.window_samples}")
    print(f"Step samples: {buffer.step_samples}")
    print(f"Channels: {buffer.n_channels}")

    # Simulate streaming data
    fs = 250
    duration = 1.0  # 1 second of data
    n_samples = int(fs * duration)

    # Generate synthetic 8-channel EEG with SSVEP-like signals
    t = np.arange(n_samples) / fs
    test_data = np.zeros((8, n_samples))
    for ch in range(8):
        # Add 10 Hz SSVEP component + noise
        test_data[ch] = 10 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 2

    # Stream in chunks of 12 samples (like real-time)
    chunk_size = 12
    windows_extracted = 0

    for i in range(0, n_samples, chunk_size):
        chunk = test_data[:, i:i+chunk_size]
        buffer.append(chunk)

        while buffer.ready():
            window = buffer.get_window()
            if window is not None:
                windows_extracted += 1
                print(f"Window {windows_extracted}: shape={window.shape}, "
                      f"mean={window.mean():.2f}, std={window.std():.2f}")

    print(f"\nTotal windows extracted: {windows_extracted}")
    print(f"Total samples: {buffer.total_samples}")
    print("EEGBuffer test passed!")
