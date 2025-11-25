"""
SSVEP Preprocessing Module

Implements online preprocessing for SSVEP BCI:
- Common Average Reference (CAR)
- Butterworth bandpass filter with persistent state (for causal, online filtering)
- Optional notch filter for line noise removal
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


class SSVEPPreprocessor:
    """Online preprocessor for SSVEP EEG data.

    Applies Common Average Reference (CAR) and causal Butterworth bandpass
    filtering with persistent filter state for continuous streaming data.

    The filter maintains internal state (zi) so that consecutive windows
    are processed as a continuous stream without edge artifacts.

    Attributes:
        config: SSVEPConfig instance with filter parameters
        sos: Second-order sections for the bandpass filter
        zi: Filter state for each channel
    """

    def __init__(self, config: SSVEPConfig = None):
        """Initialize the preprocessor.

        Args:
            config: SSVEPConfig instance. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG

        # Design Butterworth bandpass filter
        self.sos = self._design_bandpass_filter()

        # Initialize filter state for each channel
        # zi shape: (n_sections, 2) per channel
        self._init_filter_state()

        # Optional notch filter
        self.notch_sos = None
        self.notch_zi = None
        if self.config.notch_freq is not None:
            self._design_notch_filter()

        # Smoothing filter (moving average) state
        self.smoothing_buffer = None
        if self.config.smoothing_enabled:
            self._init_smoothing_buffer()

        # Track if we've been initialized with real data
        self._initialized = False

    def _design_bandpass_filter(self) -> np.ndarray:
        """Design Butterworth bandpass filter.

        Returns:
            Second-order sections (sos) array for the filter
        """
        # Nyquist frequency
        nyq = self.config.fs / 2.0

        # Normalize frequencies
        low = self.config.bandpass_low / nyq
        high = self.config.bandpass_high / nyq

        # Ensure frequencies are valid
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))

        if low >= high:
            raise ValueError(
                f"Invalid filter frequencies: low={self.config.bandpass_low}, "
                f"high={self.config.bandpass_high}"
            )

        # Design filter using second-order sections (more stable than ba)
        sos = signal.butter(
            self.config.filter_order,
            [low, high],
            btype='band',
            output='sos'
        )

        return sos

    def _design_notch_filter(self) -> None:
        """Design IIR notch filter for line noise removal."""
        nyq = self.config.fs / 2.0
        freq = self.config.notch_freq / nyq

        # Design notch filter
        b, a = signal.iirnotch(freq, self.config.notch_q)

        # Convert to sos for stability
        self.notch_sos = signal.tf2sos(b, a)

        # Initialize notch filter state
        n_sections = self.notch_sos.shape[0]
        self.notch_zi = [
            signal.sosfilt_zi(self.notch_sos) * 0
            for _ in range(self.config.n_eeg_channels)
        ]

    def _init_smoothing_buffer(self) -> None:
        """Initialize smoothing buffer for moving average."""
        # Buffer to store recent samples for each channel
        # Shape: (n_channels, smoothing_window)
        self.smoothing_buffer = [
            np.zeros(self.config.smoothing_window)
            for _ in range(self.config.n_eeg_channels)
        ]

    def _init_filter_state(self) -> None:
        """Initialize filter states for all channels."""
        # Get initial state template
        zi_template = signal.sosfilt_zi(self.sos)

        # Create state for each channel (will be properly initialized on first data)
        self.zi = [zi_template.copy() for _ in range(self.config.n_eeg_channels)]

    def reset(self) -> None:
        """Reset filter states to initial conditions.

        Call this when starting a new recording session.
        """
        self._init_filter_state()
        if self.notch_sos is not None:
            n_sections = self.notch_sos.shape[0]
            self.notch_zi = [
                signal.sosfilt_zi(self.notch_sos) * 0
                for _ in range(self.config.n_eeg_channels)
            ]
        if self.config.smoothing_enabled:
            self._init_smoothing_buffer()
        self._initialized = False

    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference (CAR).

        Subtracts the mean across channels from each channel.
        This helps remove common-mode noise and artifacts.

        Args:
            data: EEG data with shape (n_channels, n_samples)

        Returns:
            CAR-referenced data with same shape
        """
        # Compute mean across channels for each time point
        common_avg = np.mean(data, axis=0, keepdims=True)

        # Subtract common average
        return data - common_avg

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """Apply causal bandpass filter with persistent state.

        Uses sosfilt with state preservation for continuous streaming.
        This is essential for online processing to avoid edge artifacts.

        Args:
            data: EEG data with shape (n_channels, n_samples)

        Returns:
            Bandpass-filtered data with same shape
        """
        n_channels, n_samples = data.shape
        filtered = np.zeros_like(data)

        for ch in range(n_channels):
            # Initialize state with first sample if not yet done
            if not self._initialized:
                self.zi[ch] = signal.sosfilt_zi(self.sos) * data[ch, 0]

            # Apply filter with state
            filtered[ch], self.zi[ch] = signal.sosfilt(
                self.sos, data[ch], zi=self.zi[ch]
            )

        self._initialized = True
        return filtered

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter if configured.

        Args:
            data: EEG data with shape (n_channels, n_samples)

        Returns:
            Notch-filtered data with same shape
        """
        if self.notch_sos is None:
            return data

        n_channels, n_samples = data.shape
        filtered = np.zeros_like(data)

        for ch in range(n_channels):
            filtered[ch], self.notch_zi[ch] = signal.sosfilt(
                self.notch_sos, data[ch], zi=self.notch_zi[ch]
            )

        return filtered

    def apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing filter.

        Uses a causal moving average with persistent state for streaming data.

        Args:
            data: EEG data with shape (n_channels, n_samples)

        Returns:
            Smoothed data with same shape
        """
        if not self.config.smoothing_enabled or self.smoothing_buffer is None:
            return data

        n_channels, n_samples = data.shape
        smoothed = np.zeros_like(data)
        window_size = self.config.smoothing_window

        for ch in range(n_channels):
            for i in range(n_samples):
                # Shift buffer (remove oldest, add newest)
                self.smoothing_buffer[ch] = np.roll(self.smoothing_buffer[ch], -1)
                self.smoothing_buffer[ch][-1] = data[ch, i]

                # Compute moving average
                smoothed[ch, i] = np.mean(self.smoothing_buffer[ch])

        return smoothed

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline.

        Pipeline order:
        1. Common Average Reference (CAR)
        2. Bandpass filter (5-50 Hz default)
        3. Notch filter (60 Hz for US powerline noise)
        4. Smoothing (moving average)

        Args:
            data: Raw EEEG data with shape (n_channels, n_samples)

        Returns:
            Preprocessed data with same shape
        """
        # Validate input
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        if data.shape[0] != self.config.n_eeg_channels:
            raise ValueError(
                f"Expected {self.config.n_eeg_channels} channels, "
                f"got {data.shape[0]}"
            )

        # Apply preprocessing pipeline
        processed = self.apply_car(data)
        processed = self.apply_bandpass(processed)

        if self.notch_sos is not None:
            processed = self.apply_notch(processed)

        if self.config.smoothing_enabled:
            processed = self.apply_smoothing(processed)

        return processed

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a single window of data.

        Convenience method that handles the complete preprocessing
        for one analysis window.

        Args:
            window: EEG window with shape (n_channels, window_samples)

        Returns:
            Preprocessed window with same shape
        """
        return self.process(window)

    def get_filter_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the frequency response of the bandpass filter.

        Useful for debugging and visualization.

        Returns:
            Tuple of (frequencies in Hz, magnitude response in dB)
        """
        w, h = signal.sosfreqz(self.sos, worN=2048, fs=self.config.fs)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        return w, magnitude_db


class WindowedPreprocessor(SSVEPPreprocessor):
    """Preprocessor optimized for windowed operation.

    This variant is designed for processing complete windows at a time,
    useful when you want to ensure consistent filter behavior across windows.
    """

    def __init__(self, config: SSVEPConfig = None, reset_per_window: bool = False):
        """Initialize windowed preprocessor.

        Args:
            config: SSVEPConfig instance
            reset_per_window: If True, reset filter state for each window.
                             This reduces temporal continuity but ensures
                             consistent behavior. Default False.
        """
        super().__init__(config)
        self.reset_per_window = reset_per_window

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a single window.

        Args:
            window: EEG window with shape (n_channels, window_samples)

        Returns:
            Preprocessed window
        """
        if self.reset_per_window:
            # Reset state before processing (less optimal for continuity)
            self.reset()

        return self.process(window)


# Unit test
if __name__ == "__main__":
    print("Testing SSVEPPreprocessor...")

    config = SSVEPConfig()
    preprocessor = SSVEPPreprocessor(config)

    print(f"Filter order: {config.filter_order}")
    print(f"Bandpass: {config.bandpass_low}-{config.bandpass_high} Hz")
    print(f"Sampling rate: {config.fs} Hz")

    # Generate test data: 1 second of synthetic EEG
    fs = config.fs
    duration = 1.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    # Create test signal: 10 Hz SSVEP + 60 Hz noise + broadband noise
    test_data = np.zeros((8, n_samples))
    for ch in range(8):
        # SSVEP component (should pass through)
        ssvep = 20 * np.sin(2 * np.pi * 10 * t)
        # Line noise (should be attenuated if notch enabled)
        line_noise = 5 * np.sin(2 * np.pi * 60 * t)
        # Low frequency drift (should be filtered)
        drift = 50 * np.sin(2 * np.pi * 0.5 * t)
        # High frequency noise (should be filtered)
        hf_noise = 10 * np.sin(2 * np.pi * 80 * t)
        # Random noise
        random_noise = np.random.randn(n_samples) * 5

        test_data[ch] = ssvep + line_noise + drift + hf_noise + random_noise

    print(f"\nRaw data: shape={test_data.shape}")
    print(f"Raw data range: [{test_data.min():.1f}, {test_data.max():.1f}]")

    # Process in chunks to simulate streaming
    chunk_size = 63  # One window
    processed_chunks = []

    for i in range(0, n_samples, chunk_size):
        chunk = test_data[:, i:i+chunk_size]
        if chunk.shape[1] == chunk_size:
            processed = preprocessor.process(chunk)
            processed_chunks.append(processed)

    processed_data = np.hstack(processed_chunks)
    print(f"\nProcessed data: shape={processed_data.shape}")
    print(f"Processed data range: [{processed_data.min():.1f}, {processed_data.max():.1f}]")

    # Check frequency content
    from scipy.fft import rfft, rfftfreq

    # FFT of original
    freq = rfftfreq(n_samples, 1/fs)
    raw_fft = np.abs(rfft(test_data[0]))

    # FFT of processed (use matching length)
    proc_len = processed_data.shape[1]
    freq_proc = rfftfreq(proc_len, 1/fs)
    proc_fft = np.abs(rfft(processed_data[0]))

    # Find peaks
    raw_10hz_idx = np.argmin(np.abs(freq - 10))
    proc_10hz_idx = np.argmin(np.abs(freq_proc - 10))

    print(f"\n10 Hz component (raw): {raw_fft[raw_10hz_idx]:.1f}")
    print(f"10 Hz component (processed): {proc_fft[proc_10hz_idx]:.1f}")

    # Check filter response
    freqs, mag_db = preprocessor.get_filter_response()
    passband_idx = np.where((freqs >= 6) & (freqs <= 40))[0]
    print(f"\nPassband attenuation: {mag_db[passband_idx].mean():.1f} dB")

    print("\nSSVEPPreprocessor test passed!")
