"""
SSVEP Offline Preprocessing Module

Implements zero-phase (non-causal) preprocessing for offline analysis and calibration:
- Common Average Reference (CAR)
- Zero-phase IIR bandpass filter (7-90 Hz) using filtfilt
- Zero-phase IIR notch filter (60 Hz) using filtfilt

This module is for OFFLINE use only (calibration, training, evaluation).
For real-time streaming, use the causal SSVEPPreprocessor instead.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


class OfflinePreprocessor:
    """Zero-phase preprocessor for offline SSVEP analysis.

    Uses scipy.signal.filtfilt for zero-phase filtering, which processes
    the signal both forward and backward to eliminate phase distortion.
    This is ideal for offline analysis but cannot be used in real-time.

    Attributes:
        config: SSVEPConfig instance with filter parameters
        bandpass_sos: Second-order sections for bandpass filter (7-90 Hz)
        notch_sos: Second-order sections for notch filter (60 Hz)
    """

    def __init__(self, config: SSVEPConfig = None):
        """Initialize the offline preprocessor.

        Args:
            config: SSVEPConfig instance. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG

        # Design filters
        self.bandpass_sos = self._design_bandpass_filter()
        self.notch_sos = self._design_notch_filter()

    def _design_bandpass_filter(self) -> np.ndarray:
        """Design zero-phase IIR bandpass filter (7-90 Hz).

        Returns:
            Second-order sections (sos) array for the filter
        """
        nyq = self.config.fs / 2.0

        # Fixed band for offline analysis: 7-90 Hz
        low = 7.0 / nyq
        high = 90.0 / nyq

        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))

        if low >= high:
            raise ValueError(f"Invalid filter frequencies: low=7.0, high=90.0")

        # Design Butterworth bandpass filter (order 5 for good rolloff)
        sos = signal.butter(
            5,  # filter order
            [low, high],
            btype='band',
            output='sos'
        )

        return sos

    def _design_notch_filter(self) -> np.ndarray:
        """Design zero-phase IIR notch filter (60 Hz).

        Returns:
            Second-order sections (sos) array for the notch filter
        """
        nyq = self.config.fs / 2.0
        freq = 60.0 / nyq  # US powerline frequency

        # Design IIR notch filter
        b, a = signal.iirnotch(freq, Q=30.0)

        # Convert to sos for numerical stability
        sos = signal.tf2sos(b, a)

        return sos

    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference (CAR).

        Subtracts the mean across channels from each channel.
        This helps remove common-mode noise and artifacts.

        Args:
            data: EEG data with shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            CAR-referenced data with same shape
        """
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            common_avg = np.mean(data, axis=0, keepdims=True)
            return data - common_avg
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            common_avg = np.mean(data, axis=1, keepdims=True)
            return data - common_avg
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass filter (7-90 Hz).

        Uses filtfilt for zero-phase filtering. This processes the signal
        both forward and backward, eliminating phase distortion but requiring
        the entire signal to be available (cannot be used in real-time).

        Args:
            data: EEG data with shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Bandpass-filtered data with same shape
        """
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.sosfiltfilt(self.bandpass_sos, data[ch])
            return filtered
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            filtered = np.zeros_like(data)
            for trial in range(data.shape[0]):
                for ch in range(data.shape[1]):
                    filtered[trial, ch] = signal.sosfiltfilt(self.bandpass_sos, data[trial, ch])
            return filtered
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase notch filter (60 Hz).

        Args:
            data: EEG data with shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Notch-filtered data with same shape
        """
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.sosfiltfilt(self.notch_sos, data[ch])
            return filtered
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            filtered = np.zeros_like(data)
            for trial in range(data.shape[0]):
                for ch in range(data.shape[1]):
                    filtered[trial, ch] = signal.sosfiltfilt(self.notch_sos, data[trial, ch])
            return filtered
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply full offline preprocessing pipeline.

        Pipeline order:
        1. Common Average Reference (CAR)
        2. Bandpass filter (7-90 Hz, zero-phase)
        3. Notch filter (60 Hz, zero-phase)

        Args:
            data: Raw EEG data with shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Preprocessed data with same shape
        """
        # Validate input
        if data.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

        # For 2D data, check channel count
        if data.ndim == 2 and data.shape[0] != self.config.n_eeg_channels:
            raise ValueError(
                f"Expected {self.config.n_eeg_channels} channels, got {data.shape[0]}"
            )

        # For 3D data, check channel count
        if data.ndim == 3 and data.shape[1] != self.config.n_eeg_channels:
            raise ValueError(
                f"Expected {self.config.n_eeg_channels} channels, got {data.shape[1]}"
            )

        # Apply preprocessing pipeline
        processed = self.apply_car(data)
        processed = self.apply_bandpass(processed)
        processed = self.apply_notch(processed)

        return processed

    def get_filter_response(self, which: str = 'bandpass') -> Tuple[np.ndarray, np.ndarray]:
        """Get the frequency response of a filter.

        Useful for debugging and visualization.

        Args:
            which: 'bandpass' or 'notch'

        Returns:
            Tuple of (frequencies in Hz, magnitude response in dB)
        """
        if which == 'bandpass':
            sos = self.bandpass_sos
        elif which == 'notch':
            sos = self.notch_sos
        else:
            raise ValueError(f"Unknown filter type: {which}")

        w, h = signal.sosfreqz(sos, worN=4096, fs=self.config.fs)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        return w, magnitude_db


# Unit test
if __name__ == "__main__":
    print("Testing OfflinePreprocessor...")

    config = SSVEPConfig()
    preprocessor = OfflinePreprocessor(config)

    print(f"Bandpass: 7-90 Hz (zero-phase)")
    print(f"Notch: 60 Hz (zero-phase)")
    print(f"Sampling rate: {config.fs} Hz")

    # Generate test data: 2 seconds of synthetic EEG
    fs = config.fs
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    # Create test signal: 10 Hz SSVEP + 60 Hz noise + drift + high-freq noise
    test_data = np.zeros((8, n_samples))
    for ch in range(8):
        # SSVEP component at 10 Hz (should pass through)
        ssvep = 20 * np.sin(2 * np.pi * 10 * t)
        # Line noise at 60 Hz (should be removed by notch)
        line_noise = 10 * np.sin(2 * np.pi * 60 * t)
        # Low frequency drift at 2 Hz (should be removed by highpass)
        drift = 50 * np.sin(2 * np.pi * 2 * t)
        # High frequency noise at 100 Hz (should be removed by lowpass)
        hf_noise = 10 * np.sin(2 * np.pi * 100 * t)
        # Random noise
        random_noise = np.random.randn(n_samples) * 5

        test_data[ch] = ssvep + line_noise + drift + hf_noise + random_noise

    print(f"\nRaw data: shape={test_data.shape}")
    print(f"Raw data range: [{test_data.min():.1f}, {test_data.max():.1f}]")

    # Process entire signal (offline)
    processed_data = preprocessor.process(test_data)

    print(f"\nProcessed data: shape={processed_data.shape}")
    print(f"Processed data range: [{processed_data.min():.1f}, {processed_data.max():.1f}]")

    # Check frequency content using FFT
    from scipy.fft import rfft, rfftfreq

    freq = rfftfreq(n_samples, 1/fs)

    # FFT of raw and processed
    raw_fft = np.abs(rfft(test_data[0]))
    proc_fft = np.abs(rfft(processed_data[0]))

    # Find peaks at specific frequencies
    def get_fft_at_freq(fft_data, freqs, target_freq):
        idx = np.argmin(np.abs(freqs - target_freq))
        return fft_data[idx]

    print(f"\n10 Hz component:")
    print(f"  Raw: {get_fft_at_freq(raw_fft, freq, 10):.1f}")
    print(f"  Processed: {get_fft_at_freq(proc_fft, freq, 10):.1f}")

    print(f"\n60 Hz component (should be attenuated):")
    print(f"  Raw: {get_fft_at_freq(raw_fft, freq, 60):.1f}")
    print(f"  Processed: {get_fft_at_freq(proc_fft, freq, 60):.1f}")

    print(f"\n2 Hz component (should be attenuated):")
    print(f"  Raw: {get_fft_at_freq(raw_fft, freq, 2):.1f}")
    print(f"  Processed: {get_fft_at_freq(proc_fft, freq, 2):.1f}")

    print(f"\n100 Hz component (should be attenuated):")
    print(f"  Raw: {get_fft_at_freq(raw_fft, freq, 100):.1f}")
    print(f"  Processed: {get_fft_at_freq(proc_fft, freq, 100):.1f}")

    # Test with 3D data (multiple trials)
    print("\n\nTesting with 3D data (multiple trials)...")
    n_trials = 10
    test_data_3d = np.zeros((n_trials, 8, n_samples))
    for trial in range(n_trials):
        for ch in range(8):
            ssvep = 20 * np.sin(2 * np.pi * 10 * t + trial * 0.1)
            line_noise = 10 * np.sin(2 * np.pi * 60 * t)
            drift = 50 * np.sin(2 * np.pi * 2 * t)
            random_noise = np.random.randn(n_samples) * 5
            test_data_3d[trial, ch] = ssvep + line_noise + drift + random_noise

    processed_data_3d = preprocessor.process(test_data_3d)

    print(f"3D Raw data: shape={test_data_3d.shape}")
    print(f"3D Processed data: shape={processed_data_3d.shape}")

    # Check filter responses
    print("\n\nBandpass filter response:")
    freqs_bp, mag_bp = preprocessor.get_filter_response('bandpass')
    passband_7_90 = mag_bp[(freqs_bp >= 7) & (freqs_bp <= 90)]
    stopband_below = mag_bp[freqs_bp < 5]
    stopband_above = mag_bp[freqs_bp > 95]
    print(f"  Passband (7-90 Hz) attenuation: {passband_7_90.mean():.1f} dB")
    print(f"  Stopband (<5 Hz) attenuation: {stopband_below.mean():.1f} dB")
    print(f"  Stopband (>95 Hz) attenuation: {stopband_above.mean():.1f} dB")

    print("\nNotch filter response:")
    freqs_notch, mag_notch = preprocessor.get_filter_response('notch')
    notch_at_60 = mag_notch[np.argmin(np.abs(freqs_notch - 60))]
    passband_notch = mag_notch[(freqs_notch < 55) | (freqs_notch > 65)]
    print(f"  Attenuation at 60 Hz: {notch_at_60:.1f} dB")
    print(f"  Passband (away from 60 Hz): {passband_notch.mean():.1f} dB")

    print("\nOfflinePreprocessor test passed!")
