"""
Filter Bank Analysis for SSVEP-BCI

Implements multiple Chebyshev Type I bandpass filters for improved
SSVEP frequency discrimination. Based on Nakanishi et al. (2018).

Reference:
    Nakanishi, M., et al. (2018). TRCA for SSVEP-BCI.
    IEEE Trans. Biomed. Eng., 65(1), 104-112.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List


class FilterBank:
    """Multi-band Chebyshev filter bank for SSVEP analysis.

    Uses 10 sub-bands with progressively higher lower cutoffs:
    - Band 0: 6-90 Hz
    - Band 1: 14-90 Hz
    - Band 2: 22-90 Hz
    - ...
    - Band 9: 78-90 Hz

    Each band is weighted by coefficients: [1.25, 0.776, 0.622, ...]
    to emphasize lower frequency bands where SSVEP response is stronger.
    """

    def __init__(self, fs: int = 250, num_bands: int = 5):
        """Initialize filter bank.

        Args:
            fs: Sampling frequency in Hz
            num_bands: Number of filter bands (1-10, default 5)
        """
        self.fs = fs
        self.num_bands = min(max(num_bands, 1), 10)  # Clamp to 1-10

        # Filter parameters from reference
        self.passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        self.stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        self.gpass = 3      # Max passband ripple (dB)
        self.gstop = 40     # Min stopband attenuation (dB)
        self.Rp = 0.5       # Passband ripple (dB)

        # Pre-compute filter bank coefficients
        self.fb_coefs = self._compute_coefficients()

        # Pre-design all filters
        self.filters = self._design_filters()

    def _compute_coefficients(self) -> np.ndarray:
        """Compute filter bank weighting coefficients.

        Lower frequency bands get higher weights:
        fb_coefs[i] = i^(-1.25) + 0.25

        Returns:
            Array of shape (num_bands,) with coefficients
        """
        indices = np.arange(1, self.num_bands + 1, dtype=float)
        return np.power(indices, -1.25) + 0.25

    def _design_filters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Design Chebyshev Type I bandpass filters for all bands.

        Returns:
            List of (sos_bp, sos_notch) tuples for each band
        """
        nyq = self.fs / 2.0
        filters = []

        for fb_i in range(self.num_bands):
            # Bandpass filter design
            Wp = [self.passband[fb_i] / nyq, 90.0 / nyq]
            Ws = [self.stopband[fb_i] / nyq, 100.0 / nyq]

            # Ensure valid frequency range
            Wp = [max(0.001, min(w, 0.999)) for w in Wp]
            Ws = [max(0.001, min(w, 0.999)) for w in Ws]

            # Design Chebyshev Type I filter
            N, Wn = signal.cheb1ord(Wp, Ws, self.gpass, self.gstop)
            sos_bp = signal.cheby1(N, self.Rp, Wn, btype='band', output='sos')

            # 60 Hz notch filter (same for all bands)
            b, a = signal.iirnotch(60.0 / nyq, Q=30.0)
            sos_notch = signal.tf2sos(b, a)

            filters.append((sos_bp, sos_notch))

        return filters

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        """Apply filter bank using zero-phase filtering (for training).

        Args:
            data: Input EEG data
                - 2D: (n_channels, n_samples) → single trial
                - 3D: (n_trials, n_channels, n_samples) → multiple trials

        Returns:
            Filtered data with filter bank dimension added:
                - 2D input: (n_channels, n_samples, n_filterbanks)
                - 3D input: (n_trials, n_channels, n_samples, n_filterbanks)
        """
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            return self._apply_offline_single(data)

        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            n_trials = data.shape[0]
            filtered_trials = []

            for trial_idx in range(n_trials):
                filtered = self._apply_offline_single(data[trial_idx])
                filtered_trials.append(filtered)

            # Stack to (n_trials, n_channels, n_samples, n_filterbanks)
            return np.stack(filtered_trials, axis=0)

        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    def _apply_offline_single(self, trial: np.ndarray) -> np.ndarray:
        """Apply filter bank to single trial using filtfilt.

        Args:
            trial: (n_channels, n_samples)

        Returns:
            (n_channels, n_samples, n_filterbanks)
        """
        n_channels, n_samples = trial.shape
        filtered = np.zeros((n_channels, n_samples, self.num_bands))

        for fb_i in range(self.num_bands):
            sos_bp, sos_notch = self.filters[fb_i]

            for ch in range(n_channels):
                # Bandpass
                filtered[ch, :, fb_i] = signal.sosfiltfilt(sos_bp, trial[ch])
                # Notch
                filtered[ch, :, fb_i] = signal.sosfiltfilt(sos_notch, filtered[ch, :, fb_i])

        return filtered

    def apply_online(self, data: np.ndarray, filter_states: List) -> Tuple[np.ndarray, List]:
        """Apply filter bank using causal filtering (for real-time).

        Args:
            data: (n_channels, n_samples) - incoming EEG chunk
            filter_states: List of filter states from previous calls
                           (empty list on first call)

        Returns:
            filtered: (n_channels, n_samples, n_filterbanks)
            filter_states: Updated filter states for next call
        """
        n_channels, n_samples = data.shape

        # Initialize filter states if needed
        if not filter_states:
            filter_states = self._init_online_states(n_channels)

        filtered = np.zeros((n_channels, n_samples, self.num_bands))

        for fb_i in range(self.num_bands):
            sos_bp, sos_notch = self.filters[fb_i]
            zi_bp_list = filter_states[fb_i]['bp']
            zi_notch_list = filter_states[fb_i]['notch']

            for ch in range(n_channels):
                # Bandpass
                filtered[ch, :, fb_i], zi_bp_list[ch] = signal.sosfilt(
                    sos_bp, data[ch], zi=zi_bp_list[ch]
                )
                # Notch
                filtered[ch, :, fb_i], zi_notch_list[ch] = signal.sosfilt(
                    sos_notch, filtered[ch, :, fb_i], zi=zi_notch_list[ch]
                )

        return filtered, filter_states

    def _init_online_states(self, n_channels: int) -> List:
        """Initialize filter states for online filtering.

        Args:
            n_channels: Number of EEG channels

        Returns:
            List of filter states for each band and channel
        """
        states = []

        for fb_i in range(self.num_bands):
            sos_bp, sos_notch = self.filters[fb_i]

            # Create initial conditions
            zi_bp = [signal.sosfilt_zi(sos_bp) for _ in range(n_channels)]
            zi_notch = [signal.sosfilt_zi(sos_notch) for _ in range(n_channels)]

            states.append({'bp': zi_bp, 'notch': zi_notch})

        return states

    def reset_online_states(self, filter_states: List, data_init: np.ndarray) -> List:
        """Reset filter states based on initial data values.

        Args:
            filter_states: Current filter states
            data_init: (n_channels, 1) - initial data values for scaling

        Returns:
            Reset filter states
        """
        n_channels = data_init.shape[0]

        for fb_i in range(self.num_bands):
            sos_bp, sos_notch = self.filters[fb_i]

            for ch in range(n_channels):
                # Scale initial conditions by first data value
                filter_states[fb_i]['bp'][ch] = (
                    signal.sosfilt_zi(sos_bp) * data_init[ch, 0]
                )
                filter_states[fb_i]['notch'][ch] = (
                    signal.sosfilt_zi(sos_notch) * data_init[ch, 0]
                )

        return filter_states

    def get_coefficients(self) -> np.ndarray:
        """Get filter bank weighting coefficients.

        Returns:
            Array of shape (num_bands,)
        """
        return self.fb_coefs.copy()

    def get_num_bands(self) -> int:
        """Get number of filter banks.

        Returns:
            Number of bands
        """
        return self.num_bands
