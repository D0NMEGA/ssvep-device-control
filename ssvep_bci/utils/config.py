"""
SSVEP BCI Configuration

Contains all configurable parameters for the SSVEP BCI system:
- Sampling and window parameters
- SSVEP target frequencies (matched to Arduino LED flicker rates)
- Filter settings
- Decision thresholds
- Channel configuration
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SSVEPConfig:
    """Configuration for the SSVEP BCI system."""

    # ==========================================================================
    # Sampling Parameters
    # ==========================================================================
    fs: int = 250  # Sampling frequency (Hz) - OpenBCI Cyton

    # Window parameters
    # 250 ms window at 250 Hz = 62.5 samples, use 63 for integer
    window_samples: int = 63  # ~252 ms window
    step_samples: int = 12    # ~48 ms step (~80% overlap)

    # ==========================================================================
    # SSVEP Target Frequencies (Hz) - Must match Arduino LED flicker rates
    # ==========================================================================
    # Arduino pins: D2=8.57Hz, D3=10Hz, D4=12Hz, D5=15Hz
    target_frequencies: Tuple[float, ...] = (8.57, 10.0, 12.0, 15.0)

    # Number of harmonics to include in reference signals (1 = fundamental only)
    n_harmonics: int = 2  # fundamental + 1st harmonic

    # ==========================================================================
    # Filter Parameters
    # ==========================================================================
    # Butterworth bandpass filter
    filter_order: int = 5
    bandpass_low: float = 6.0   # Hz - below lowest SSVEP frequency
    bandpass_high: float = 40.0  # Hz - above 2nd harmonic of 15 Hz

    # Optional notch filter (set to None to disable)
    notch_freq: float = None  # 60.0 for US, 50.0 for EU, None to disable
    notch_q: float = 30.0     # Quality factor for notch filter

    # ==========================================================================
    # Decision Thresholds
    # ==========================================================================
    confidence_threshold: float = 0.55  # Minimum correlation for valid detection
    margin_threshold: float = 0.15      # Minimum margin between top-2 correlations

    # Voting/agreement parameters
    agreement_window: int = 2  # Number of consecutive windows that must agree

    # ==========================================================================
    # Channel Configuration
    # ==========================================================================
    # OpenBCI Cyton has 8 channels (0-7)
    # For SSVEP, use occipital/parietal channels
    # Default: use all 8 channels, adjust based on your montage
    # Typical SSVEP montage: O1, O2, Oz, POz, PO3, PO4, PO7, PO8
    eeg_channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])

    # Total channels on Cyton (including aux channels)
    n_total_channels: int = 8

    # ==========================================================================
    # BrainFlow Configuration
    # ==========================================================================
    # Board ID for OpenBCI Cyton
    board_id: int = 0  # Cyton board

    # Serial port (set to None to auto-detect, or specify like "COM3" or "/dev/ttyUSB0")
    serial_port: str = None

    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_directory: str = "logs"
    log_prefix: str = "ssvep_session"

    # ==========================================================================
    # Performance Targets
    # ==========================================================================
    max_processing_latency_ms: float = 20.0  # Max time for window processing
    target_accuracy: float = 0.90  # 90% classification accuracy target

    # ==========================================================================
    # Derived Properties
    # ==========================================================================
    @property
    def window_duration_ms(self) -> float:
        """Window duration in milliseconds."""
        return (self.window_samples / self.fs) * 1000

    @property
    def step_duration_ms(self) -> float:
        """Step duration in milliseconds."""
        return (self.step_samples / self.fs) * 1000

    @property
    def n_eeg_channels(self) -> int:
        """Number of EEG channels used for SSVEP."""
        return len(self.eeg_channels)

    @property
    def time_vector(self) -> np.ndarray:
        """Time vector for one window (in seconds)."""
        return np.arange(self.window_samples) / self.fs

    def get_frequency_label(self, freq_idx: int) -> str:
        """Get human-readable label for frequency index."""
        if 0 <= freq_idx < len(self.target_frequencies):
            return f"{self.target_frequencies[freq_idx]:.2f} Hz"
        return "Unknown"

    def get_led_pin(self, freq_idx: int) -> int:
        """Get Arduino pin number for frequency index."""
        # Arduino whitePins[] = {2, 3, 4, 5} for indices 0-3
        if 0 <= freq_idx < 4:
            return freq_idx + 2
        return -1


# Global default configuration instance
DEFAULT_CONFIG = SSVEPConfig()


def create_config(**kwargs) -> SSVEPConfig:
    """Create a configuration with custom parameters.

    Args:
        **kwargs: Any SSVEPConfig parameter to override

    Returns:
        SSVEPConfig instance with specified overrides
    """
    return SSVEPConfig(**kwargs)
