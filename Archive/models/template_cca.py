"""
Template-based CCA Decoder

Extends standard CCA with subject-specific templates from calibration data.
Uses individual templates (IT-CCA) for improved classification accuracy.

Reference:
    Chen, X., et al. (2015). Filter bank canonical correlation analysis for
    implementing a high-speed SSVEP-based brain-computer interface.
    Journal of Neural Engineering.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import SSVEPConfig
from models.calibration import CalibrationData, CalibrationCollector
from models.cca_decoder import SSVEPDecoder, DecisionResult


@dataclass
class TemplateCCAConfig:
    """Configuration for template CCA."""
    # Number of CCA components
    n_components: int = 1

    # Template averaging
    use_averaged_templates: bool = True

    # Combination weights for standard and template CCA
    standard_weight: float = 0.3  # Weight for standard CCA correlation
    template_weight: float = 0.7  # Weight for template correlation

    # Template window (relative to trial start)
    template_start_s: float = 0.5  # Skip first 0.5s (transient)
    template_duration_s: float = 3.0  # Use 3s of data


class TemplateCCADecoder:
    """Template-based CCA decoder using calibration data.

    Combines standard CCA (sine/cosine references) with individual
    templates from calibration for improved classification.
    """

    def __init__(
        self,
        config: SSVEPConfig = None,
        template_config: TemplateCCAConfig = None,
        calibration_data: CalibrationData = None
    ):
        """Initialize template CCA decoder.

        Args:
            config: SSVEP configuration
            template_config: Template CCA configuration
            calibration_data: Calibration data (can be loaded later)
        """
        self.config = config or SSVEPConfig()
        self.template_config = template_config or TemplateCCAConfig()

        # Standard CCA decoder (fallback)
        self.standard_decoder = SSVEPDecoder(self.config)

        # Templates: {frequency: averaged_template}
        self.templates: Dict[float, np.ndarray] = {}

        # Spatial filters learned from templates
        self.spatial_filters: Dict[float, np.ndarray] = {}

        # CCA model for template matching
        self.cca = CCA(n_components=self.template_config.n_components)

        # Load calibration if provided
        if calibration_data is not None:
            self.load_calibration(calibration_data)

        # Voting state
        self._recent_predictions: List[Optional[float]] = []

    def load_calibration(self, data: CalibrationData) -> None:
        """Load and process calibration data.

        Args:
            data: CalibrationData object
        """
        # Calculate template parameters
        start_sample = int(self.template_config.template_start_s * data.fs)
        n_samples = int(self.template_config.template_duration_s * data.fs)

        # Process each frequency
        for freq in self.config.target_frequencies:
            # Find trials for this frequency
            freq_mask = np.isclose(data.labels, freq, atol=0.1)
            freq_epochs = data.epochs[freq_mask]

            if len(freq_epochs) == 0:
                continue

            # Extract template window from each trial
            templates = []
            for epoch in freq_epochs:
                # Ensure we have enough samples
                if epoch.shape[1] >= start_sample + n_samples:
                    template = epoch[:, start_sample:start_sample + n_samples]
                    templates.append(template)

            if len(templates) == 0:
                continue

            # Average templates
            if self.template_config.use_averaged_templates:
                avg_template = np.mean(templates, axis=0)
            else:
                # Use all templates (for multi-template approach)
                avg_template = np.concatenate(templates, axis=1)

            self.templates[freq] = avg_template

            # Learn spatial filter using CCA with sine reference
            self._learn_spatial_filter(freq, templates)

        print(f"Loaded templates for {len(self.templates)} frequencies")

    def _learn_spatial_filter(
        self,
        freq: float,
        templates: List[np.ndarray]
    ) -> None:
        """Learn spatial filter from templates using CCA.

        Args:
            freq: Target frequency
            templates: List of template arrays
        """
        if len(templates) == 0:
            return

        # Concatenate all templates
        all_data = np.concatenate(templates, axis=1)  # (n_channels, total_samples)
        n_samples = all_data.shape[1]

        # Generate sine reference for this frequency
        t = np.arange(n_samples) / self.config.fs
        refs = []
        for h in range(1, self.config.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        ref_signal = np.array(refs).T  # (n_samples, n_harmonics*2)

        # Fit CCA
        try:
            cca = CCA(n_components=self.template_config.n_components)
            cca.fit(all_data.T, ref_signal)
            self.spatial_filters[freq] = cca.x_weights_
        except Exception as e:
            print(f"Warning: Could not learn spatial filter for {freq} Hz: {e}")

    def load_from_file(self, npz_path: str) -> None:
        """Load calibration from file.

        Args:
            npz_path: Path to calibration .npz file
        """
        data = CalibrationCollector.load(npz_path)
        self.load_calibration(data)

    def has_templates(self) -> bool:
        """Check if templates are loaded."""
        return len(self.templates) > 0

    def compute_correlations(
        self,
        window: np.ndarray
    ) -> Dict[float, float]:
        """Compute correlations for all frequencies.

        Uses weighted combination of standard CCA and template correlation.

        Args:
            window: EEG data (n_channels, n_samples)

        Returns:
            Dict mapping frequency -> combined correlation
        """
        correlations = {}

        for freq in self.config.target_frequencies:
            # Standard CCA correlation
            std_corr = self._standard_cca_correlation(window, freq)

            # Template correlation (if available)
            if freq in self.templates:
                tmpl_corr = self._template_correlation(window, freq)
                # Weighted combination
                combined = (
                    self.template_config.standard_weight * std_corr +
                    self.template_config.template_weight * tmpl_corr
                )
            else:
                combined = std_corr

            correlations[freq] = combined

        return correlations

    def _standard_cca_correlation(
        self,
        window: np.ndarray,
        freq: float
    ) -> float:
        """Compute standard CCA correlation with sine reference.

        Args:
            window: EEG window
            freq: Target frequency

        Returns:
            Maximum canonical correlation
        """
        n_samples = window.shape[1]
        t = np.arange(n_samples) / self.config.fs

        # Generate reference
        refs = []
        for h in range(1, self.config.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        ref_signal = np.array(refs).T  # (n_samples, n_harmonics*2)

        # CCA
        try:
            cca = CCA(n_components=1)
            X_c, Y_c = cca.fit_transform(window.T, ref_signal)
            corr = np.abs(np.corrcoef(X_c.flatten(), Y_c.flatten())[0, 1])
            return corr if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _template_correlation(
        self,
        window: np.ndarray,
        freq: float
    ) -> float:
        """Compute correlation with template.

        Uses spatial filtering if available.

        Args:
            window: EEG window
            freq: Target frequency

        Returns:
            Template correlation
        """
        if freq not in self.templates:
            return 0.0

        template = self.templates[freq]

        # Match window size to template
        n_samples = min(window.shape[1], template.shape[1])
        window_cut = window[:, :n_samples]
        template_cut = template[:, :n_samples]

        try:
            # If spatial filter is available, apply it
            if freq in self.spatial_filters:
                sf = self.spatial_filters[freq]
                # Project both to spatial filter
                win_proj = np.dot(sf.T, window_cut)
                tmpl_proj = np.dot(sf.T, template_cut)
                # Correlation of projected signals
                corr = np.abs(np.corrcoef(win_proj.flatten(), tmpl_proj.flatten())[0, 1])
            else:
                # Use CCA between window and template
                cca = CCA(n_components=1)
                X_c, Y_c = cca.fit_transform(window_cut.T, template_cut.T)
                corr = np.abs(np.corrcoef(X_c.flatten(), Y_c.flatten())[0, 1])

            return corr if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def step(self, window: np.ndarray) -> DecisionResult:
        """Process one window and return decision.

        Args:
            window: EEG data (n_channels, n_samples)

        Returns:
            DecisionResult with classification info
        """
        import time
        start_time = time.perf_counter()

        # Compute correlations
        if self.has_templates():
            correlations = self.compute_correlations(window)
        else:
            # Fallback to standard decoder
            return self.standard_decoder.step(window)

        # Find best frequency
        sorted_freqs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        best_freq, best_corr = sorted_freqs[0]
        second_corr = sorted_freqs[1][1] if len(sorted_freqs) > 1 else 0

        margin = best_corr - second_corr

        # Decision logic
        instantaneous = None
        if best_corr >= self.config.confidence_threshold and margin >= self.config.margin_threshold:
            instantaneous = best_freq

        # Voting
        self._recent_predictions.append(instantaneous)
        if len(self._recent_predictions) > self.config.agreement_window:
            self._recent_predictions.pop(0)

        committed = None
        if len(self._recent_predictions) >= self.config.agreement_window:
            if all(p == instantaneous and p is not None for p in self._recent_predictions):
                committed = instantaneous

        processing_time = (time.perf_counter() - start_time) * 1000
        timestamp_ms = time.time() * 1000

        return DecisionResult(
            correlations=correlations,
            max_corr=best_corr,
            margin=margin,
            instantaneous_prediction=instantaneous,
            committed_prediction=committed,
            confidence_met=best_corr >= self.config.confidence_threshold,
            margin_met=margin >= self.config.margin_threshold,
            processing_time_ms=processing_time,
            timestamp_ms=timestamp_ms
        )

    def reset(self) -> None:
        """Reset voting state."""
        self._recent_predictions = []
        self.standard_decoder.reset()


# Unit test
if __name__ == "__main__":
    print("Template CCA Decoder Test")
    print("=" * 50)

    # Create synthetic calibration data
    config = SSVEPConfig()
    fs = config.fs
    n_channels = 8
    n_samples = fs * 4  # 4 seconds

    # Generate fake epochs
    n_trials = 20  # 5 per frequency
    epochs = np.zeros((n_trials, n_channels, n_samples))
    labels = np.zeros(n_trials)
    event_ids = np.zeros(n_trials, dtype=int)

    freqs = list(config.target_frequencies) * 5
    np.random.shuffle(freqs)

    from models.calibration import FREQ_TO_EVENT_ID

    for i, freq in enumerate(freqs):
        # Generate SSVEP-like signal
        t = np.arange(n_samples) / fs
        ssvep = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
        for ch in range(n_channels):
            epochs[i, ch] = ssvep * (1 + 0.1 * ch) + np.random.randn(n_samples) * 0.5
        labels[i] = freq
        event_ids[i] = FREQ_TO_EVENT_ID[freq]

    # Create calibration data
    cal_data = CalibrationData(
        epochs=epochs,
        labels=labels,
        event_ids=event_ids,
        subject_id="test",
        session_time="20240101_000000",
        fs=fs,
        channel_names=config.electrode_names,
        frequencies=config.target_frequencies,
        trial_duration=4.0,
        trials_per_frequency=5
    )

    # Create decoder with templates
    decoder = TemplateCCADecoder(config, calibration_data=cal_data)

    print(f"\nTemplates loaded: {decoder.has_templates()}")
    print(f"Frequencies: {list(decoder.templates.keys())}")

    # Test classification
    print("\nTesting classification...")
    for freq in config.target_frequencies:
        # Generate test window
        t = np.arange(63) / fs  # 252ms window
        test_signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        test_window = np.zeros((n_channels, 63))
        for ch in range(n_channels):
            test_window[ch] = test_signal + np.random.randn(63) * 0.3

        result = decoder.step(test_window)
        print(f"\n  True: {freq:.2f} Hz")
        print(f"  Correlations: {', '.join(f'{f:.1f}:{c:.3f}' for f, c in result.correlations.items())}")
        print(f"  Instant: {result.instantaneous_prediction}")
        print(f"  Latency: {result.processing_time_ms:.2f} ms")
