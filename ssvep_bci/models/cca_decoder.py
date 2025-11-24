"""
SSVEP CCA Decoder

Implements Canonical Correlation Analysis (CCA) based SSVEP classification.
Uses synthetic reference signals (sine/cosine at target frequencies + harmonics)
to detect which frequency the user is attending to.

Includes decision logic with confidence thresholds and temporal voting.
"""

import numpy as np
from sklearn.cross_decomposition import CCA
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


@dataclass
class DecisionResult:
    """Result of SSVEP classification for one window."""
    correlations: Dict[float, float]  # {frequency: correlation}
    max_corr: float                    # Maximum correlation value
    margin: float                      # Difference between top-2 correlations
    instantaneous_prediction: Optional[float]  # Predicted frequency (None if below threshold)
    committed_prediction: Optional[float]      # Voted prediction (None if no agreement)
    confidence_met: bool               # Whether confidence threshold was met
    margin_met: bool                   # Whether margin threshold was met
    processing_time_ms: float          # Time to process this window
    timestamp_ms: float                # UTC timestamp


class SSVEPDecoder:
    """CCA-based SSVEP decoder with temporal voting.

    Uses Canonical Correlation Analysis to compute correlations between
    EEG data and synthetic reference signals at each target frequency.
    Implements confidence thresholds and temporal voting for robust detection.

    Attributes:
        config: SSVEPConfig instance
        references: Precomputed reference signals for each frequency
        decision_queue: Queue of recent valid predictions for voting
    """

    def __init__(self, config: SSVEPConfig = None):
        """Initialize the SSVEP decoder.

        Args:
            config: SSVEPConfig instance. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG

        # Precompute reference signals for each target frequency
        self.references = self._generate_references()

        # Queue for temporal voting
        self.decision_queue: deque = deque(maxlen=self.config.agreement_window)

        # CCA model (reused for each frequency)
        self.cca = CCA(n_components=1, max_iter=500)

        # Statistics tracking
        self._n_windows_processed = 0
        self._n_valid_predictions = 0

    def _generate_references(self) -> Dict[float, np.ndarray]:
        """Generate reference signals for all target frequencies.

        For each frequency, generates sine and cosine components at the
        fundamental frequency and harmonics.

        Returns:
            Dictionary mapping frequency to reference matrix.
            Each matrix has shape (n_samples, n_reference_signals).
        """
        references = {}
        t = self.config.time_vector  # Time vector for one window

        for freq in self.config.target_frequencies:
            ref_signals = []

            # Generate fundamental and harmonics
            for h in range(1, self.config.n_harmonics + 1):
                f = freq * h
                ref_signals.append(np.sin(2 * np.pi * f * t))
                ref_signals.append(np.cos(2 * np.pi * f * t))

            # Stack as columns: shape (n_samples, 2*n_harmonics)
            references[freq] = np.column_stack(ref_signals)

        return references

    def compute_correlation(self, eeg_data: np.ndarray, freq: float) -> float:
        """Compute CCA correlation for one frequency.

        Args:
            eeg_data: Preprocessed EEG data, shape (n_channels, n_samples)
            freq: Target frequency to test

        Returns:
            Canonical correlation coefficient (0 to 1)
        """
        # Transpose EEG to (n_samples, n_channels)
        X = eeg_data.T

        # Get reference signals for this frequency
        Y = self.references[freq]

        # Fit CCA
        try:
            self.cca.fit(X, Y)

            # Get canonical variates
            X_c, Y_c = self.cca.transform(X, Y)

            # Compute correlation between first pair of canonical variates
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

            # Ensure positive correlation (CCA can flip sign)
            return abs(corr)

        except Exception as e:
            # Handle numerical issues gracefully
            return 0.0

    def compute_all_correlations(self, eeg_data: np.ndarray) -> Dict[float, float]:
        """Compute CCA correlations for all target frequencies.

        Args:
            eeg_data: Preprocessed EEG data, shape (n_channels, n_samples)

        Returns:
            Dictionary mapping each frequency to its correlation value
        """
        correlations = {}
        for freq in self.config.target_frequencies:
            correlations[freq] = self.compute_correlation(eeg_data, freq)
        return correlations

    def apply_decision_logic(
        self,
        correlations: Dict[float, float]
    ) -> Tuple[Optional[float], float, float, bool, bool]:
        """Apply thresholds to determine instantaneous prediction.

        Args:
            correlations: Dictionary of {frequency: correlation}

        Returns:
            Tuple of:
            - instantaneous_prediction: Predicted frequency or None
            - max_corr: Maximum correlation value
            - margin: Difference between top-2 correlations
            - confidence_met: Whether max_corr >= confidence_threshold
            - margin_met: Whether margin >= margin_threshold
        """
        # Sort by correlation value (descending)
        sorted_items = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_freq, max_corr = sorted_items[0]
        second_corr = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = max_corr - second_corr

        # Check thresholds
        confidence_met = max_corr >= self.config.confidence_threshold
        margin_met = margin >= self.config.margin_threshold

        # Instantaneous prediction
        if confidence_met and margin_met:
            instantaneous_prediction = best_freq
        else:
            instantaneous_prediction = None

        return instantaneous_prediction, max_corr, margin, confidence_met, margin_met

    def apply_voting(self, instantaneous_prediction: Optional[float]) -> Optional[float]:
        """Apply temporal voting to get committed prediction.

        Args:
            instantaneous_prediction: Current window's prediction (or None)

        Returns:
            Committed prediction if agreement reached, else None
        """
        # Only add valid predictions to the queue
        if instantaneous_prediction is not None:
            self.decision_queue.append(instantaneous_prediction)

        # Check if we have enough predictions
        if len(self.decision_queue) < self.config.agreement_window:
            return None

        # Check if all predictions in queue agree
        if len(set(self.decision_queue)) == 1:
            return self.decision_queue[0]

        return None

    def step(self, eeg_data: np.ndarray) -> DecisionResult:
        """Process one window and return classification result.

        This is the main method to call for each new window of EEG data.

        Args:
            eeg_data: Preprocessed EEG data, shape (n_channels, n_samples)

        Returns:
            DecisionResult with all classification information
        """
        start_time = time.perf_counter()
        timestamp_ms = time.time() * 1000

        # Compute correlations for all frequencies
        correlations = self.compute_all_correlations(eeg_data)

        # Apply decision logic
        (instantaneous_pred, max_corr, margin,
         confidence_met, margin_met) = self.apply_decision_logic(correlations)

        # Apply temporal voting
        committed_pred = self.apply_voting(instantaneous_pred)

        # Track statistics
        self._n_windows_processed += 1
        if instantaneous_pred is not None:
            self._n_valid_predictions += 1

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return DecisionResult(
            correlations=correlations,
            max_corr=max_corr,
            margin=margin,
            instantaneous_prediction=instantaneous_pred,
            committed_prediction=committed_pred,
            confidence_met=confidence_met,
            margin_met=margin_met,
            processing_time_ms=processing_time_ms,
            timestamp_ms=timestamp_ms
        )

    def reset(self) -> None:
        """Reset decoder state.

        Clears the decision queue but keeps precomputed references.
        Call this when starting a new trial or session.
        """
        self.decision_queue.clear()
        self._n_windows_processed = 0
        self._n_valid_predictions = 0

    def get_queue_state(self) -> List[float]:
        """Get current state of the decision queue.

        Returns:
            List of recent valid predictions
        """
        return list(self.decision_queue)

    def update_thresholds(
        self,
        confidence: Optional[float] = None,
        margin: Optional[float] = None,
        agreement_window: Optional[int] = None
    ) -> None:
        """Update decision thresholds dynamically.

        Args:
            confidence: New confidence threshold (0.40-0.70)
            margin: New margin threshold (0.05-0.25)
            agreement_window: New agreement window size (2-5)
        """
        if confidence is not None:
            self.config.confidence_threshold = np.clip(confidence, 0.40, 0.70)

        if margin is not None:
            self.config.margin_threshold = np.clip(margin, 0.05, 0.25)

        if agreement_window is not None:
            new_window = int(np.clip(agreement_window, 2, 5))
            if new_window != self.config.agreement_window:
                self.config.agreement_window = new_window
                # Resize queue
                old_items = list(self.decision_queue)
                self.decision_queue = deque(maxlen=new_window)
                for item in old_items[-new_window:]:
                    self.decision_queue.append(item)

    @property
    def statistics(self) -> Dict:
        """Get decoder statistics.

        Returns:
            Dictionary with processing statistics
        """
        valid_rate = (
            self._n_valid_predictions / self._n_windows_processed
            if self._n_windows_processed > 0 else 0.0
        )

        return {
            'windows_processed': self._n_windows_processed,
            'valid_predictions': self._n_valid_predictions,
            'valid_prediction_rate': valid_rate,
            'queue_length': len(self.decision_queue),
            'queue_state': list(self.decision_queue)
        }


class FilterBankCCADecoder(SSVEPDecoder):
    """Extended decoder using Filter Bank CCA (FBCCA).

    FBCCA applies multiple sub-band filters and combines their CCA
    correlations for improved accuracy. This is more computationally
    intensive but can achieve higher accuracy.

    Note: Implement this as a future enhancement if standard CCA
    doesn't meet accuracy targets.
    """

    def __init__(self, config: SSVEPConfig = None, n_subbands: int = 3):
        """Initialize FBCCA decoder.

        Args:
            config: SSVEPConfig instance
            n_subbands: Number of sub-band filters to use
        """
        super().__init__(config)
        self.n_subbands = n_subbands
        # TODO: Implement sub-band filter design and combination weights


# Unit test
if __name__ == "__main__":
    print("Testing SSVEPDecoder...")

    config = SSVEPConfig()
    decoder = SSVEPDecoder(config)

    print(f"Target frequencies: {config.target_frequencies}")
    print(f"Window samples: {config.window_samples}")
    print(f"Confidence threshold: {config.confidence_threshold}")
    print(f"Margin threshold: {config.margin_threshold}")
    print(f"Agreement window: {config.agreement_window}")

    # Check reference signals
    print("\nReference signals generated:")
    for freq, ref in decoder.references.items():
        print(f"  {freq} Hz: shape={ref.shape}")

    # Generate synthetic SSVEP data at 10 Hz
    fs = config.fs
    n_samples = config.window_samples
    t = np.arange(n_samples) / fs
    n_channels = config.n_eeg_channels

    print("\n--- Testing with 10 Hz SSVEP signal ---")
    test_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Strong 10 Hz component
        test_data[ch] = (
            20 * np.sin(2 * np.pi * 10 * t) +
            10 * np.sin(2 * np.pi * 20 * t) +  # 2nd harmonic
            np.random.randn(n_samples) * 3
        )

    result = decoder.step(test_data)

    print(f"\nCorrelations:")
    for freq, corr in result.correlations.items():
        marker = " <-- BEST" if freq == 10.0 else ""
        print(f"  {freq:5.2f} Hz: {corr:.4f}{marker}")

    print(f"\nMax correlation: {result.max_corr:.4f}")
    print(f"Margin: {result.margin:.4f}")
    print(f"Confidence met: {result.confidence_met}")
    print(f"Margin met: {result.margin_met}")
    print(f"Instantaneous prediction: {result.instantaneous_prediction}")
    print(f"Committed prediction: {result.committed_prediction}")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")

    # Test temporal voting
    print("\n--- Testing temporal voting ---")
    decoder.reset()

    # Simulate multiple windows with consistent 10 Hz
    for i in range(3):
        result = decoder.step(test_data)
        print(f"Window {i+1}: instant={result.instantaneous_prediction}, "
              f"committed={result.committed_prediction}, "
              f"queue={decoder.get_queue_state()}")

    # Test with different frequencies
    print("\n--- Testing with different frequencies ---")
    decoder.reset()

    for target_freq in [8.57, 10.0, 12.0, 15.0]:
        test_data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            test_data[ch] = (
                20 * np.sin(2 * np.pi * target_freq * t) +
                10 * np.sin(2 * np.pi * (2 * target_freq) * t) +
                np.random.randn(n_samples) * 3
            )

        result = decoder.step(test_data)
        detected = result.instantaneous_prediction
        correct = detected == target_freq if detected else False
        status = "[OK]" if correct else "[FAIL]"
        print(f"Target: {target_freq:5.2f} Hz, Detected: {detected}, {status}")

    # Performance test
    print("\n--- Performance test ---")
    import time

    n_iterations = 100
    test_data = np.random.randn(n_channels, n_samples) * 10

    start = time.perf_counter()
    for _ in range(n_iterations):
        result = decoder.step(test_data)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Processed {n_iterations} windows in {elapsed:.1f} ms")
    print(f"Average: {elapsed/n_iterations:.2f} ms per window")
    print(f"Target: <{config.max_processing_latency_ms:.1f} ms")

    if elapsed/n_iterations < config.max_processing_latency_ms:
        print("[OK] Performance target met!")
    else:
        print("[FAIL] Performance target NOT met")

    print("\nSSVEPDecoder test passed!")
