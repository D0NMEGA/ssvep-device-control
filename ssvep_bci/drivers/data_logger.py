"""
Data Logger for SSVEP BCI

Logs classification results, timing, and EEG data to CSV files
for offline analysis.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


class DataLogger:
    """CSV logger for SSVEP BCI session data.

    Logs classification results including correlations, predictions,
    timing, and optional ground truth labels for accuracy analysis.

    Attributes:
        config: SSVEPConfig instance
        log_dir: Directory for log files
        session_id: Unique session identifier
    """

    def __init__(self, config: SSVEPConfig = None, log_dir: str = None):
        """Initialize the data logger.

        Args:
            config: SSVEPConfig instance
            log_dir: Directory for log files. If None, uses config default.
        """
        self.config = config or DEFAULT_CONFIG
        self.log_dir = Path(log_dir or self.config.log_directory)

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._is_open = False

        # File handles and writers
        self._classification_file = None
        self._classification_writer = None

        self._raw_file = None
        self._raw_writer = None

        # Data accumulator for batch writing
        self._classification_buffer: List[Dict] = []
        self._raw_buffer: List[np.ndarray] = []

        # Statistics
        self._n_entries = 0
        self._session_start = None

    def open(self, session_name: str = None) -> str:
        """Open log files for a new session.

        Args:
            session_name: Optional name for the session

        Returns:
            Path to the classification log file
        """
        if self._is_open:
            self.close()

        # Generate filenames
        name = session_name or self.config.log_prefix
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{name}_{self.session_id}"

        # Classification log
        class_path = self.log_dir / f"{base_name}_classification.csv"
        self._classification_file = open(class_path, 'w', newline='')
        self._classification_writer = csv.DictWriter(
            self._classification_file,
            fieldnames=self._get_classification_headers()
        )
        self._classification_writer.writeheader()

        self._is_open = True
        self._session_start = datetime.now()
        self._n_entries = 0

        return str(class_path)

    def _get_classification_headers(self) -> List[str]:
        """Get column headers for classification log."""
        headers = [
            'timestamp_ms',
            'window_index',
        ]

        # Add correlation columns for each frequency
        for freq in self.config.target_frequencies:
            headers.append(f'corr_{freq:.2f}Hz')

        headers.extend([
            'max_correlation',
            'margin',
            'confidence_met',
            'margin_met',
            'instantaneous_prediction',
            'committed_prediction',
            'ground_truth',
            'correct',
            'processing_time_ms',
            'queue_state'
        ])

        return headers

    def log_classification(
        self,
        correlations: Dict[float, float],
        max_corr: float,
        margin: float,
        confidence_met: bool,
        margin_met: bool,
        instantaneous_prediction: Optional[float],
        committed_prediction: Optional[float],
        processing_time_ms: float,
        queue_state: List[float],
        ground_truth: Optional[float] = None,
        timestamp_ms: float = None
    ) -> None:
        """Log a classification result.

        Args:
            correlations: Dictionary of {frequency: correlation}
            max_corr: Maximum correlation value
            margin: Difference between top-2 correlations
            confidence_met: Whether confidence threshold was met
            margin_met: Whether margin threshold was met
            instantaneous_prediction: Current window prediction
            committed_prediction: Voted prediction
            processing_time_ms: Processing time in milliseconds
            queue_state: Current decision queue state
            ground_truth: Optional ground truth label for accuracy calculation
            timestamp_ms: UTC timestamp in milliseconds
        """
        if not self._is_open:
            return

        self._n_entries += 1

        # Build row
        row = {
            'timestamp_ms': timestamp_ms or datetime.now().timestamp() * 1000,
            'window_index': self._n_entries,
        }

        # Add correlations
        for freq in self.config.target_frequencies:
            row[f'corr_{freq:.2f}Hz'] = f"{correlations.get(freq, 0.0):.6f}"

        # Add decision info
        row['max_correlation'] = f"{max_corr:.6f}"
        row['margin'] = f"{margin:.6f}"
        row['confidence_met'] = confidence_met
        row['margin_met'] = margin_met
        row['instantaneous_prediction'] = instantaneous_prediction or ''
        row['committed_prediction'] = committed_prediction or ''
        row['ground_truth'] = ground_truth or ''

        # Calculate correctness if ground truth provided
        if ground_truth is not None and committed_prediction is not None:
            row['correct'] = committed_prediction == ground_truth
        else:
            row['correct'] = ''

        row['processing_time_ms'] = f"{processing_time_ms:.3f}"
        row['queue_state'] = str(queue_state)

        # Write to file
        self._classification_writer.writerow(row)
        self._classification_file.flush()

    def log_raw_eeg(self, data: np.ndarray, timestamp_ms: float = None) -> None:
        """Log raw EEG data.

        Args:
            data: EEG data with shape (n_channels, n_samples)
            timestamp_ms: UTC timestamp for first sample
        """
        if self._raw_file is None:
            # Open raw data file on first call
            base_name = f"{self.config.log_prefix}_{self.session_id}"
            raw_path = self.log_dir / f"{base_name}_raw.csv"

            headers = ['timestamp_ms'] + [f'ch{i}' for i in range(data.shape[0])]
            self._raw_file = open(raw_path, 'w', newline='')
            self._raw_writer = csv.writer(self._raw_file)
            self._raw_writer.writerow(headers)

        # Write each sample
        ts = timestamp_ms or datetime.now().timestamp() * 1000
        dt = 1000.0 / self.config.fs  # ms per sample

        for i in range(data.shape[1]):
            row = [ts + i * dt] + data[:, i].tolist()
            self._raw_writer.writerow(row)

        self._raw_file.flush()

    def close(self) -> Dict[str, Any]:
        """Close log files and return session summary.

        Returns:
            Dictionary with session statistics
        """
        summary = {
            'session_id': self.session_id,
            'n_entries': self._n_entries,
            'duration_s': 0.0
        }

        if self._session_start:
            summary['duration_s'] = (
                datetime.now() - self._session_start
            ).total_seconds()

        if self._classification_file:
            self._classification_file.close()
            self._classification_file = None

        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None

        self._is_open = False

        return summary

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class SessionAnalyzer:
    """Analyze logged SSVEP sessions.

    Computes accuracy, ITR, and other metrics from log files.
    """

    def __init__(self, log_path: str):
        """Initialize analyzer with a log file.

        Args:
            log_path: Path to classification log CSV file
        """
        self.log_path = Path(log_path)
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        """Load the log file.

        Returns:
            True if loaded successfully
        """
        try:
            self.df = pd.read_csv(self.log_path)
            return True
        except Exception as e:
            print(f"Error loading log: {e}")
            return False

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute classification accuracy.

        Returns:
            Dictionary with accuracy metrics
        """
        if self.df is None:
            return {}

        # Filter rows with ground truth
        gt_df = self.df[self.df['ground_truth'].notna() & (self.df['ground_truth'] != '')]

        if len(gt_df) == 0:
            return {'accuracy': None, 'n_trials': 0}

        # Committed predictions accuracy
        committed_df = gt_df[gt_df['committed_prediction'].notna() & (gt_df['committed_prediction'] != '')]

        if len(committed_df) > 0:
            committed_df['correct'] = committed_df['committed_prediction'].astype(float) == committed_df['ground_truth'].astype(float)
            committed_accuracy = committed_df['correct'].mean()
        else:
            committed_accuracy = None

        # Instantaneous predictions accuracy
        instant_df = gt_df[gt_df['instantaneous_prediction'].notna() & (gt_df['instantaneous_prediction'] != '')]

        if len(instant_df) > 0:
            instant_df['correct'] = instant_df['instantaneous_prediction'].astype(float) == instant_df['ground_truth'].astype(float)
            instant_accuracy = instant_df['correct'].mean()
        else:
            instant_accuracy = None

        return {
            'committed_accuracy': committed_accuracy,
            'instantaneous_accuracy': instant_accuracy,
            'n_trials_with_gt': len(gt_df),
            'n_committed': len(committed_df),
            'n_instant': len(instant_df)
        }

    def compute_itr(self, n_classes: int = 4, trial_duration_s: float = 1.0) -> float:
        """Compute Information Transfer Rate.

        ITR (bits/min) = (log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))) * 60/T

        Args:
            n_classes: Number of classes (default 4)
            trial_duration_s: Average trial duration in seconds

        Returns:
            ITR in bits per minute
        """
        accuracy = self.compute_accuracy()
        p = accuracy.get('committed_accuracy')

        if p is None or p <= 1/n_classes:
            return 0.0

        # Clip to avoid log(0)
        p = np.clip(p, 0.001, 0.999)

        n = n_classes
        if p == 1.0:
            itr_per_trial = np.log2(n)
        else:
            itr_per_trial = (
                np.log2(n) +
                p * np.log2(p) +
                (1 - p) * np.log2((1 - p) / (n - 1))
            )

        itr_per_min = itr_per_trial * (60.0 / trial_duration_s)

        return max(0.0, itr_per_min)

    def compute_latency_stats(self) -> Dict[str, float]:
        """Compute processing latency statistics.

        Returns:
            Dictionary with latency statistics
        """
        if self.df is None:
            return {}

        latencies = pd.to_numeric(self.df['processing_time_ms'], errors='coerce')
        latencies = latencies.dropna()

        if len(latencies) == 0:
            return {}

        return {
            'mean_latency_ms': latencies.mean(),
            'std_latency_ms': latencies.std(),
            'min_latency_ms': latencies.min(),
            'max_latency_ms': latencies.max(),
            'median_latency_ms': latencies.median()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get full session summary.

        Returns:
            Dictionary with all computed metrics
        """
        summary = {}
        summary.update(self.compute_accuracy())
        summary.update(self.compute_latency_stats())
        summary['itr_bits_per_min'] = self.compute_itr()

        return summary


# Unit test
if __name__ == "__main__":
    print("Testing DataLogger...")

    config = SSVEPConfig()
    logger = DataLogger(config, log_dir="test_logs")

    # Open session
    log_path = logger.open(session_name="test_session")
    print(f"Log file: {log_path}")

    # Log some fake classification results
    import random

    for i in range(10):
        correlations = {
            8.57: random.uniform(0.3, 0.8),
            10.0: random.uniform(0.4, 0.9),
            12.0: random.uniform(0.3, 0.7),
            15.0: random.uniform(0.2, 0.6)
        }

        max_corr = max(correlations.values())
        sorted_corrs = sorted(correlations.values(), reverse=True)
        margin = sorted_corrs[0] - sorted_corrs[1]

        logger.log_classification(
            correlations=correlations,
            max_corr=max_corr,
            margin=margin,
            confidence_met=max_corr >= 0.55,
            margin_met=margin >= 0.15,
            instantaneous_prediction=10.0 if max_corr >= 0.55 else None,
            committed_prediction=10.0 if i >= 2 else None,
            processing_time_ms=random.uniform(5, 15),
            queue_state=[10.0, 10.0] if i >= 1 else [],
            ground_truth=10.0
        )

    # Close and get summary
    summary = logger.close()
    print(f"\nSession summary: {summary}")

    # Analyze the log
    print("\n--- Analyzing logged session ---")
    analyzer = SessionAnalyzer(log_path)

    if analyzer.load():
        print(f"Loaded {len(analyzer.df)} entries")

        accuracy = analyzer.compute_accuracy()
        print(f"Accuracy: {accuracy}")

        latency = analyzer.compute_latency_stats()
        print(f"Latency: {latency}")

        itr = analyzer.compute_itr()
        print(f"ITR: {itr:.2f} bits/min")

    # Cleanup test files
    import shutil
    shutil.rmtree("test_logs", ignore_errors=True)

    print("\nDataLogger test passed!")
