#!/usr/bin/env python3
"""
SSVEP BCI Command-Line Test

Real-time SSVEP classification using CCA with OpenBCI Cyton.
This script demonstrates the complete pipeline:
1. Connect to Cyton (or use synthetic data for testing)
2. Stream EEG data
3. Apply preprocessing (CAR + bandpass)
4. Compute CCA correlations for each target frequency
5. Apply decision logic with temporal voting
6. Log results to CSV

Usage:
    python cli_test.py                    # Use synthetic data
    python cli_test.py --port COM3        # Connect to Cyton on COM3
    python cli_test.py --duration 60      # Run for 60 seconds
    python cli_test.py --target 10.0      # Simulate 10 Hz SSVEP (synthetic only)
"""

import argparse
import sys
import time
import signal
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import SSVEPConfig, create_config
from models.eeg_buffer import EEGBuffer
from models.preprocessor import SSVEPPreprocessor
from models.cca_decoder import SSVEPDecoder
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.data_logger import DataLogger


class SSVEPPipeline:
    """Complete SSVEP classification pipeline.

    Integrates all components: acquisition, buffering, preprocessing,
    classification, and logging.
    """

    def __init__(
        self,
        config: SSVEPConfig = None,
        serial_port: str = None,
        use_synthetic: bool = True,
        synthetic_target: float = 10.0,
        log_results: bool = True,
        use_preprocessing: bool = False
    ):
        """Initialize the pipeline.

        Args:
            config: SSVEPConfig instance
            serial_port: Serial port for Cyton (e.g., "COM3")
            use_synthetic: Use synthetic data for testing
            synthetic_target: Target frequency for synthetic data
            log_results: Whether to log results to CSV
            use_preprocessing: Whether to apply CAR + bandpass filtering
        """
        self.config = config or SSVEPConfig()
        self.use_synthetic = use_synthetic
        self.synthetic_target = synthetic_target
        self.log_results = log_results
        self.use_preprocessing = use_preprocessing

        # Initialize components
        self.buffer = EEGBuffer(self.config)
        self.preprocessor = SSVEPPreprocessor(self.config)
        self.decoder = SSVEPDecoder(self.config)

        # Driver
        if use_synthetic:
            self.driver = SyntheticSSVEPDriver(
                self.config,
                target_frequency=synthetic_target
            )
        else:
            self.driver = BrainFlowDriver(self.config, use_synthetic=False)

        self.serial_port = serial_port

        # Logger
        self.logger = DataLogger(self.config) if log_results else None

        # State
        self.is_running = False
        self._n_windows = 0
        self._start_time = None

        # Warmup: skip first N windows for filter stabilization
        self._warmup_windows = 10
        self._warmup_count = 0

        # Statistics
        self._latencies = []
        self._predictions = []

    def connect(self) -> bool:
        """Connect to the EEG device.

        Returns:
            True if connection successful
        """
        print(f"Connecting to {'synthetic' if self.use_synthetic else 'Cyton'} board...")

        if self.driver.connect(self.serial_port):
            print(f"  EEG channels: {self.driver.eeg_channels}")
            print(f"  Sampling rate: {self.driver.sampling_rate} Hz")
            return True
        else:
            print("  Connection failed!")
            return False

    def start(self) -> bool:
        """Start the acquisition and processing.

        Returns:
            True if started successfully
        """
        if not self.driver.is_connected:
            if not self.connect():
                return False

        # Open logger
        if self.logger:
            log_path = self.logger.open()
            print(f"Logging to: {log_path}")

        # Reset components
        self.buffer.reset()
        self.preprocessor.reset()
        self.decoder.reset()
        self._latencies = []
        self._predictions = []
        self._n_windows = 0
        self._warmup_count = 0

        # Start streaming
        if not self.driver.start_stream():
            print("Failed to start streaming!")
            return False

        self.is_running = True
        self._start_time = time.time()

        print("\n" + "=" * 60)
        print("SSVEP Classification Started")
        print("=" * 60)
        print(f"Target frequencies: {self.config.target_frequencies} Hz")
        print(f"Window: {self.config.window_duration_ms:.0f} ms, "
              f"Step: {self.config.step_duration_ms:.0f} ms")
        print(f"Confidence threshold: {self.config.confidence_threshold}")
        print(f"Margin threshold: {self.config.margin_threshold}")
        print("=" * 60 + "\n")

        return True

    def stop(self) -> dict:
        """Stop acquisition and return summary statistics.

        Returns:
            Dictionary with session statistics
        """
        self.is_running = False
        self.driver.stop_stream()

        # Close logger
        if self.logger:
            self.logger.close()

        # Compute statistics
        duration = time.time() - self._start_time if self._start_time else 0

        stats = {
            'duration_s': duration,
            'n_windows': self._n_windows,
            'windows_per_sec': self._n_windows / duration if duration > 0 else 0,
        }

        if self._latencies:
            stats['mean_latency_ms'] = np.mean(self._latencies)
            stats['std_latency_ms'] = np.std(self._latencies)
            stats['max_latency_ms'] = np.max(self._latencies)

        if self._predictions:
            # Count predictions per frequency
            from collections import Counter
            pred_counts = Counter(self._predictions)
            stats['prediction_counts'] = dict(pred_counts)

        return stats

    def process_step(self) -> None:
        """Process one step: get data, classify, display.

        Call this repeatedly in your main loop.
        """
        if not self.is_running:
            return

        # Get new data from driver
        data = self.driver.get_data()

        if data is not None and data.shape[1] > 0:
            # Add to buffer
            self.buffer.append(data)

        # Process windows while available
        while self.buffer.ready():
            window = self.buffer.get_window()
            if window is None:
                break

            # Preprocess if enabled
            if self.use_preprocessing:
                processed = self.preprocessor.process(window)
            else:
                processed = window

            # Skip classification during warmup (filter stabilization)
            if self.use_preprocessing and self._warmup_count < self._warmup_windows:
                self._warmup_count += 1
                print(f"\rWarming up filter... {self._warmup_count}/{self._warmup_windows}", end="")
                if self._warmup_count == self._warmup_windows:
                    print("\nFilter warmed up. Starting classification.\n")
                continue

            # Classify
            result = self.decoder.step(processed)

            self._n_windows += 1
            self._latencies.append(result.processing_time_ms)

            if result.instantaneous_prediction:
                self._predictions.append(result.instantaneous_prediction)

            # Log
            if self.logger:
                self.logger.log_classification(
                    correlations=result.correlations,
                    max_corr=result.max_corr,
                    margin=result.margin,
                    confidence_met=result.confidence_met,
                    margin_met=result.margin_met,
                    instantaneous_prediction=result.instantaneous_prediction,
                    committed_prediction=result.committed_prediction,
                    processing_time_ms=result.processing_time_ms,
                    queue_state=self.decoder.get_queue_state(),
                    ground_truth=self.synthetic_target if self.use_synthetic else None,
                    timestamp_ms=result.timestamp_ms
                )

            # Display
            self._display_result(result)

    def _display_result(self, result) -> None:
        """Display classification result to console."""
        # Build correlation bar display
        corr_display = []
        for freq in self.config.target_frequencies:
            corr = result.correlations.get(freq, 0)
            bar_len = int(corr * 30)
            bar = '#' * bar_len + '-' * (30 - bar_len)

            # Highlight best
            if freq == result.instantaneous_prediction:
                marker = " <--"
            else:
                marker = ""

            corr_display.append(f"  {freq:5.2f} Hz |{bar}| {corr:.3f}{marker}")

        # Status line
        if result.committed_prediction:
            status = f"COMMITTED: {result.committed_prediction:.2f} Hz"
            status_color = "\033[92m"  # Green
        elif result.instantaneous_prediction:
            status = f"Detecting: {result.instantaneous_prediction:.2f} Hz"
            status_color = "\033[93m"  # Yellow
        else:
            status = "No detection"
            status_color = "\033[91m"  # Red

        # Clear and print
        print("\033[2J\033[H", end="")  # Clear screen
        print("=" * 60)
        print(f"SSVEP Classification - Window #{self._n_windows}")
        print("=" * 60)
        print("\nCCA Correlations:")
        print("-" * 50)
        for line in corr_display:
            print(line)
        print("-" * 50)
        print(f"\nMax correlation: {result.max_corr:.4f}")
        print(f"Margin: {result.margin:.4f}")
        print(f"Confidence met: {'YES' if result.confidence_met else 'NO'}")
        print(f"Margin met: {'YES' if result.margin_met else 'NO'}")
        print(f"\n{status_color}{status}\033[0m")
        print(f"\nQueue: {self.decoder.get_queue_state()}")
        print(f"Processing time: {result.processing_time_ms:.2f} ms")

        if self.use_synthetic:
            print(f"\n[Synthetic mode - simulating {self.synthetic_target} Hz]")

        elapsed = time.time() - self._start_time
        print(f"\nElapsed: {elapsed:.1f}s | Windows: {self._n_windows}")

    def disconnect(self) -> None:
        """Disconnect from the device."""
        if self.is_running:
            self.stop()
        self.driver.disconnect()


def main():
    """Main entry point for CLI test."""
    parser = argparse.ArgumentParser(
        description="SSVEP BCI Command-Line Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_test.py                    # Synthetic data, 10 Hz
  python cli_test.py --target 12.0      # Synthetic data, 12 Hz
  python cli_test.py --port COM3        # Real Cyton on COM3
  python cli_test.py --duration 60      # Run for 60 seconds
  python cli_test.py --no-log           # Don't save logs
        """
    )

    parser.add_argument(
        '--port', '-p',
        type=str,
        default=None,
        help='Serial port for Cyton (e.g., COM3, /dev/ttyUSB0)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=30.0,
        help='Duration to run in seconds (default: 30)'
    )

    parser.add_argument(
        '--target', '-t',
        type=float,
        default=10.0,
        choices=[8.57, 10.0, 12.0, 15.0],
        help='Target frequency for synthetic data (default: 10.0)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.55,
        help='Confidence threshold (default: 0.55)'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=0.15,
        help='Margin threshold (default: 0.15)'
    )

    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable logging to CSV'
    )

    parser.add_argument(
        '--list-ports',
        action='store_true',
        help='List available serial ports and exit'
    )

    args = parser.parse_args()

    # List ports if requested
    if args.list_ports:
        print("Available serial ports:")
        ports = BrainFlowDriver.list_serial_ports()
        if ports:
            for port in ports:
                print(f"  {port}")
        else:
            print("  No ports found")
        return 0

    # Create config with custom thresholds
    config = create_config(
        confidence_threshold=args.confidence,
        margin_threshold=args.margin
    )

    # Determine if using synthetic or real hardware
    use_synthetic = args.port is None

    # Create pipeline
    pipeline = SSVEPPipeline(
        config=config,
        serial_port=args.port,
        use_synthetic=use_synthetic,
        synthetic_target=args.target,
        log_results=not args.no_log
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nStopping...")
        pipeline.stop()
        pipeline.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start pipeline
    if not pipeline.start():
        print("Failed to start pipeline!")
        return 1

    # Main loop
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            pipeline.process_step()
            time.sleep(0.01)  # Small delay to prevent CPU spinning

    except KeyboardInterrupt:
        pass

    # Stop and show summary
    stats = pipeline.stop()
    pipeline.disconnect()

    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    print(f"Duration: {stats['duration_s']:.1f} seconds")
    print(f"Windows processed: {stats['n_windows']}")
    print(f"Windows per second: {stats['windows_per_sec']:.1f}")

    if 'mean_latency_ms' in stats:
        print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"Max latency: {stats['max_latency_ms']:.2f} ms")

    if 'prediction_counts' in stats:
        print("\nPrediction counts:")
        for freq, count in sorted(stats['prediction_counts'].items()):
            print(f"  {freq:.2f} Hz: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
