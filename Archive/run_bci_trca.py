#!/usr/bin/env python3
"""
SSVEP BCI with TRCA Classification

Real-time SSVEP-BCI using trained TRCA model for classification.
Uses causal preprocessing for online data processing.

Usage:
    python run_bci_trca.py models/trca_model.pkl
    python run_bci_trca.py models/trca_model.pkl --arduino COM3 --cyton COM4
    python run_bci_trca.py models/trca_model.pkl --synthetic
"""

import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig
from models.preprocessor import SSVEPPreprocessor  # Causal preprocessor for online use
from models.eeg_buffer import EEGBuffer
from models.trca import TRCA
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController


class BCIRunnerTRCA:
    """Real-time BCI runner using TRCA classification."""

    def __init__(
        self,
        model_path: str,
        arduino_port: str = None,
        cyton_port: str = None,
        use_synthetic: bool = False,
        window_ms: int = 250,
        confidence_threshold: float = 0.05,
        margin_threshold: float = 0.02
    ):
        """Initialize BCI runner with TRCA model.

        Args:
            model_path: Path to trained TRCA model pickle file
            arduino_port: Arduino serial port
            cyton_port: OpenBCI Cyton serial port
            use_synthetic: Use synthetic data
            window_ms: Analysis window duration in milliseconds
            confidence_threshold: Minimum correlation for valid prediction
            margin_threshold: Minimum margin between top 2 correlations
        """
        # Load TRCA model
        print(f"Loading TRCA model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.trca: TRCA = model_data['trca']
        self.config: SSVEPConfig = model_data['config']
        self.training_accuracy = model_data.get('training_accuracy', 0.0)
        self.test_accuracy = model_data.get('test_accuracy', 0.0)

        print(f"  Training accuracy: {self.training_accuracy*100:.1f}%")
        print(f"  Test accuracy: {self.test_accuracy*100:.1f}%")
        print(f"  Frequencies: {self.config.target_frequencies}")

        # Update window size
        self.window_samples = int(window_ms * self.config.fs / 1000)
        print(f"  Analysis window: {window_ms} ms ({self.window_samples} samples)")

        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        # Online preprocessor (causal)
        self.preprocessor = SSVEPPreprocessor(self.config)

        # EEG buffer
        self.buffer = EEGBuffer(self.config)

        # Arduino controller
        self.arduino = ArduinoController(port=arduino_port)

        # EEG driver
        if use_synthetic:
            self.eeg_driver = SyntheticSSVEPDriver(self.config, target_frequency=10.0)
        else:
            self.eeg_driver = BrainFlowDriver(self.config)
            self.eeg_driver.config.serial_port = cyton_port

        # State
        self.is_running = False
        self.last_prediction = None
        self.last_correlations = None

    def connect(self) -> bool:
        """Connect to hardware."""
        print("\n" + "=" * 60)
        print("SSVEP BCI with TRCA")
        print("=" * 60)

        # Connect Arduino
        print("\n[1/2] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  WARNING: Arduino not connected. Feedback unavailable.")
        else:
            print(f"  Connected on {self.arduino.port}")

        # Connect EEG
        mode = "Synthetic" if isinstance(self.eeg_driver, SyntheticSSVEPDriver) else "OpenBCI Cyton"
        print(f"\n[2/2] Connecting to {mode}...")
        if not self.eeg_driver.connect():
            print("  ERROR: Failed to connect to EEG device")
            return False
        print(f"  Connected! Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        return True

    def disconnect(self):
        """Disconnect hardware."""
        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()
            self.arduino.disconnect()
        self.eeg_driver.disconnect()

    def run(self):
        """Run real-time BCI loop."""
        print("\n" + "=" * 60)
        print("Real-Time Classification")
        print("=" * 60)

        print("\nFrequencies:")
        for i, freq in enumerate(self.config.target_frequencies):
            print(f"  {i}: {freq:.2f} Hz")

        print(f"\nThresholds:")
        print(f"  Confidence: {self.confidence_threshold:.3f}")
        print(f"  Margin: {self.margin_threshold:.3f}")

        print("\nStarting in 3 seconds...")
        time.sleep(3)

        # Start EEG
        if not self.eeg_driver.start_stream():
            print("ERROR: Failed to start EEG stream")
            return False

        self.is_running = True
        self.buffer.reset()
        self.preprocessor.reset()

        # Start LEDs
        if self.arduino.is_connected:
            self.arduino.start_stimulation()

        print("\nBCI running! Press Ctrl+C to stop.\n")

        try:
            window_count = 0

            while self.is_running:
                # Check emergency stop button
                if self.arduino.is_connected and self.arduino.check_button_pressed():
                    print("\n[EMERGENCY STOP] Button pressed!")
                    break

                # Get new data
                data = self.eeg_driver.get_data()
                if data is not None and data.shape[1] > 0:
                    self.buffer.add_data(data)

                # Check if we have enough data
                if self.buffer.get_num_samples() >= self.window_samples:
                    window_count += 1

                    # Get window
                    window = self.buffer.get_window(self.window_samples)

                    # Preprocess (causal, online)
                    window_preprocessed = self.preprocessor.process(window)

                    # Classify with TRCA
                    pred_idx, correlations = self.trca.predict_with_correlation(window_preprocessed)

                    # Get top 2 correlations
                    sorted_indices = np.argsort(correlations)[::-1]
                    top1_idx = sorted_indices[0]
                    top2_idx = sorted_indices[1]

                    top1_corr = correlations[top1_idx]
                    top2_corr = correlations[top2_idx]

                    margin = top1_corr - top2_corr

                    # Apply thresholds
                    if top1_corr >= self.confidence_threshold and margin >= self.margin_threshold:
                        prediction = top1_idx
                        freq = self.config.target_frequencies[prediction]

                        # Update feedback if prediction changed
                        if prediction != self.last_prediction:
                            print(f"[{window_count:4d}] Prediction: {freq:.2f} Hz "
                                  f"(corr={top1_corr:.3f}, margin={margin:.3f})")

                            if self.arduino.is_connected:
                                self.arduino.show_feedback(freq)

                            self.last_prediction = prediction
                    else:
                        # No confident prediction
                        if self.last_prediction is not None:
                            print(f"[{window_count:4d}] No confident prediction "
                                  f"(corr={top1_corr:.3f}, margin={margin:.3f})")

                            if self.arduino.is_connected:
                                self.arduino.clear_feedback()

                            self.last_prediction = None

                    self.last_correlations = correlations

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")

        # Stop
        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()

        self.eeg_driver.stop_stream()
        self.is_running = False

        return True


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP BCI with TRCA Classification"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to trained TRCA model (.pkl file)"
    )
    parser.add_argument(
        "--arduino",
        type=str,
        default=None,
        help="Arduino serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "--cyton",
        type=str,
        default=None,
        help="OpenBCI Cyton serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic EEG data for testing"
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=250,
        help="Analysis window duration in milliseconds (default: 250)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="Confidence threshold (default: 0.05)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Margin threshold (default: 0.02)"
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    # Create runner
    runner = BCIRunnerTRCA(
        model_path=args.model,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        window_ms=args.window_ms,
        confidence_threshold=args.confidence,
        margin_threshold=args.margin
    )

    # Connect
    if not runner.connect():
        print("\nERROR: Failed to connect to hardware")
        return 1

    # Run
    try:
        runner.run()
    finally:
        runner.disconnect()

    print("\nBCI session complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
