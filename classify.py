#!/usr/bin/env python3
"""
SSVEP Real-Time BCI - Online Inference with TRCA

Real-time classification using trained TRCA model.
Applies causal preprocessing and provides Arduino feedback.

Usage:
    python run_realtime.py models/alice_trca_20250115_143022.pkl
    python run_realtime.py models/model.pkl --arduino COM3 --cyton COM4
    python run_realtime.py models/model.pkl --window-ms 500
"""

import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import OnlinePreprocessor
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController, SyntheticSSVEPDriver
from ssvep_bci.buffer import EEGBuffer
from ssvep_bci.trca import TRCA


class RealtimeBCI:
    """Real-time SSVEP BCI with TRCA classification."""

    def __init__(self, model_path: str, arduino_port=None, cyton_port=None,
                 use_synthetic=False, window_ms=250):
        # Load TRCA model
        print(f"Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.trca: TRCA = model_data['trca']
        self.config: SSVEPConfig = model_data['config']

        print(f"  Subject: {model_data.get('subject', 'unknown')}")
        print(f"  Test accuracy: {model_data.get('test_accuracy', 0)*100:.1f}%")

        # Update window size
        self.window_samples = int(window_ms * self.config.fs / 1000)
        print(f"  Analysis window: {window_ms} ms ({self.window_samples} samples)")

        # Initialize components
        self.preprocessor = OnlinePreprocessor(self.config)
        self.buffer = EEGBuffer(self.config)

        # Arduino
        self.arduino = ArduinoController(port=arduino_port)

        # EEG
        if use_synthetic:
            self.eeg = SyntheticSSVEPDriver(self.config, target_frequency=10.0)
        else:
            self.eeg = BrainFlowDriver(self.config)
            self.eeg.config.serial_port = cyton_port

        self.is_running = False
        self.last_prediction = None

    def connect(self) -> bool:
        """Connect to hardware."""
        print("\n" + "=" * 60)
        print("SSVEP Real-Time BCI")
        print("=" * 60)

        # Arduino
        print("\n[1/2] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  WARNING: Arduino not connected")
        else:
            print(f"  Connected: {self.arduino.port}")

        # EEG
        mode = "Synthetic" if isinstance(self.eeg, SyntheticSSVEPDriver) else "OpenBCI Cyton"
        print(f"\n[2/2] Connecting to {mode}...")
        if not self.eeg.connect():
            print("  ERROR: Failed to connect to EEG")
            return False
        print(f"  Connected! Sampling rate: {self.eeg.sampling_rate} Hz")

        return True

    def disconnect(self):
        """Disconnect hardware."""
        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()
            self.arduino.disconnect()
        self.eeg.disconnect()

    def run(self):
        """Run real-time BCI loop."""
        print("\n" + "=" * 60)
        print("Real-Time Classification")
        print("=" * 60)

        print("\nFrequencies:")
        for i, freq in enumerate(self.config.target_frequencies):
            print(f"  {i}: {freq:.2f} Hz")

        print("\nStarting in 3 seconds...")
        time.sleep(3)

        # Start EEG
        if not self.eeg.start_stream():
            print("ERROR: Failed to start EEG stream")
            return False

        self.is_running = True
        self.buffer.reset()
        self.preprocessor.reset()

        # Start LEDs
        if self.arduino.is_connected:
            self.arduino.start_stimulation()

        print("\nBCI RUNNING! Press Ctrl+C to stop.\n")

        try:
            window_count = 0

            while self.is_running:
                # Check emergency stop
                if self.arduino.is_connected and self.arduino.check_button_pressed():
                    print("\n[EMERGENCY STOP] Button pressed!")
                    break

                # Get new data
                data = self.eeg.get_data()
                if data is not None and data.shape[1] > 0:
                    self.buffer.add_data(data)

                # Process when ready
                if self.buffer.get_num_samples() >= self.window_samples:
                    window_count += 1

                    # Get window
                    window = self.buffer.get_window(self.window_samples)

                    # Preprocess (causal, online)
                    window_preprocessed = self.preprocessor.process(window)

                    # Classify with TRCA
                    pred_idx, correlations = self.trca.predict_with_correlation(window_preprocessed)

                    # Get top 2
                    sorted_idx = np.argsort(correlations)[::-1]
                    top1_idx = sorted_idx[0]
                    top2_idx = sorted_idx[1]

                    top1_corr = correlations[top1_idx]
                    top2_corr = correlations[top2_idx]
                    margin = top1_corr - top2_corr

                    # Thresholds (TRCA has lower correlations than CCA)
                    conf_thresh = 0.03
                    margin_thresh = 0.01

                    if top1_corr >= conf_thresh and margin >= margin_thresh:
                        freq = self.config.target_frequencies[top1_idx]

                        if top1_idx != self.last_prediction:
                            print(f"[{window_count:4d}] Prediction: {freq:.2f} Hz "
                                  f"(corr={top1_corr:.3f}, margin={margin:.3f})")

                            if self.arduino.is_connected:
                                self.arduino.show_feedback(freq)

                            self.last_prediction = top1_idx
                    else:
                        if self.last_prediction is not None:
                            if self.arduino.is_connected:
                                self.arduino.clear_feedback()
                            self.last_prediction = None

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        # Cleanup
        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()

        self.eeg.stop_stream()
        self.is_running = False

        return True


def main():
    parser = argparse.ArgumentParser(description="SSVEP Real-Time BCI")
    parser.add_argument("model", type=str,
                       help="Path to trained TRCA model (.pkl file)")
    parser.add_argument("--arduino", type=str, default=None,
                       help="Arduino serial port (auto-detect if not specified)")
    parser.add_argument("--cyton", type=str, default=None,
                       help="Cyton serial port (auto-detect if not specified)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic EEG for testing")
    parser.add_argument("--window-ms", type=int, default=250,
                       help="Analysis window duration in ms (default: 250)")

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    # Create BCI
    bci = RealtimeBCI(
        model_path=args.model,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        window_ms=args.window_ms
    )

    # Connect
    if not bci.connect():
        print("\nERROR: Failed to connect to hardware")
        return 1

    # Run
    try:
        bci.run()
    finally:
        bci.disconnect()

    print("\nBCI session complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
