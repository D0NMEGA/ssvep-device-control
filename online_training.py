#!/usr/bin/env python3
"""
SSVEP Online Training - Adaptive Learning with User Feedback

Intermediate system between calibration and classify.
The system guesses which LED you're looking at, you provide feedback,
and the model improves adaptively.

Usage:
    # Start from scratch (20 trials default)
    python online_training.py

    # Start with existing model
    python online_training.py --model models/alice_trca.pkl

    # Custom number of trials
    python online_training.py --max-trials 30

    # Unlimited trials (until 'q' pressed)
    python online_training.py --max-trials 0

    # Use synthetic data for testing
    python online_training.py --synthetic

Controls:
    - Look at any LED for 4-5 seconds
    - After prediction, press:
        1-4: Correct LED number (if prediction wrong)
        ENTER: Prediction was correct
        'q': Quit and save model
    - Session ends automatically after max trials (default: 20)
    - 2-second rest period between trials
"""

import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import msvcrt  # For Windows keyboard input

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import OnlinePreprocessor
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController, SyntheticSSVEPDriver
from ssvep_bci.buffer import EEGBuffer
from ssvep_bci.trca import TRCA


class OnlineTrainingBCI:
    """Online SSVEP training with adaptive learning."""

    def __init__(self, model_path=None, arduino_port=None, cyton_port=None,
                 use_synthetic=False, subject_name="user", max_trials=20):
        self.config = SSVEPConfig()
        self.subject_name = subject_name
        self.max_trials = max_trials

        # Load existing model or create new one
        if model_path and Path(model_path).exists():
            print(f"Loading existing model: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.trca = model_data['trca']
            self.config = model_data['config']

            # Update window_samples to match model's actual template size
            if self.trca.is_fitted and self.trca.templates and len(self.trca.templates) > 0:
                # templates is a list, get first template's shape
                template_samples = self.trca.templates[0].shape[1]  # (n_channels, n_samples, n_filterbanks)
                if template_samples != self.config.window_samples:
                    print(f"  Updating window_samples: {self.config.window_samples} -> {template_samples} (from model)")
                    self.config.window_samples = template_samples

            # Load training data if available
            if 'training_data' in model_data:
                self.X_train = model_data['training_data']['X']
                self.y_train = model_data['training_data']['y']

                # Convert numpy arrays to lists (for append operations)
                if isinstance(self.X_train, np.ndarray):
                    self.X_train = list(self.X_train)
                if isinstance(self.y_train, np.ndarray):
                    self.y_train = list(self.y_train)

                print(f"  Loaded {len(self.X_train)} training trials")
            else:
                self.X_train = []
                self.y_train = []
                print("  No training data in model, starting fresh")
        else:
            print("Starting with no model - will train after first trial")
            self.trca = TRCA(self.config, n_components=1)
            self.X_train = []
            self.y_train = []

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

        # Calculate required trial duration from window_samples
        self.trial_duration = self.config.window_samples / self.config.fs + 0.5  # Add 0.5s buffer
        print(f"  Trial duration: {self.trial_duration:.1f}s ({self.config.window_samples} samples)")

        # Statistics
        self.trial_count = 0
        self.correct_predictions = 0
        self.all_predictions = []  # For confusion matrix
        self.all_true_labels = []  # For confusion matrix

    def connect(self) -> bool:
        """Connect to hardware."""
        print("\n" + "=" * 60)
        print("SSVEP Online Training System")
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

    def get_keyboard_input(self, timeout=10.0):
        """Get keyboard input with timeout (Windows)."""
        print(f"\nWas prediction correct?")
        print("  Press ENTER if correct")
        print("  Press 1-4 for actual LED number")
        print("  Press 'q' to quit")
        print(f"  ({timeout:.0f}s timeout, will assume correct if no input)")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                key = msvcrt.getch()

                # Handle different key formats
                if key == b'\r':  # Enter key
                    return 'correct'
                elif key == b'q':
                    return 'quit'
                elif key in [b'1', b'2', b'3', b'4']:
                    led_num = int(key.decode())
                    # Map LED 1-4 to class indices
                    # target_frequencies = (15, 12, 10, 8.57) → class indices (0, 1, 2, 3)
                    # LED 1=8.57→class 3, LED 2=10→class 2, LED 3=12→class 1, LED 4=15→class 0
                    led_to_class = {1: 3, 2: 2, 3: 1, 4: 0}
                    return led_to_class[led_num]

            time.sleep(0.1)

        # Timeout - assume correct
        print("  (No input - assuming correct)")
        return 'correct'

    def record_trial(self, duration=4.0):
        """Record one trial of EEG data."""
        print(f"\n[Trial {self.trial_count + 1}] Recording {duration}s...")

        # Accumulate data in a list (don't use ring buffer for long recordings)
        collected_data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            # Check emergency stop
            if self.arduino.is_connected and self.arduino.check_button_pressed():
                print("\n[EMERGENCY STOP] Button pressed!")
                return None

            # Get new data
            data = self.eeg.get_data()
            if data is not None and data.shape[1] > 0:
                collected_data.append(data)

            time.sleep(0.01)

        # Concatenate all collected data
        if not collected_data:
            print("  ERROR: No data collected")
            return None

        trial_data = np.hstack(collected_data)

        if trial_data.shape[1] < self.config.window_samples:
            print(f"  ERROR: Insufficient data collected ({trial_data.shape[1]} < {self.config.window_samples})")
            return None

        # Trim to window size
        trial_data = trial_data[:, :self.config.window_samples]

        print(f"  Collected {trial_data.shape[1]} samples ({trial_data.shape[1]/self.config.fs:.1f}s)")
        return trial_data

    def predict_trial(self, trial_data):
        """Make prediction on trial data."""
        if not self.trca.is_fitted:
            print("  Model not trained yet - skipping prediction")
            return None, None

        # Preprocess
        trial_preprocessed = self.preprocessor.process(trial_data)

        # Predict
        pred_idx, correlations = self.trca.predict_with_correlation(trial_preprocessed)
        pred_freq = self.config.target_frequencies[pred_idx]
        max_corr = correlations[pred_idx]

        # Map frequency to LED number (1-4): LED 1=8.57, 2=10, 3=12, 4=15
        led_map = {8.57: 1, 10.0: 2, 12.0: 3, 15.0: 4}
        led_num = led_map.get(round(pred_freq, 2), pred_idx)

        print(f"  Prediction: LED {led_num} ({pred_freq:.2f} Hz, correlation={max_corr:.3f})")

        # Show feedback on Arduino
        if self.arduino.is_connected:
            self.arduino.show_feedback(pred_freq)

        return pred_idx, correlations

    def add_training_sample(self, trial_data, label):
        """Add trial to training set (stores RAW data, not preprocessed)."""
        # Store RAW trial data (we'll preprocess during training)
        if len(self.X_train) == 0:
            self.X_train = [trial_data]
            self.y_train = [label]
        else:
            self.X_train.append(trial_data)
            self.y_train.append(label)

        print(f"  Added trial as class {label} ({self.config.target_frequencies[label]:.2f} Hz)")
        print(f"  Training set: {len(self.X_train)} trials")

        # Count trials per class
        class_counts = {i: 0 for i in range(len(self.config.target_frequencies))}
        for y in self.y_train:
            class_counts[y] += 1

        print(f"  Per-class counts: {class_counts}")

    def retrain_model(self):
        """Retrain TRCA model with current training data."""
        if len(self.X_train) < 4:
            print("  Need at least 4 trials to train model (skipping)")
            return False

        print("\n  Retraining model...")

        # Check if we have at least 2 classes
        y_array = np.array(self.y_train)
        unique_classes = np.unique(y_array)
        if len(unique_classes) < 2:
            print(f"  Only {len(unique_classes)} class(es) in training data - need at least 2")
            return False

        # Preprocess all raw trials with filterbank
        X_preprocessed = []
        for trial in self.X_train:
            # Check if already preprocessed (has 3 dimensions with filterbanks)
            if trial.ndim == 3:
                X_preprocessed.append(trial)
            else:
                # Raw data - preprocess it
                trial_proc = self.preprocessor.process(trial)
                X_preprocessed.append(trial_proc)

        X_array = np.array(X_preprocessed)

        # Train
        self.trca = TRCA(self.config, n_components=1)
        self.trca.fit(X_array, y_array)

        print(f"  Model retrained with {len(X_array)} trials")
        return True

    def save_model(self):
        """Save model and training data."""
        if not self.trca.is_fitted:
            print("Model not trained - nothing to save")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/{self.subject_name}_online_{timestamp}.pkl"

        Path("models").mkdir(exist_ok=True)

        model_data = {
            'trca': self.trca,
            'config': self.config,
            'subject': self.subject_name,
            'timestamp': timestamp,
            'trial_count': self.trial_count,
            'training_data': {
                'X': self.X_train,
                'y': self.y_train
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved: {filename}")
        return filename

    def print_diagnostics(self):
        """Print end-of-session diagnostics."""
        print("\n" + "=" * 60)
        print("SESSION DIAGNOSTICS")
        print("=" * 60)

        # Overall statistics
        print(f"\nTotal trials: {self.trial_count}")
        if self.trial_count > 0:
            accuracy = (self.correct_predictions / self.trial_count) * 100
            print(f"Overall accuracy: {accuracy:.1f}% ({self.correct_predictions}/{self.trial_count})")

        # Per-class trial counts
        print("\nTrials per class:")
        class_counts = {i: 0 for i in range(len(self.config.target_frequencies))}
        for y in self.y_train:
            class_counts[y] += 1

        led_map = {0: 4, 1: 3, 2: 2, 3: 1}  # Class index to LED number
        for class_i in range(len(self.config.target_frequencies)):
            freq = self.config.target_frequencies[class_i]
            led_num = led_map[class_i]
            count = class_counts[class_i]
            print(f"  LED {led_num} ({freq:.2f} Hz): {count} trials")

        # Confusion matrix (if we have predictions)
        if len(self.all_predictions) > 0 and len(self.all_true_labels) > 0:
            print("\nConfusion Matrix:")
            n_classes = len(self.config.target_frequencies)
            confusion = np.zeros((n_classes, n_classes), dtype=int)

            for pred, true in zip(self.all_predictions, self.all_true_labels):
                confusion[true, pred] += 1

            # Print header (LED numbers)
            print("        ", end="")
            for i in range(n_classes):
                led_num = led_map[i]
                print(f"LED{led_num}", end="  ")
            print(" (Predicted)")

            # Print rows
            for true_i in range(n_classes):
                led_num = led_map[true_i]
                print(f"  LED{led_num} ", end="")
                for pred_i in range(n_classes):
                    print(f" {confusion[true_i, pred_i]:3d} ", end=" ")
                print()

            print("(Actual)")

            # Per-class accuracy
            print("\nPer-class accuracy:")
            for class_i in range(n_classes):
                led_num = led_map[class_i]
                freq = self.config.target_frequencies[class_i]
                total = confusion[class_i, :].sum()
                if total > 0:
                    correct = confusion[class_i, class_i]
                    acc = (correct / total) * 100
                    print(f"  LED {led_num} ({freq:.2f} Hz): {acc:.1f}% ({correct}/{total})")
                else:
                    print(f"  LED {led_num} ({freq:.2f} Hz): N/A (no trials)")

        print("\n" + "=" * 60)

    def run(self):
        """Run online training loop."""
        print("\n" + "=" * 60)
        print("Online Training Mode")
        print("=" * 60)

        print("\nFrequencies:")
        # Map to LED 1-4 (ascending): LED 1=8.57, 2=10, 3=12, 4=15
        led_map = {8.57: 1, 10.0: 2, 12.0: 3, 15.0: 4}
        for freq in sorted(self.config.target_frequencies):
            led_num = led_map.get(round(freq, 2), "?")
            print(f"  LED {led_num}: {freq:.2f} Hz")

        print("\nInstructions:")
        print(f"  1. Look at any LED for {self.trial_duration:.1f} seconds")
        print("  2. System will predict which LED")
        print("  3. You provide feedback to correct/confirm")
        print("  4. Model improves over time!")

        if self.max_trials is not None:
            print(f"\nSession: {self.max_trials} trials (press 'q' to quit early)")
        else:
            print(f"\nSession: Unlimited trials (press 'q' to quit and save)")

        print("\nStarting in 3 seconds...")
        time.sleep(3)

        # Start EEG
        if not self.eeg.start_stream():
            print("ERROR: Failed to start EEG stream")
            return False

        self.is_running = True
        self.preprocessor.reset()

        # Start LEDs
        if self.arduino.is_connected:
            self.arduino.start_stimulation()

        print("\n" + "=" * 60)
        print("ONLINE TRAINING ACTIVE!")
        print("=" * 60)

        # Track whether exit was intentional (for saving)
        intentional_quit = False

        try:
            while self.is_running:
                # Check if max trials reached
                if self.max_trials is not None and self.trial_count >= self.max_trials:
                    print(f"\n[Session Complete] Reached {self.max_trials} trials")
                    intentional_quit = True
                    break

                # Record trial with calculated duration
                trial_data = self.record_trial(duration=self.trial_duration)

                if trial_data is None:
                    print("Emergency stop or recording failed")
                    break

                # Make prediction if model is trained
                pred_idx, correlations = self.predict_trial(trial_data)

                # Get user feedback
                feedback = self.get_keyboard_input(timeout=10.0)

                if feedback == 'quit':
                    print("\nQuitting...")
                    intentional_quit = True
                    break

                # Determine true label
                if feedback == 'correct':
                    if pred_idx is not None:
                        true_label = pred_idx
                        self.correct_predictions += 1
                        print(f"  ✓ Prediction confirmed correct!")
                    else:
                        print("  No prediction was made, skipping trial")
                        continue
                elif isinstance(feedback, int):
                    true_label = feedback  # This is already mapped to class index
                    # Map class index back to LED number for display
                    # Class 0=15Hz→LED4, Class 1=12Hz→LED3, Class 2=10Hz→LED2, Class 3=8.57Hz→LED1
                    class_to_led = {0: 4, 1: 3, 2: 2, 3: 1}
                    led_num = class_to_led[true_label]

                    # Check if user input matches prediction (manual reinforcement vs correction)
                    if pred_idx is not None and true_label == pred_idx:
                        self.correct_predictions += 1
                        print(f"  ✓ Manually confirmed LED {led_num} ({self.config.target_frequencies[true_label]:.2f} Hz)")
                    else:
                        print(f"  ✗ Corrected to LED {led_num} ({self.config.target_frequencies[true_label]:.2f} Hz)")
                else:
                    print("  Invalid input, skipping trial")
                    continue

                # Clear Arduino feedback
                if self.arduino.is_connected:
                    self.arduino.clear_feedback()

                # Add to training data
                self.add_training_sample(trial_data, true_label)
                self.trial_count += 1

                # Track predictions for confusion matrix
                if pred_idx is not None:
                    self.all_predictions.append(pred_idx)
                    self.all_true_labels.append(true_label)

                # Retrain model
                self.retrain_model()

                # Show statistics
                if self.trial_count > 0 and self.trca.is_fitted:
                    accuracy = (self.correct_predictions / self.trial_count) * 100
                    print(f"\n  Statistics: {self.correct_predictions}/{self.trial_count} correct ({accuracy:.1f}%)")

                # Rest period between trials
                print("\n" + "-" * 60)
                if self.max_trials is None or self.trial_count < self.max_trials:
                    print("Rest period: 2 seconds...")
                    time.sleep(2)

        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C)")
            intentional_quit = False

        # Cleanup
        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()

        self.eeg.stop_stream()
        self.is_running = False

        # Print diagnostics if we completed any trials
        if self.trial_count > 0:
            self.print_diagnostics()

        # Save model only if quit intentionally with 'q' or max trials reached
        if self.trial_count > 0 and intentional_quit:
            self.save_model()
        elif self.trial_count > 0 and not intentional_quit:
            print("\nModel not saved (use 'q' to quit and save)")

        return True


def main():
    parser = argparse.ArgumentParser(description="SSVEP Online Training System")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to existing model to improve (optional)")
    parser.add_argument("--arduino", type=str, default=None,
                       help="Arduino serial port (auto-detect if not specified)")
    parser.add_argument("--cyton", type=str, default=None,
                       help="Cyton serial port (auto-detect if not specified)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic EEG for testing")
    parser.add_argument("--subject", type=str, default="user",
                       help="Subject name for saved model")
    parser.add_argument("--max-trials", type=int, default=20,
                       help="Maximum number of trials (default: 20, use 0 for unlimited)")

    args = parser.parse_args()

    # Handle unlimited trials
    max_trials = args.max_trials if args.max_trials > 0 else None

    # Create BCI
    bci = OnlineTrainingBCI(
        model_path=args.model,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        subject_name=args.subject,
        max_trials=max_trials
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

    print("\nOnline training session complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
