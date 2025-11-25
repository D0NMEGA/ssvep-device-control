#!/usr/bin/env python3
"""
SSVEP Calibration - Data Collection & TRCA Training

Unified script that:
1. Records calibration data with LSL markers (XDF format)
2. Trains TRCA model from collected data
3. Saves trained model for real-time use

Usage:
    python run_calibration.py --subject alice
    python run_calibration.py --subject bob --trials 3 --baseline 30
    python run_calibration.py --arduino COM3 --cyton COM4
"""

import sys
import time
import argparse
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import (XDFWriter, LSLMarkerSender, OfflinePreprocessor,
                         extract_trials_from_xdf)
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController, SyntheticSSVEPDriver
from ssvep_bci.trca import TRCA


def run_calibration(subject: str, session: int, config: SSVEPConfig,
                   arduino_port=None, cyton_port=None, use_synthetic=False,
                   trials_per_freq=5, trial_duration=4.0, rest_duration=2.0,
                   baseline_duration=30.0):
    """Run calibration data collection.

    Returns:
        Path to XDF file if successful, None otherwise
    """
    print("=" * 60)
    print("SSVEP Calibration - Data Collection")
    print("=" * 60)

    # Connect Arduino
    arduino = ArduinoController(port=arduino_port)
    print("\n[1/2] Connecting to Arduino...")
    if not arduino.connect():
        print("  WARNING: Arduino not connected")
    else:
        print(f"  Connected: {arduino.port}")

    # Connect EEG
    if use_synthetic:
        eeg = SyntheticSSVEPDriver(config, target_frequency=10.0)
        mode = "Synthetic"
    else:
        eeg = BrainFlowDriver(config)
        eeg.config.serial_port = cyton_port
        mode = "OpenBCI Cyton"

    print(f"\n[2/2] Connecting to {mode}...")
    if not eeg.connect():
        print("  ERROR: Failed to connect to EEG")
        return None
    print(f"  Connected! Sampling rate: {eeg.sampling_rate} Hz")

    # Create XDF writer and marker sender
    xdf_writer = XDFWriter(subject, session, config)
    marker_sender = LSLMarkerSender()

    # Instructions
    print("\n" + "=" * 60)
    print("IMPORTANT: Start LabRecorder NOW")
    print("=" * 60)
    print("1. Open LabRecorder")
    print("2. Check for 'SSVEP-EEG' and 'SSVEP-Markers' streams")
    print("3. Click 'Start' to begin recording")
    print("4. Press Enter here when ready")
    input("\nPress Enter when LabRecorder is recording...")

    # Start streaming
    if not eeg.start_stream():
        print("ERROR: Failed to start EEG stream")
        return None

    # Send calibration start
    marker_sender.send_start_calibration(subject=subject, session=session)

    try:
        # Baseline period
        if baseline_duration > 0:
            print(f"\n[Baseline] {baseline_duration}s - Eyes open, relaxed")
            marker_sender.send_baseline_start(duration_sec=baseline_duration)

            start_t = time.time()
            while time.time() - start_t < baseline_duration:
                data = eeg.get_data()
                if data is not None:
                    for i in range(data.shape[1]):
                        xdf_writer.push_eeg(data[:, i])
                time.sleep(0.01)

            marker_sender.send_baseline_end()

        # Generate trial sequence (randomized)
        frequencies = [8.57, 10.0, 12.0, 15.0]
        trials = []
        for freq in frequencies:
            trials.extend([(freq, i+1) for i in range(trials_per_freq)])
        np.random.shuffle(trials)

        total_trials = len(trials)

        # LED labels
        led_labels = {8.57: "LED 1 (far left)", 10.0: "LED 2 (center-left)",
                     12.0: "LED 3 (center-right)", 15.0: "LED 4 (far right)"}

        # Trial loop
        for trial_idx, (freq, _) in enumerate(trials):
            print(f"\n[Trial {trial_idx+1}/{total_trials}] {freq:.2f} Hz - {led_labels[freq]}")

            # Send markers
            marker_sender.send_trial_start(trial=trial_idx+1, target_freq=freq)

            # Visual cue (red LED)
            if arduino.is_connected:
                arduino.clear_feedback()
                time.sleep(0.1)
                arduino.show_feedback(freq)
                time.sleep(0.3)
                arduino.start_stimulation()

            # Stim on
            marker_sender.send_stim_on(freq=freq, trial=trial_idx+1)

            # Record trial
            start_t = time.time()
            while time.time() - start_t < trial_duration:
                data = eeg.get_data()
                if data is not None:
                    for i in range(data.shape[1]):
                        xdf_writer.push_eeg(data[:, i])
                time.sleep(0.01)

            # Stim off
            marker_sender.send_stim_off(freq=freq, trial=trial_idx+1)

            if arduino.is_connected:
                arduino.stop_stimulation()
                arduino.clear_feedback()

            # Rest
            if trial_idx < total_trials - 1:
                print("  Rest...")
                marker_sender.send_rest_start(trial=trial_idx+1)

                start_t = time.time()
                while time.time() - start_t < rest_duration:
                    data = eeg.get_data()
                    if data is not None:
                        for i in range(data.shape[1]):
                            xdf_writer.push_eeg(data[:, i])
                    time.sleep(0.01)

                marker_sender.send_rest_end(trial=trial_idx+1)

        # End calibration
        marker_sender.send_end_calibration(n_trials=total_trials)

        print("\n" + "=" * 60)
        print("Data Collection Complete!")
        print("=" * 60)
        print("Stop LabRecorder and save the XDF file.")

        # Get XDF path from user
        print("\nEnter the path to the saved XDF file:")
        xdf_path = input("> ").strip()

        return Path(xdf_path) if xdf_path else None

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return None

    finally:
        eeg.stop_stream()
        eeg.disconnect()
        if arduino.is_connected:
            arduino.disconnect()


def train_trca_model(xdf_path: Path, subject: str, config: SSVEPConfig):
    """Train TRCA model from XDF file.

    Returns:
        Path to saved model file
    """
    print("\n" + "=" * 60)
    print("TRCA Model Training")
    print("=" * 60)

    # Extract trials
    print(f"\nLoading {xdf_path}...")
    X, y = extract_trials_from_xdf(str(xdf_path), config,
                                   trial_duration_sec=4.0, baseline_sec=0.5)

    print(f"  Extracted {len(X)} trials")
    print(f"  Shape: {X.shape}")

    # Preprocess (offline, zero-phase)
    print("\nPreprocessing (7-90 Hz bandpass + 60 Hz notch, zero-phase)...")
    preprocessor = OfflinePreprocessor(config)
    X_preprocessed = preprocessor.process(X)

    # Train/test split (80/20)
    n_total = len(X_preprocessed)
    n_test = int(n_total * 0.2)
    indices = np.random.permutation(n_total)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X_preprocessed[train_idx], y[train_idx]
    X_test, y_test = X_preprocessed[test_idx], y[test_idx]

    print(f"  Training: {len(X_train)} trials")
    print(f"  Testing: {len(X_test)} trials")

    # Train TRCA
    print("\nTraining TRCA...")
    trca = TRCA(config, n_components=1)
    trca.fit(X_train, y_train)

    # Evaluate
    y_pred_train, _ = trca.predict_with_correlation(X_train)
    y_pred_test, _ = trca.predict_with_correlation(X_test)

    acc_train = np.mean(y_pred_train == y_train)
    acc_test = np.mean(y_pred_test == y_test)

    print(f"\n  Training accuracy: {acc_train*100:.1f}%")
    print(f"  Test accuracy: {acc_test*100:.1f}%")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{subject}_trca_{timestamp}.pkl"

    model_data = {
        'trca': trca,
        'config': config,
        'train_accuracy': acc_train,
        'test_accuracy': acc_test,
        'subject': subject,
        'timestamp': datetime.now().isoformat()
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="SSVEP Calibration & Training")
    parser.add_argument("--subject", type=str, default="default",
                       help="Subject ID (e.g., alice)")
    parser.add_argument("--session", type=int, default=1,
                       help="Session number")
    parser.add_argument("--arduino", type=str, default=None,
                       help="Arduino serial port (auto-detect if not specified)")
    parser.add_argument("--cyton", type=str, default=None,
                       help="Cyton serial port (auto-detect if not specified)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic EEG for testing")
    parser.add_argument("--trials", type=int, default=5,
                       help="Trials per frequency (default: 5)")
    parser.add_argument("--baseline", type=float, default=30.0,
                       help="Baseline duration in seconds (default: 30, 0 to skip)")

    args = parser.parse_args()

    config = SSVEPConfig()

    # Run calibration
    xdf_path = run_calibration(
        subject=args.subject,
        session=args.session,
        config=config,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        trials_per_freq=args.trials,
        baseline_duration=args.baseline
    )

    if not xdf_path or not Path(xdf_path).exists():
        print("\nCalibration failed or XDF file not found")
        return 1

    # Train model
    model_path = train_trca_model(xdf_path, args.subject, config)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"\nRun real-time BCI:")
    print(f"  python run_realtime.py {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
