#!/usr/bin/env python3
"""
SSVEP Offline Training - Data Collection & TRCA Training

Unified script for structured offline training that:
1. Records calibration data with LSL markers (XDF format)
2. Trains TRCA model from collected data (batch learning)
3. Saves trained model for real-time use

Features:
- Auto-detects session number (increments from previous sessions)
- Batch learning: Combines all previous sessions for better accuracy
- LOOCV evaluation for model validation

Usage:
    python offline_training.py --subject alice
    python offline_training.py --subject bob --trials 3 --baseline 30
    python offline_training.py --arduino COM3 --cyton COM4
    python offline_training.py --subject alice --session 5  # Manual override
"""

import sys
import time
import signal
import argparse
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import (XDFWriter, OfflinePreprocessor,
                         extract_trials_from_xdf)
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController, SyntheticSSVEPDriver
from ssvep_bci.trca import TRCA


# =============================================================================
# Signal Handlers for Clean Shutdown
# =============================================================================

# Global reference for cleanup handlers
_cleanup_handlers = []

def register_cleanup(handler):
    """Register cleanup function to run on exit/interrupt."""
    _cleanup_handlers.append(handler)

def signal_handler(sig, frame):
    """Handle Ctrl+C and kill signals gracefully."""
    print("\n\n" + "="*60)
    print("Shutdown signal received. Cleaning up LSL streams...")
    print("="*60)

    # Run all registered cleanup handlers
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Cleanup error: {e}")

    print("✓ Cleanup complete. Exiting.")
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # External kill


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

    # Create XDF writer (includes marker sender)
    xdf_writer = XDFWriter(subject, session, config)
    marker_sender = xdf_writer.marker_sender  # Use XDFWriter's marker sender

    # Register cleanup for signal handler (Ctrl+C graceful shutdown)
    def cleanup_lsl():
        xdf_writer.close()
    register_cleanup(cleanup_lsl)

    # Instructions
    print("\n" + "=" * 60)
    print("IMPORTANT: Start LabRecorder NOW")
    print("=" * 60)
    print("1. Open LabRecorder")
    print("2. Click 'Update' to refresh streams")
    print("3. Select BOTH streams:")
    print("   - SSVEP-EEG (EEG, 8 channels, 250 Hz)")
    print("   - SSVEP-Markers (Markers)")
    print("4. Click 'Start' to begin recording")
    print("5. Press Enter here when ready")
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

        # LED labels and indices
        led_labels = {8.57: "LED 1 (far left)", 10.0: "LED 2 (center-left)",
                     12.0: "LED 3 (center-right)", 15.0: "LED 4 (far right)"}
        led_indices = {8.57: 1, 10.0: 2, 12.0: 3, 15.0: 4}

        # Trial loop
        global_trial_counter = 0
        emergency_stop = False

        for trial_idx, (freq, _) in enumerate(trials):
            # Check for emergency stop button
            if arduino.is_connected and arduino.check_button_pressed():
                print("\n" + "="*60)
                print("[EMERGENCY STOP] Button pressed!")
                print("="*60)
                print("Calibration aborted. Data will NOT be saved.")
                emergency_stop = True
                break

            global_trial_counter += 1
            led_idx = led_indices[freq]

            print(f"\n[Trial {trial_idx+1}/{total_trials}] {freq:.2f} Hz - {led_labels[freq]}")

            # Send markers with global trial ID and LED index
            marker_sender.send_trial_start(
                trial=trial_idx+1,
                target_freq=freq,
                global_trial_id=global_trial_counter,
                led_index=led_idx
            )

            # Visual cue (red LED)
            if arduino.is_connected:
                arduino.clear_feedback()
                time.sleep(0.1)
                arduino.show_feedback(freq)
                time.sleep(0.3)
                arduino.start_stimulation()

            # Stim on (accurate onset for TRCA extraction)
            marker_sender.send_stim_on(
                freq=freq,
                trial=trial_idx+1,
                global_trial_id=global_trial_counter,
                led_index=led_idx
            )

            # Record trial
            start_t = time.time()
            while time.time() - start_t < trial_duration:
                # Check emergency stop during trial
                if arduino.is_connected and arduino.check_button_pressed():
                    print("\n" + "="*60)
                    print("[EMERGENCY STOP] Button pressed!")
                    print("="*60)
                    print("Calibration aborted. Data will NOT be saved.")
                    emergency_stop = True
                    break

                data = eeg.get_data()
                if data is not None:
                    for i in range(data.shape[1]):
                        xdf_writer.push_eeg(data[:, i])
                time.sleep(0.01)

            # Break outer loop if emergency stop triggered
            if emergency_stop:
                break

            # Stim off (stimulation offset)
            marker_sender.send_stim_off(
                freq=freq,
                trial=trial_idx+1,
                global_trial_id=global_trial_counter,
                led_index=led_idx
            )

            if arduino.is_connected:
                arduino.stop_stimulation()
                arduino.clear_feedback()

            # Rest
            if trial_idx < total_trials - 1 and not emergency_stop:
                print("  Rest...")
                marker_sender.send_rest_start(trial=trial_idx+1)

                start_t = time.time()
                while time.time() - start_t < rest_duration:
                    # Check emergency stop during rest
                    if arduino.is_connected and arduino.check_button_pressed():
                        print("\n" + "="*60)
                        print("[EMERGENCY STOP] Button pressed!")
                        print("="*60)
                        print("Calibration aborted. Data will NOT be saved.")
                        emergency_stop = True
                        break

                    data = eeg.get_data()
                    if data is not None:
                        for i in range(data.shape[1]):
                            xdf_writer.push_eeg(data[:, i])
                    time.sleep(0.01)

                if not emergency_stop:
                    marker_sender.send_rest_end(trial=trial_idx+1)

        # If emergency stop was triggered, abort without saving
        if emergency_stop:
            print("\nEmergency stop activated - exiting without saving data.")
            return None

        # End calibration
        marker_sender.send_end_calibration(n_trials=total_trials)

        print("\n" + "=" * 60)
        print("Data Collection Complete!")
        print("=" * 60)
        print("Stop LabRecorder and save the XDF file.")

        # Auto-generate expected XDF path
        calib_dir = Path("calibration")
        calib_dir.mkdir(exist_ok=True)

        expected_xdf = calib_dir / f"{subject}_S{session:03d}_eeg.xdf"

        print(f"\nExpected XDF file: {expected_xdf}")
        print("Make sure LabRecorder saves to this location")
        print("(or press Enter to specify different path)")

        user_input = input("> ").strip()
        if user_input:
            return Path(user_input)
        else:
            return expected_xdf

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return None

    finally:
        # CRITICAL: Close LSL streams FIRST to prevent zombie streams in LabRecorder
        xdf_writer.close()

        # Then disconnect hardware
        eeg.stop_stream()
        eeg.disconnect()
        if arduino.is_connected:
            arduino.disconnect()


def train_trca_model(xdf_path: Path, subject: str, config: SSVEPConfig):
    """Train TRCA model with batch learning (combines all previous sessions).

    Returns:
        Path to saved model file
    """
    print("\n" + "=" * 60)
    print("TRCA Model Training (Batch Learning)")
    print("=" * 60)

    # Find all XDF files for this subject (batch learning)
    calib_dir = xdf_path.parent
    subject_xdfs = sorted(calib_dir.glob(f"{subject}_*.xdf"))

    print(f"\nFound {len(subject_xdfs)} calibration session(s) for {subject}:")
    for xdf in subject_xdfs:
        print(f"  - {xdf.name}")

    # Load and combine trials from all sessions
    all_trials = []
    all_labels = []

    for xdf_file in subject_xdfs:
        print(f"\nLoading {xdf_file.name}...")
        X, y = extract_trials_from_xdf(str(xdf_file), config,
                                       trial_duration_sec=4.0, baseline_sec=0.5)
        print(f"  Extracted {len(X)} trials")

        all_trials.append(X)
        all_labels.append(y)

    # Concatenate all sessions
    X_combined = np.concatenate(all_trials, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)

    print(f"\nTotal trials across all sessions: {len(X_combined)}")
    print(f"  Shape: {X_combined.shape}")

    # Preprocess with filter bank (offline, zero-phase)
    print(f"\nPreprocessing with {config.num_fbs} filter banks (Chebyshev, zero-phase)...")
    preprocessor = OfflinePreprocessor(config)
    X_preprocessed = preprocessor.process(X_combined)

    print(f"  Preprocessed shape: {X_preprocessed.shape}")

    # Train/test split (80/20)
    n_total = len(X_preprocessed)
    n_test = int(n_total * 0.2)
    indices = np.random.permutation(n_total)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X_preprocessed[train_idx], y_combined[train_idx]
    X_test, y_test = X_preprocessed[test_idx], y_combined[test_idx]

    print(f"  Training: {len(X_train)} trials")
    print(f"  Testing: {len(X_test)} trials")

    # Train TRCA with filter banks
    print("\nTraining TRCA with filter bank analysis...")
    trca = TRCA(config, n_components=1)
    trca.fit(X_train, y_train)

    # Evaluate on train/test split
    y_pred_train, _ = trca.predict_with_correlation(X_train)
    y_pred_test, _ = trca.predict_with_correlation(X_test)

    acc_train = np.mean(y_pred_train == y_train)
    acc_test = np.mean(y_pred_test == y_test)

    print(f"\n  Training accuracy: {acc_train*100:.1f}%")
    print(f"  Test accuracy: {acc_test*100:.1f}%")

    # Optional: LOOCV evaluation for better accuracy estimate
    print("\nRunning Leave-One-Out Cross-Validation...")
    loocv_correct = 0
    n_total_trials = len(X_preprocessed)

    for loocv_i in range(n_total_trials):
        # Train on all except one
        train_mask = np.ones(n_total_trials, dtype=bool)
        train_mask[loocv_i] = False

        X_loocv_train = X_preprocessed[train_mask]
        y_loocv_train = y_combined[train_mask]

        X_loocv_test = X_preprocessed[loocv_i:loocv_i+1]
        y_loocv_test = y_combined[loocv_i]

        # Train and predict
        trca_loocv = TRCA(config, n_components=1)
        trca_loocv.fit(X_loocv_train, y_loocv_train)

        y_pred_loocv, _ = trca_loocv.predict_with_correlation(X_loocv_test)

        if y_pred_loocv == y_loocv_test:
            loocv_correct += 1

        # Progress indicator
        if (loocv_i + 1) % 10 == 0 or (loocv_i + 1) == n_total_trials:
            print(f"  Progress: {loocv_i+1}/{n_total_trials} trials evaluated")

    acc_loocv = loocv_correct / n_total_trials
    print(f"\n  LOOCV accuracy: {acc_loocv*100:.1f}% ({loocv_correct}/{n_total_trials} correct)")

    # Retrain on ALL data for final model (since we already validated with LOOCV)
    print("\nRetraining on all data for final model...")
    trca_final = TRCA(config, n_components=1)
    trca_final.fit(X_preprocessed, y_combined)
    print("  Final model trained on all trials")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{subject}_trca_{timestamp}.pkl"

    model_data = {
        'trca': trca_final,
        'config': config,
        'train_accuracy': acc_train,
        'test_accuracy': acc_test,
        'loocv_accuracy': acc_loocv,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'n_sessions': len(subject_xdfs),
        'n_total_trials': len(X_combined)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved: {model_path}")
    print(f"Trained on {len(subject_xdfs)} session(s), {len(X_combined)} total trials")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="SSVEP Calibration & Training")
    parser.add_argument("--subject", type=str, default="default",
                       help="Subject ID (e.g., alice)")
    parser.add_argument("--session", type=int, default=None,
                       help="Session number (auto-detected if not specified)")
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

    # Auto-detect session number if not specified
    if args.session is None:
        calib_dir = Path("calibration")
        calib_dir.mkdir(exist_ok=True)

        # Find existing XDF files for this subject
        existing_xdfs = sorted(calib_dir.glob(f"{args.subject}_S*.xdf"))

        if existing_xdfs:
            # Extract session numbers from filenames (format: subject_S001_eeg.xdf)
            session_numbers = []
            for xdf_file in existing_xdfs:
                try:
                    # Parse filename: subject_S001_eeg.xdf -> 001
                    parts = xdf_file.stem.split('_')
                    for part in parts:
                        if part.startswith('S') and part[1:].isdigit():
                            session_numbers.append(int(part[1:]))
                            break
                except:
                    continue

            if session_numbers:
                next_session = max(session_numbers) + 1
                print(f"Auto-detected: Found {len(session_numbers)} previous session(s)")
                print(f"Starting session {next_session} for subject '{args.subject}'")
            else:
                next_session = 1
                print(f"Starting first session for subject '{args.subject}'")
        else:
            next_session = 1
            print(f"No previous sessions found. Starting session 1 for subject '{args.subject}'")

        args.session = next_session
    else:
        print(f"Using manual session number: {args.session}")

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

    if not xdf_path:
        print("\n" + "=" * 60)
        print("CALIBRATION ABORTED")
        print("=" * 60)
        print("Emergency stop activated or user cancelled.")
        print("No data was saved, no model trained.")
        return 1

    if not Path(xdf_path).exists():
        print("\nXDF file not found - training skipped")
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
