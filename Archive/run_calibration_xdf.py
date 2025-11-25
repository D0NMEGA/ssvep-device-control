#!/usr/bin/env python3
"""
SSVEP Calibration Runner with XDF/LSL Support

Collects calibration data for personalized SSVEP classification using:
- XDF format for data storage (LSL-compatible)
- LSL marker stream for event synchronization
- Optional baseline period
- Zero-phase filtering for offline analysis
- TRCA-based classification

Usage:
    python run_calibration_xdf.py                     # Auto-detect ports
    python run_calibration_xdf.py --arduino COM3      # Specify Arduino
    python run_calibration_xdf.py --cyton COM4        # Specify Cyton
    python run_calibration_xdf.py --subject alice     # Set subject ID
    python run_calibration_xdf.py --baseline 30       # 30 second baseline
"""

import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from pylsl import local_clock

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig
from utils.lsl_markers import LSLMarkerSender, EventMarkers
from utils.io_xdf import XDFCalibrationWriter
from models.eeg_buffer import EEGBuffer
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController

# Global flag to prevent beep buffering during trials
_beep_lock = threading.Lock()
_suppress_beeps = False


def beep_start():
    """Play start beep (high pitch) in non-blocking way."""
    global _suppress_beeps
    if _suppress_beeps:
        return

    def _beep():
        if _beep_lock.acquire(blocking=False):
            try:
                import winsound
                winsound.Beep(1000, 200)  # 1000 Hz for 200ms
            except:
                print("\a", end="", flush=True)
            finally:
                _beep_lock.release()

    threading.Thread(target=_beep, daemon=True).start()


def beep_end():
    """Play end beep (low pitch) in non-blocking way."""
    global _suppress_beeps
    if _suppress_beeps:
        return

    def _beep():
        if _beep_lock.acquire(blocking=False):
            try:
                import winsound
                winsound.Beep(500, 200)  # 500 Hz for 200ms
            except:
                print("\a", end="", flush=True)
            finally:
                _beep_lock.release()

    threading.Thread(target=_beep, daemon=True).start()


def beep_rest():
    """Play rest beep (two short beeps) in non-blocking way."""
    global _suppress_beeps
    if _suppress_beeps:
        return

    def _beep():
        if _beep_lock.acquire(blocking=False):
            try:
                import winsound
                winsound.Beep(800, 100)
                time.sleep(0.05)
                winsound.Beep(800, 100)
            except:
                print("\a", end="", flush=True)
            finally:
                _beep_lock.release()

    threading.Thread(target=_beep, daemon=True).start()


def suppress_beeps(suppress: bool):
    """Enable/disable beep suppression during critical periods."""
    global _suppress_beeps
    _suppress_beeps = suppress


class CalibrationRunnerXDF:
    """CLI runner for calibration data collection with XDF/LSL."""

    def __init__(
        self,
        arduino_port: str = None,
        cyton_port: str = None,
        use_synthetic: bool = False,
        subject_id: str = "default",
        session: int = 1,
        trial_duration: float = 4.0,
        rest_duration: float = 2.0,
        trials_per_frequency: int = 5,
        baseline_duration: float = 0.0
    ):
        self.config = SSVEPConfig()
        self.use_synthetic = use_synthetic
        self.subject_id = subject_id
        self.session = session
        self.trial_duration = trial_duration
        self.rest_duration = rest_duration
        self.trials_per_frequency = trials_per_frequency
        self.baseline_duration = baseline_duration

        # Arduino controller
        self.arduino = ArduinoController(port=arduino_port)

        # EEG driver
        if use_synthetic:
            self.eeg_driver = SyntheticSSVEPDriver(self.config, target_frequency=10.0)
        else:
            self.eeg_driver = BrainFlowDriver(self.config)
            self.eeg_driver.config.serial_port = cyton_port

        # XDF writer (creates LSL outlets)
        self.xdf_writer = XDFCalibrationWriter(
            subject=subject_id,
            session=session,
            config=self.config,
            stream_name_eeg="SSVEP-EEG",
            stream_name_markers="SSVEP-Markers"
        )

        # LSL marker sender
        self.marker_sender = LSLMarkerSender(stream_name="SSVEP-Markers")

        # State
        self.is_running = False
        self.total_trials = trials_per_frequency * len(self.config.target_frequencies)

    def connect(self) -> bool:
        """Connect to all hardware."""
        print("=" * 60)
        print("SSVEP Calibration Data Collection (XDF/LSL)")
        print("=" * 60)

        # Connect Arduino
        print("\n[1/2] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  WARNING: Arduino not connected. LED control unavailable.")
            print("  Continuing with audio cues only...")
        else:
            print(f"  Connected on {self.arduino.port}")

        # Connect EEG
        mode = "Synthetic EEG" if self.use_synthetic else "OpenBCI Cyton"
        print(f"\n[2/2] Connecting to {mode}...")
        if not self.eeg_driver.connect():
            print("  ERROR: Failed to connect to EEG device")
            return False
        print(f"  Connected! Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        print("\n[LSL] EEG and Marker streams are now available.")
        print("      Start LabRecorder NOW to capture this session to XDF.")

        return True

    def disconnect(self):
        """Disconnect all hardware."""
        if self.arduino.is_connected:
            self.arduino.disconnect()
        self.eeg_driver.disconnect()

    def run_baseline(self):
        """Run baseline period if configured."""
        if self.baseline_duration <= 0:
            return

        print("\n" + "=" * 60)
        print("Baseline Period")
        print("=" * 60)
        print(f"Duration: {self.baseline_duration} seconds")
        print("Please relax and keep your eyes open.")
        print("Starting in 3 seconds...")

        time.sleep(3)

        # Send baseline start marker
        self.marker_sender.send_baseline_start(duration_sec=self.baseline_duration)

        baseline_start = time.time()

        print("\nBaseline recording...")

        # Stream EEG during baseline
        while time.time() - baseline_start < self.baseline_duration:
            data = self.eeg_driver.get_data()
            if data is not None and data.shape[1] > 0:
                # Push to LSL
                for sample_idx in range(data.shape[1]):
                    sample = data[:, sample_idx]
                    self.xdf_writer.push_eeg(sample)
            time.sleep(0.01)

        # Send baseline end marker
        self.marker_sender.send_baseline_end()

        print("Baseline complete!")
        time.sleep(2)

    def run(self) -> bool:
        """Run the calibration session.

        Returns:
            True if completed successfully
        """
        print("\n" + "=" * 60)
        print("Calibration Session")
        print("=" * 60)

        print(f"\nSubject: {self.subject_id}")
        print(f"Session: {self.session}")
        print(f"Trials per frequency: {self.trials_per_frequency}")
        print(f"Trial duration: {self.trial_duration}s")
        print(f"Rest duration: {self.rest_duration}s")
        print(f"Total trials: {self.total_trials}")

        if self.baseline_duration > 0:
            print(f"Baseline duration: {self.baseline_duration}s")

        # LED positions (left to right)
        led_labels = {
            8.57: "LED 1 (far left)",
            10.0: "LED 2 (center-left)",
            12.0: "LED 3 (center-right)",
            15.0: "LED 4 (far right)",
        }

        print("\nLED Layout (left to right):")
        print("  LED 1 (far left)     = 8.57 Hz")
        print("  LED 2 (center-left)  = 10.0 Hz")
        print("  LED 3 (center-right) = 12.0 Hz")
        print("  LED 4 (far right)    = 15.0 Hz")

        # Countdown
        print("\n" + "-" * 60)
        print("Starting in 10 seconds...")
        print("Get ready to focus on the LEDs when cued.")
        print("-" * 60)

        for i in range(10, 0, -1):
            print(f"  {i}...", end=" ", flush=True)
            time.sleep(1)
        print("\nGO!\n")

        # Start EEG streaming
        if not self.eeg_driver.start_stream():
            print("ERROR: Failed to start EEG stream")
            return False

        self.is_running = True

        try:
            # Send calibration start marker
            self.marker_sender.send_start_calibration(
                subject=self.subject_id,
                session=self.session,
                date=datetime.now().isoformat()
            )

            # Run baseline if configured
            if self.baseline_duration > 0:
                self.run_baseline()

            # Generate trial sequence (randomized)
            trial_sequence = []
            for freq_idx, freq in enumerate(self.config.target_frequencies):
                for _ in range(self.trials_per_frequency):
                    trial_sequence.append((freq_idx, freq))

            # Shuffle trials
            np.random.shuffle(trial_sequence)

            # Main calibration loop
            for trial_idx, (freq_idx, freq) in enumerate(trial_sequence):
                # Show trial info
                led_label = led_labels.get(freq, f"{freq:.2f} Hz")
                print(f"\n{'='*50}")
                print(f"  Trial {trial_idx + 1}/{self.total_trials}")
                print(f"{'='*50}")
                print(f"  >>> LOOK AT: {freq:.2f} Hz - {led_label} <<<")
                print(f"{'='*50}")

                # Send trial start marker
                self.marker_sender.send_trial_start(
                    trial=trial_idx + 1,
                    target_freq=freq
                )

                # Start audio cue
                beep_start()
                time.sleep(0.3)

                # Suppress all beeps during trial
                suppress_beeps(True)

                # Light up the RED LED under the target as a visual cue
                if self.arduino.is_connected:
                    self.arduino.clear_feedback()
                    time.sleep(0.1)
                    self.arduino.show_feedback(freq)
                    time.sleep(0.3)
                    self.arduino.start_stimulation()

                # Send stim on marker
                self.marker_sender.send_stim_on(freq=freq, trial=trial_idx + 1)

                # Collect data for trial_duration
                trial_start = time.time()

                while time.time() - trial_start < self.trial_duration:
                    data = self.eeg_driver.get_data()
                    if data is not None and data.shape[1] > 0:
                        # Push to LSL
                        for sample_idx in range(data.shape[1]):
                            sample = data[:, sample_idx]
                            self.xdf_writer.push_eeg(sample)
                    time.sleep(0.01)

                # Send stim off marker
                self.marker_sender.send_stim_off(freq=freq, trial=trial_idx + 1)

                # Re-enable beeps and play end beep
                suppress_beeps(False)
                beep_end()
                time.sleep(0.3)

                if self.arduino.is_connected:
                    self.arduino.stop_stimulation()
                    self.arduino.clear_feedback()

                # Progress
                progress = (trial_idx + 1) / self.total_trials * 100
                print(f"Progress: {progress:.0f}%")

                # Rest period (unless last trial)
                if trial_idx < self.total_trials - 1:
                    print("Rest...")

                    # Send rest start marker
                    self.marker_sender.send_rest_start(trial=trial_idx + 1, duration_sec=self.rest_duration)

                    beep_rest()
                    rest_start = time.time()

                    # Continue streaming during rest
                    while time.time() - rest_start < self.rest_duration:
                        data = self.eeg_driver.get_data()
                        if data is not None and data.shape[1] > 0:
                            for sample_idx in range(data.shape[1]):
                                sample = data[:, sample_idx]
                                self.xdf_writer.push_eeg(sample)
                        time.sleep(0.01)

                    # Send rest end marker
                    self.marker_sender.send_rest_end(trial=trial_idx + 1)

            # Send calibration end marker
            self.marker_sender.send_end_calibration(
                n_trials=self.total_trials,
                duration_sec=time.time() - trial_start
            )

            print("\n" + "=" * 60)
            print("Calibration Complete!")
            print("=" * 60)
            print("\nStop LabRecorder to save the XDF file.")

        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user.")
            self.is_running = False

        # Stop streaming
        self.eeg_driver.stop_stream()
        self.is_running = False

        return True


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP Calibration with XDF/LSL Support"
    )
    parser.add_argument(
        "--arduino",
        type=str,
        default=None,
        help="Arduino serial port (e.g., COM3). Auto-detect if not specified."
    )
    parser.add_argument(
        "--cyton",
        type=str,
        default=None,
        help="OpenBCI Cyton serial port (e.g., COM4). Auto-detect if not specified."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic EEG data for testing (no hardware required)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="default",
        help="Subject ID (default: 'default')"
    )
    parser.add_argument(
        "--session",
        type=int,
        default=1,
        help="Session number (default: 1)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Trials per frequency (default: 5)"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=30.0,
        help="Baseline duration in seconds (default: 30, set to 0 to skip)"
    )
    parser.add_argument(
        "--trial-duration",
        type=float,
        default=4.0,
        help="Trial duration in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--rest-duration",
        type=float,
        default=2.0,
        help="Rest duration in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    # Create runner
    runner = CalibrationRunnerXDF(
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        subject_id=args.subject,
        session=args.session,
        trial_duration=args.trial_duration,
        rest_duration=args.rest_duration,
        trials_per_frequency=args.trials,
        baseline_duration=args.baseline
    )

    # Connect
    if not runner.connect():
        print("\nERROR: Failed to connect to hardware")
        return 1

    # Wait for user to start LabRecorder
    print("\n" + "=" * 60)
    print("IMPORTANT: Start LabRecorder NOW")
    print("=" * 60)
    print("1. Open LabRecorder")
    print("2. Check that you see 'SSVEP-EEG' and 'SSVEP-Markers' streams")
    print("3. Click 'Update' to refresh if needed")
    print("4. Click 'Start' to begin recording")
    print("5. Press Enter here when ready to begin calibration")
    input("\nPress Enter when LabRecorder is recording...")

    # Run calibration
    try:
        success = runner.run()
    finally:
        runner.disconnect()

    if success:
        print("\nCalibration completed successfully!")
        print("\nNext steps:")
        print("1. Stop LabRecorder and save the XDF file")
        print("2. Run analysis script to process XDF and train TRCA model")
        return 0
    else:
        print("\nCalibration failed or incomplete.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
