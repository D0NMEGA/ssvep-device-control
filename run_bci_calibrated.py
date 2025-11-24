#!/usr/bin/env python3
"""
SSVEP BCI with Calibrated Templates

Uses personalized templates from calibration for improved classification.
Run this AFTER completing calibration with run_calibration.py.

Usage:
    python run_bci_calibrated.py --calibration calibration/calibration_Donovan_Santine_*.npz
    python run_bci_calibrated.py --calibration calibration/calibration_Donovan_Santine_*.npz --cyton COM3
"""

import sys
import time
import signal
import argparse
from pathlib import Path
import glob

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig, create_config
from models.eeg_buffer import EEGBuffer
from models.template_cca import TemplateCCADecoder, TemplateCCAConfig
from models.calibration import CalibrationCollector, CalibrationData
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController, BCIFeedbackController


class CalibratedBCI:
    """BCI system using calibrated templates."""

    def __init__(
        self,
        calibration_data: CalibrationData,
        arduino_port: str = None,
        cyton_port: str = None,
        use_synthetic: bool = False,
        synthetic_target: float = 10.0,
        margin_threshold: float = 0.10
    ):
        self.use_synthetic = use_synthetic
        self.synthetic_target = synthetic_target

        # Config
        self.config = create_config(margin_threshold=margin_threshold)

        # Store calibration data
        self.calibration_data = calibration_data
        print(f"Using calibration data:")
        print(f"  Subject: {self.calibration_data.subject_id}")
        print(f"  Epochs: {self.calibration_data.epochs.shape}")
        print(f"  Frequencies: {self.calibration_data.frequencies}")

        # Create template-based decoder
        template_config = TemplateCCAConfig(
            standard_weight=0.3,  # 30% standard CCA
            template_weight=0.7,  # 70% template matching
        )
        self.decoder = TemplateCCADecoder(
            config=self.config,
            template_config=template_config,
            calibration_data=self.calibration_data
        )

        # Arduino controller
        self.arduino = ArduinoController(port=arduino_port)

        # EEG components
        self.buffer = EEGBuffer(self.config)

        if use_synthetic:
            self.eeg_driver = SyntheticSSVEPDriver(
                self.config,
                target_frequency=synthetic_target
            )
        else:
            self.eeg_driver = BrainFlowDriver(self.config)
            self.eeg_driver.config.serial_port = cyton_port

        # Feedback controller
        self.feedback = None

        # State
        self.is_running = False
        self._n_windows = 0
        self._correct_predictions = 0

        # LED labels for display
        self.led_labels = {
            8.57: "LED 1 (far left)",
            10.0: "LED 2 (center-left)",
            12.0: "LED 3 (center-right)",
            15.0: "LED 4 (far right)",
        }

    def connect(self) -> bool:
        """Connect to all hardware."""
        print("\n" + "=" * 60)
        print("SSVEP BCI with Calibrated Templates")
        print("=" * 60)

        # Connect to Arduino
        print("\n[1/2] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  WARNING: Arduino not connected")
        else:
            print(f"  Connected on {self.arduino.port}")
            self.feedback = BCIFeedbackController(self.arduino)

        # Connect to EEG
        mode = "Synthetic EEG" if self.use_synthetic else "OpenBCI Cyton"
        print(f"\n[2/2] Connecting to {mode}...")
        if not self.eeg_driver.connect():
            print("  ERROR: Failed to connect to EEG device")
            return False
        print(f"  Connected! Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        return True

    def start(self) -> bool:
        """Start BCI acquisition."""
        # Start Arduino
        if self.arduino.is_connected:
            self.arduino.start_stimulation()
            time.sleep(0.5)

        # Start EEG
        if not self.eeg_driver.start_stream():
            print("ERROR: Failed to start EEG stream")
            return False

        self.buffer.reset()
        self.decoder.reset()
        self._n_windows = 0
        self._correct_predictions = 0
        self.is_running = True

        print("\n" + "=" * 60)
        print("BCI Running with Calibrated Templates")
        print("=" * 60)
        print("LED Layout (left to right):")
        print("  8.57 Hz | 10.0 Hz | 12.0 Hz | 15.0 Hz")
        print("\nPress Ctrl+C to stop\n")

        return True

    def stop(self) -> dict:
        """Stop BCI and return stats."""
        self.is_running = False
        self.eeg_driver.stop_stream()

        if self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()

        stats = {'windows_processed': self._n_windows}
        if self.use_synthetic and self._n_windows > 0:
            stats['accuracy'] = self._correct_predictions / self._n_windows * 100

        return stats

    def disconnect(self):
        """Disconnect all hardware."""
        if self.is_running:
            self.stop()
        self.eeg_driver.disconnect()
        if self.arduino.is_connected:
            self.arduino.disconnect()

    def process_step(self):
        """Process one BCI step."""
        if not self.is_running:
            return

        # Get EEG data
        data = self.eeg_driver.get_data()
        if data is not None and data.shape[1] > 0:
            self.buffer.append(data)

        # Process windows
        while self.buffer.ready():
            window = self.buffer.get_window()
            if window is None:
                break

            result = self.decoder.step(window)
            self._n_windows += 1

            # Send feedback
            if result.committed_prediction is not None and self.feedback:
                self.feedback.on_prediction(result.committed_prediction)

                if self.use_synthetic and result.committed_prediction == self.synthetic_target:
                    self._correct_predictions += 1

            # Display
            self._display_result(result)

    def _display_result(self, result):
        """Display classification result."""
        # Build correlation display
        freq_strs = []
        for freq in self.config.target_frequencies:
            corr = result.correlations.get(freq, 0)
            bar = '#' * int(corr * 20) + '-' * (20 - int(corr * 20))
            marker = " <--" if freq == result.instantaneous_prediction else ""
            label = self.led_labels.get(freq, f"{freq:.2f} Hz")
            freq_strs.append(f"{freq:5.2f}Hz ({label:20}) |{bar}| {corr:.3f}{marker}")

        # Status
        if result.committed_prediction:
            label = self.led_labels.get(result.committed_prediction, "")
            status = f"\033[92mDETECTED: {result.committed_prediction:.2f} Hz - {label}\033[0m"
        else:
            status = "\033[93mSearching...\033[0m"

        # Clear and print
        print("\033[2J\033[H", end="")
        print("=" * 70)
        print(f"SSVEP BCI (Calibrated) - Window #{self._n_windows}")
        print("=" * 70)
        print("\nCorrelations:")
        for s in freq_strs:
            print(f"  {s}")
        print(f"\nStatus: {status}")
        print(f"Margin: {result.margin:.3f} (threshold: {self.config.margin_threshold})")
        print(f"Latency: {result.processing_time_ms:.1f} ms")

        if self.use_synthetic:
            print(f"\n[Synthetic mode - simulating {self.synthetic_target} Hz]")
            if self._n_windows > 0:
                acc = self._correct_predictions / self._n_windows * 100
                print(f"Accuracy: {acc:.1f}%")


def find_calibration(
    subject: str = None,
    directory: str = "calibration",
    combine_all: bool = True,
    max_sessions: int = None
):
    """Find and optionally combine calibration data.

    Args:
        subject: Subject ID
        directory: Calibration directory
        combine_all: If True, combine all sessions for incremental learning
        max_sessions: Max sessions to combine (None = all)

    Returns:
        CalibrationData object or None
    """
    if subject:
        if combine_all:
            # Use incremental learning - combine all sessions
            return CalibrationCollector.get_combined_calibration(
                subject, directory, max_sessions
            )
        else:
            # Just use latest session
            path = CalibrationCollector.get_latest_session(subject, directory)
            if path:
                return CalibrationCollector.load(path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP BCI with Calibrated Templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bci_calibrated.py --subject Donovan_Santine --cyton COM3
  python run_bci_calibrated.py --calibration calibration/my_calibration.npz
  python run_bci_calibrated.py --subject Donovan_Santine --synthetic
        """
    )

    parser.add_argument(
        '--calibration', '-c',
        type=str,
        default=None,
        help='Path to calibration .npz file'
    )

    parser.add_argument(
        '--subject', '-s',
        type=str,
        default=None,
        help='Subject ID (finds latest calibration for this subject)'
    )

    parser.add_argument(
        '--arduino', '-a',
        type=str,
        default=None,
        help='Arduino serial port'
    )

    parser.add_argument(
        '--cyton',
        type=str,
        default=None,
        help='Cyton serial port'
    )

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic EEG (testing)'
    )

    parser.add_argument(
        '--target', '-t',
        type=float,
        default=10.0,
        choices=[8.57, 10.0, 12.0, 15.0],
        help='Target frequency for synthetic mode'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Duration in seconds'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=0.10,
        help='Margin threshold'
    )

    parser.add_argument(
        '--list-calibrations',
        action='store_true',
        help='List available calibration files'
    )

    parser.add_argument(
        '--latest-only',
        action='store_true',
        help='Use only the latest session (disable incremental learning)'
    )

    parser.add_argument(
        '--max-sessions',
        type=int,
        default=None,
        help='Max number of sessions to combine (default: all)'
    )

    args = parser.parse_args()

    # List calibrations
    if args.list_calibrations:
        print("Available calibration sessions:")
        cal_dir = Path("calibration")
        if cal_dir.exists():
            for subject_dir in sorted(cal_dir.iterdir()):
                if subject_dir.is_dir():
                    sessions = list(subject_dir.glob("session_*.npz"))
                    print(f"\n  {subject_dir.name}/ ({len(sessions)} sessions)")
                    for s in sorted(sessions)[-3:]:  # Show last 3
                        print(f"    - {s.name}")
                    if len(sessions) > 3:
                        print(f"    ... and {len(sessions)-3} more")
        return 0

    # Find calibration data
    if args.calibration:
        # Direct file path
        cal_data = CalibrationCollector.load(args.calibration)
        print(f"Loaded calibration from: {args.calibration}")
    elif args.subject:
        # Get calibration for subject (with incremental learning)
        sessions = CalibrationCollector.get_subject_sessions(args.subject)
        if not sessions:
            print(f"ERROR: No calibration found for subject '{args.subject}'")
            print(f"Run: python run_calibration.py --subject {args.subject}")
            return 1

        if args.latest_only:
            cal_data = CalibrationCollector.load(sessions[-1])
            print(f"Using latest session: {sessions[-1]}")
        else:
            # Incremental learning - combine sessions
            cal_data = find_calibration(
                args.subject,
                combine_all=True,
                max_sessions=args.max_sessions
            )
            print(f"Combined {len(sessions)} session(s) for incremental learning")
    else:
        print("ERROR: Please specify --subject or --calibration")
        print("Run: python run_bci_calibrated.py --subject YOUR_NAME --cyton COM3")
        return 1

    # Create BCI
    bci = CalibratedBCI(
        calibration_data=cal_data,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        synthetic_target=args.target,
        margin_threshold=args.margin
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\nStopping...")
        stats = bci.stop()
        bci.disconnect()
        print("\nSession Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Connect
    if not bci.connect():
        return 1

    # Start
    if not bci.start():
        bci.disconnect()
        return 1

    # Main loop
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            bci.process_step()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    # Stop
    stats = bci.stop()
    bci.disconnect()

    print("\n" + "=" * 60)
    print("Session Complete")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
