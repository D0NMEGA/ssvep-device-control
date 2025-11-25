#!/usr/bin/env python3
"""
SSVEP BCI with Arduino Visual Feedback

This script runs the complete BCI loop:
1. Connects to Arduino and starts LED stimulation
2. Connects to OpenBCI Cyton for EEG acquisition
3. Classifies which LED the user is looking at
4. Lights up the corresponding red LED as feedback

Usage:
    python run_bci_with_feedback.py                     # Auto-detect ports
    python run_bci_with_feedback.py --arduino COM3      # Specify Arduino port
    python run_bci_with_feedback.py --cyton COM4        # Specify Cyton port
    python run_bci_with_feedback.py --synthetic         # Test without hardware
"""

import sys
import time
import signal
import argparse
from pathlib import Path

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig, create_config
from models.eeg_buffer import EEGBuffer
from models.cca_decoder import SSVEPDecoder
from models.template_cca import TemplateCCADecoder, TemplateCCAConfig
from models.calibration import CalibrationCollector
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController, BCIFeedbackController


class BCIWithFeedback:
    """Complete BCI system with Arduino visual feedback."""

    def __init__(
        self,
        arduino_port: str = None,
        cyton_port: str = None,
        subject: str = None,
        use_synthetic: bool = False,
        synthetic_target: float = 10.0,
        margin_threshold: float = 0.10,
        feedback_duration: float = 0.5
    ):
        """Initialize the BCI system.

        Args:
            arduino_port: Serial port for Arduino (auto-detect if None)
            cyton_port: Serial port for Cyton (auto-detect if None)
            subject: Subject ID for loading calibration templates (optional)
            use_synthetic: Use synthetic EEG for testing (no Cyton needed)
            synthetic_target: Target frequency for synthetic mode
            margin_threshold: Classification margin threshold
            feedback_duration: How long to show feedback LED (seconds)
        """
        self.use_synthetic = use_synthetic
        self.synthetic_target = synthetic_target
        self.subject = subject

        # Config with adjusted threshold
        self.config = create_config(margin_threshold=margin_threshold)

        # Arduino controller
        self.arduino = ArduinoController(port=arduino_port)

        # EEG components
        self.buffer = EEGBuffer(self.config)

        # Decoder - try to load templates if subject provided
        self.decoder = None
        self.using_templates = False

        if subject:
            self._load_calibration()

        # Create standard decoder if no templates loaded
        if self.decoder is None:
            print("Using standard CCA (no calibration templates)")
            self.decoder = SSVEPDecoder(self.config)

        if use_synthetic:
            self.eeg_driver = SyntheticSSVEPDriver(
                self.config,
                target_frequency=synthetic_target
            )
        else:
            self.eeg_driver = BrainFlowDriver(self.config)
            self.eeg_driver.config.serial_port = cyton_port

        # Feedback controller
        self.feedback = None  # Created after Arduino connects

        # State
        self.is_running = False
        self._n_windows = 0
        self._correct_predictions = 0

    def _load_calibration(self):
        """Load calibration templates for subject."""
        sessions = CalibrationCollector.get_subject_sessions(self.subject)
        if not sessions:
            print(f"No calibration found for subject '{self.subject}'")
            return

        try:
            # Combine all sessions
            cal_data = CalibrationCollector.get_combined_calibration(self.subject)
            if cal_data is None:
                return

            # Create template decoder
            template_config = TemplateCCAConfig(
                standard_weight=0.3,
                template_weight=0.7
            )

            self.decoder = TemplateCCADecoder(
                self.config,
                template_config,
                cal_data
            )

            self.using_templates = True
            print(f"✓ Loaded calibration for {self.subject} ({len(sessions)} sessions)")

        except Exception as e:
            print(f"Error loading calibration: {e}")

    def connect(self) -> bool:
        """Connect to all hardware.

        Returns:
            True if all connections successful
        """
        print("=" * 60)
        print("SSVEP BCI with Visual Feedback")
        print("=" * 60)

        # Connect to Arduino
        print("\n[1/2] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  ERROR: Failed to connect to Arduino")
            print("  Tip: Check that Arduino is connected and port is correct")
            return False
        print(f"  Connected on {self.arduino.port}")

        # Create feedback controller
        self.feedback = BCIFeedbackController(self.arduino)

        # Connect to EEG
        print(f"\n[2/2] Connecting to {'Synthetic EEG' if self.use_synthetic else 'OpenBCI Cyton'}...")
        if not self.eeg_driver.connect():
            print("  ERROR: Failed to connect to EEG device")
            self.arduino.disconnect()
            return False
        print(f"  Connected! Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        return True

    def start(self) -> bool:
        """Start stimulation and EEG acquisition.

        Returns:
            True if started successfully
        """
        # Start Arduino stimulation
        print("\nStarting LED stimulation...")
        self.arduino.start_stimulation()
        time.sleep(0.5)

        # Start EEG streaming
        print("Starting EEG acquisition...")
        if not self.eeg_driver.start_stream():
            print("ERROR: Failed to start EEG stream")
            self.arduino.stop_stimulation()
            return False

        # Reset state
        self.buffer.reset()
        self.decoder.reset()
        self._n_windows = 0
        self._correct_predictions = 0

        self.is_running = True

        print("\n" + "=" * 60)
        print("BCI Running - Look at one of the flickering LEDs")
        print("=" * 60)
        print(f"Target frequencies: {self.config.target_frequencies} Hz")
        print("Press Ctrl+C to stop\n")

        return True

    def stop(self) -> dict:
        """Stop the BCI and return statistics.

        Returns:
            Dictionary with session statistics
        """
        self.is_running = False

        # Stop EEG
        self.eeg_driver.stop_stream()

        # Stop Arduino
        self.arduino.stop_stimulation()
        self.arduino.clear_feedback()

        stats = {
            'windows_processed': self._n_windows,
        }

        if self.use_synthetic and self._n_windows > 0:
            accuracy = self._correct_predictions / self._n_windows * 100
            stats['accuracy'] = accuracy

        return stats

    def disconnect(self) -> None:
        """Disconnect from all hardware."""
        if self.is_running:
            self.stop()

        self.eeg_driver.disconnect()
        self.arduino.disconnect()

    def process_step(self) -> None:
        """Process one step of the BCI loop."""
        if not self.is_running:
            return

        # Get EEG data
        data = self.eeg_driver.get_data()

        if data is not None and data.shape[1] > 0:
            self.buffer.append(data)

        # Process available windows
        while self.buffer.ready():
            window = self.buffer.get_window()
            if window is None:
                break

            # Classify
            result = self.decoder.step(window)
            self._n_windows += 1

            # Send feedback for committed predictions
            if result.committed_prediction is not None:
                self.feedback.on_prediction(result.committed_prediction)

                # Track accuracy for synthetic mode
                if self.use_synthetic and result.committed_prediction == self.synthetic_target:
                    self._correct_predictions += 1

            # Display
            self._display_result(result)

    def _display_result(self, result) -> None:
        """Display classification result."""
        # Build display
        freq_strs = []
        for freq in self.config.target_frequencies:
            corr = result.correlations.get(freq, 0)
            bar = '#' * int(corr * 20) + '-' * (20 - int(corr * 20))
            marker = " <--" if freq == result.instantaneous_prediction else ""
            freq_strs.append(f"{freq:5.2f}Hz |{bar}| {corr:.3f}{marker}")

        # Status
        if result.committed_prediction:
            status = f"\033[92mDETECTED: {result.committed_prediction:.2f} Hz\033[0m"
        else:
            status = "\033[93mSearching...\033[0m"

        # Clear screen and print
        print("\033[2J\033[H", end="")
        print("=" * 55)
        print(f"SSVEP BCI - Window #{self._n_windows}")
        print("=" * 55)
        print("\nCorrelations:")
        for s in freq_strs:
            print(f"  {s}")
        print(f"\nStatus: {status}")
        print(f"Margin: {result.margin:.3f} (threshold: {self.config.margin_threshold})")

        if self.use_synthetic:
            print(f"\n[Synthetic mode - simulating {self.synthetic_target} Hz]")
            if self._n_windows > 0:
                acc = self._correct_predictions / self._n_windows * 100
                print(f"Accuracy: {acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP BCI with Arduino Visual Feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bci_with_feedback.py                   # Auto-detect everything
  python run_bci_with_feedback.py --synthetic       # Test without Cyton
  python run_bci_with_feedback.py --arduino COM3    # Specify Arduino port
  python run_bci_with_feedback.py --cyton COM4      # Specify Cyton port
        """
    )

    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID for loading calibration templates (optional)'
    )

    parser.add_argument(
        '--arduino', '-a',
        type=str,
        default=None,
        help='Arduino serial port (auto-detect if not specified)'
    )

    parser.add_argument(
        '--cyton', '-c',
        type=str,
        default=None,
        help='Cyton serial port (auto-detect if not specified)'
    )

    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Use synthetic EEG (no Cyton required)'
    )

    parser.add_argument(
        '--target', '-t',
        type=float,
        default=10.0,
        choices=[8.57, 10.0, 12.0, 15.0],
        help='Target frequency for synthetic mode (default: 10.0)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Duration in seconds (default: 60)'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=0.10,
        help='Margin threshold (default: 0.10)'
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
        for port, desc in ArduinoController.list_ports():
            print(f"  {port}: {desc}")
        return 0

    # Create BCI
    bci = BCIWithFeedback(
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        subject=args.subject,
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

    # Stop and show stats
    stats = bci.stop()
    bci.disconnect()

    print("\n" + "=" * 55)
    print("Session Complete")
    print("=" * 55)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
