#!/usr/bin/env python3
"""
SSVEP Calibration Runner

Collects calibration data for personalized SSVEP classification.
Uses single-LED paradigm with audio cues.

Usage:
    python run_calibration.py                     # Auto-detect ports
    python run_calibration.py --arduino COM3      # Specify Arduino
    python run_calibration.py --cyton COM4        # Specify Cyton
    python run_calibration.py --synthetic         # Test without hardware
    python run_calibration.py --subject alice     # Set subject ID
"""

import sys
import time
import argparse
import winsound  # Windows only - for audio cues
from pathlib import Path

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig
from models.eeg_buffer import EEGBuffer
from models.calibration import CalibrationCollector, FREQ_TO_EVENT_ID
from drivers.brainflow_driver import BrainFlowDriver, SyntheticSSVEPDriver
from drivers.arduino_controller import ArduinoController


def beep_start():
    """Play start beep (high pitch)."""
    try:
        winsound.Beep(1000, 200)  # 1000 Hz for 200ms
    except:
        print("\a", end="", flush=True)  # Fallback beep


def beep_end():
    """Play end beep (low pitch)."""
    try:
        winsound.Beep(500, 200)  # 500 Hz for 200ms
    except:
        print("\a", end="", flush=True)


def beep_rest():
    """Play rest beep (two short beeps)."""
    try:
        winsound.Beep(800, 100)
        time.sleep(0.05)
        winsound.Beep(800, 100)
    except:
        print("\a", end="", flush=True)


class CalibrationRunner:
    """CLI runner for calibration data collection."""

    def __init__(
        self,
        arduino_port: str = None,
        cyton_port: str = None,
        use_synthetic: bool = False,
        subject_id: str = "default",
        trial_duration: float = 4.0,
        rest_duration: float = 2.0,
        trials_per_frequency: int = 5
    ):
        self.config = SSVEPConfig()
        self.use_synthetic = use_synthetic

        # Arduino controller
        self.arduino = ArduinoController(port=arduino_port)

        # EEG driver
        if use_synthetic:
            self.eeg_driver = SyntheticSSVEPDriver(self.config, target_frequency=10.0)
        else:
            self.eeg_driver = BrainFlowDriver(self.config)
            self.eeg_driver.config.serial_port = cyton_port

        # EEG buffer
        self.buffer = EEGBuffer(self.config)

        # Calibration collector
        self.collector = CalibrationCollector(
            config=self.config,
            trial_duration=trial_duration,
            rest_duration=rest_duration,
            trials_per_frequency=trials_per_frequency,
            subject_id=subject_id
        )

        # State
        self.is_running = False

    def connect(self) -> bool:
        """Connect to all hardware."""
        print("=" * 60)
        print("SSVEP Calibration Data Collection")
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

        return True

    def disconnect(self):
        """Disconnect all hardware."""
        if self.arduino.is_connected:
            self.arduino.disconnect()
        self.eeg_driver.disconnect()

    def run(self) -> bool:
        """Run the calibration session.

        Returns:
            True if completed successfully
        """
        print("\n" + "=" * 60)
        print("Calibration Session")
        print("=" * 60)

        # Show trial sequence
        print(f"\nSubject: {self.collector.session.subject_id}")
        print(f"Trials per frequency: {self.collector.session.trials_per_frequency}")
        print(f"Trial duration: {self.collector.session.trial_duration}s")
        print(f"Rest duration: {self.collector.session.rest_duration}s")
        print(f"Total trials: {self.collector.session.total_trials}")

        # LED positions (left to right): LED 1, LED 2, LED 3, LED 4
        # Physical layout: 8.57 Hz on far left, 15 Hz on far right
        self.led_labels = {
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
        self.buffer.reset()

        try:
            # Main calibration loop
            while self.collector.session.current_trial < self.collector.session.total_trials:
                trial_idx, freq, event_id = self.collector.get_current_trial_info()

                # Show trial info with clear left-to-right label
                led_label = self.led_labels.get(freq, f"{freq:.2f} Hz")
                print(f"\n{'='*50}")
                print(f"  Trial {trial_idx + 1}/{self.collector.session.total_trials}")
                print(f"{'='*50}")
                print(f"  >>> LOOK AT: {freq:.2f} Hz - {led_label} <<<")
                print(f"{'='*50}")

                # Start audio cue
                beep_start()

                # Light up the RED LED under the target as a visual cue
                if self.arduino.is_connected:
                    self.arduino.clear_feedback()  # Turn off all red LEDs first
                    time.sleep(0.1)
                    self.arduino.show_feedback(freq)  # Light red LED under target
                    time.sleep(0.3)  # Brief pause to see the cue
                    self.arduino.start_stimulation()  # Start all white LEDs

                # Start trial
                self.collector.start_trial()
                trial_start = time.time()

                # Collect data for trial_duration
                while time.time() - trial_start < self.collector.session.trial_duration:
                    data = self.eeg_driver.get_data()
                    if data is not None and data.shape[1] > 0:
                        self.collector.add_data(data)
                    time.sleep(0.01)

                # End trial
                beep_end()
                self.collector.end_trial()

                if self.arduino.is_connected:
                    self.arduino.stop_stimulation()
                    self.arduino.clear_feedback()  # Turn off red LED cue

                # Progress
                progress = (trial_idx + 1) / self.collector.session.total_trials * 100
                print(f"Progress: {progress:.0f}%")

                # Rest period (unless last trial)
                if self.collector.session.current_trial < self.collector.session.total_trials:
                    print("Rest...")
                    beep_rest()
                    time.sleep(self.collector.session.rest_duration)

        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user.")
            self.is_running = False

        # Stop streaming
        self.eeg_driver.stop_stream()
        self.is_running = False

        return self.collector.session.is_complete

    def save(self, output_dir: str = "calibration") -> tuple:
        """Save calibration data.

        Returns:
            Tuple of (npz_path, json_path)
        """
        return self.collector.save(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP Calibration Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_calibration.py                       # Auto-detect
  python run_calibration.py --synthetic           # Test mode
  python run_calibration.py --subject alice       # Set subject ID
  python run_calibration.py --trials 3 --duration 3  # Shorter session
        """
    )

    parser.add_argument(
        '--arduino', '-a',
        type=str,
        default=None,
        help='Arduino serial port'
    )

    parser.add_argument(
        '--cyton', '-c',
        type=str,
        default=None,
        help='Cyton serial port'
    )

    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Use synthetic EEG (testing)'
    )

    parser.add_argument(
        '--subject',
        type=str,
        default='default',
        help='Subject ID for saving data'
    )

    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=5,
        help='Trials per frequency (default: 5)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=4.0,
        help='Trial duration in seconds (default: 4.0)'
    )

    parser.add_argument(
        '--rest', '-r',
        type=float,
        default=2.0,
        help='Rest duration in seconds (default: 2.0)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='calibration',
        help='Output directory (default: calibration)'
    )

    parser.add_argument(
        '--list-ports',
        action='store_true',
        help='List available serial ports'
    )

    args = parser.parse_args()

    # List ports
    if args.list_ports:
        print("Available serial ports:")
        for port, desc in ArduinoController.list_ports():
            print(f"  {port}: {desc}")
        return 0

    # Create runner
    runner = CalibrationRunner(
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        subject_id=args.subject,
        trial_duration=args.duration,
        rest_duration=args.rest,
        trials_per_frequency=args.trials
    )

    # Connect
    if not runner.connect():
        return 1

    try:
        # Run calibration
        if runner.run():
            print("\n" + "=" * 60)
            print("Calibration Complete!")
            print("=" * 60)

            # Save data
            npz_path, json_path = runner.save(args.output)
            print(f"\nData saved to:")
            print(f"  {npz_path}")
            print(f"  {json_path}")

            # Show summary
            data = runner.collector.get_calibration_data()
            if data:
                print(f"\nSummary:")
                print(f"  Subject: {data.subject_id}")
                print(f"  Epochs: {data.epochs.shape}")
                print(f"  Frequencies: {data.frequencies}")

                # Count per frequency
                for freq in data.frequencies:
                    count = np.sum(np.isclose(data.labels, freq, atol=0.1))
                    print(f"    {freq:.2f} Hz: {count} trials")

            return 0
        else:
            print("\nCalibration incomplete.")
            return 1

    finally:
        runner.disconnect()


if __name__ == "__main__":
    import numpy as np  # For summary stats
    sys.exit(main())
