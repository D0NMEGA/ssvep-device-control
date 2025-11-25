#!/usr/bin/env python3
"""
Real-time Hardware Troubleshooting

Tests EEG and Arduino connections, displays signal quality,
and helps identify hardware issues.

Usage:
    python troubleshoot_hardware.py --cyton COM3 --arduino COM4
    python troubleshoot_hardware.py  # Auto-detect ports
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController
from ssvep_bci.preprocessor import OnlinePreprocessor


class HardwareTroubleshooter:
    """Real-time hardware testing and diagnostics."""

    def __init__(self, cyton_port: str = None, arduino_port: str = None):
        self.config = SSVEPConfig()
        self.cyton_port = cyton_port
        self.arduino_port = arduino_port

        # Drivers
        self.eeg_driver = None
        self.arduino = None

        # Preprocessor for filtering
        self.preprocessor = OnlinePreprocessor(self.config)

        # Signal quality tracking
        self.signal_history = [deque(maxlen=250) for _ in range(8)]  # 1 second
        self.signal_history_filtered = [deque(maxlen=250) for _ in range(8)]  # 1 second filtered
        self.impedance_warnings = set()

    def test_arduino(self) -> bool:
        """Test Arduino connection and LED control."""
        print("\n" + "="*60)
        print("ARDUINO TEST")
        print("="*60)

        # Connect
        self.arduino = ArduinoController(port=self.arduino_port)
        print("\n[1/4] Connecting to Arduino...")
        if not self.arduino.connect():
            print("  [FAIL] Could not connect to Arduino")
            return False
        print(f"  [OK] Connected on {self.arduino.port}")

        # Test white LED stimulation
        print("\n[2/4] Testing white LEDs (3 seconds)...")
        self.arduino.start_stimulation()
        for i in range(3):
            print(f"  White LEDs flickering... {3-i}s")
            time.sleep(1)
        self.arduino.stop_stimulation()
        print("  [OK] White LEDs working")

        # Test red feedback LEDs
        print("\n[3/4] Testing red feedback LEDs...")
        for freq, label in [(8.57, "far left"), (10.0, "center-left"),
                            (12.0, "center-right"), (15.0, "far right")]:
            print(f"  Lighting {freq} Hz ({label})...")
            self.arduino.show_feedback(freq)
            time.sleep(0.8)
            self.arduino.clear_feedback()
            time.sleep(0.2)
        print("  [OK] Red feedback LEDs working")

        # Test serial communication
        print("\n[4/4] Testing serial communication...")
        self.arduino.start_stimulation()
        time.sleep(0.5)
        self.arduino.stop_stimulation()
        time.sleep(0.5)
        self.arduino.clear_feedback()
        print("  [OK] Serial communication working")

        return True

    def test_cyton(self) -> bool:
        """Test Cyton connection and EEG acquisition."""
        print("\n" + "="*60)
        print("CYTON EEG TEST")
        print("="*60)

        # Connect
        print("\n[1/3] Connecting to Cyton...")
        self.eeg_driver = BrainFlowDriver(self.config)
        self.eeg_driver.config.serial_port = self.cyton_port
        if not self.eeg_driver.connect():
            print("  [FAIL] Could not connect to Cyton")
            return False
        print(f"  [OK] Connected! Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        # Start streaming
        print("\n[2/3] Starting EEG stream...")
        if not self.eeg_driver.start_stream():
            print("  [FAIL] Could not start stream")
            return False
        print("  [OK] Stream started")
        time.sleep(1)  # Let buffer fill

        # Check data acquisition
        print("\n[3/3] Testing data acquisition (5 seconds)...")
        samples_collected = 0
        for i in range(5):
            data = self.eeg_driver.get_data()
            if data is not None and data.shape[1] > 0:
                samples_collected += data.shape[1]
                print(f"  Second {i+1}: Got {data.shape[1]} samples, "
                      f"shape={data.shape}, mean={np.mean(np.abs(data)):.1f} µV")
            else:
                print(f"  Second {i+1}: No data")
            time.sleep(1)

        print(f"\n  Total samples: {samples_collected}")
        print(f"  Expected: ~1250 (250 Hz * 5 seconds)")

        if samples_collected < 1000:
            print("  [FAIL] Not enough samples received")
            return False

        print("  [OK] Data acquisition working")
        return True

    def analyze_signal_quality(self, duration: int = 10) -> dict:
        """Analyze EEG signal quality in real-time.

        Args:
            duration: Duration in seconds

        Returns:
            Dictionary with signal quality metrics
        """
        print("\n" + "="*60)
        print("SIGNAL QUALITY ANALYSIS")
        print("="*60)
        print(f"\nAnalyzing EEG for {duration} seconds...")
        print("Filters enabled: 5-50 Hz bandpass, 60 Hz notch, smoothing")
        print("Keep still and relax...\n")

        metrics = {
            'mean': [[] for _ in range(8)],
            'std': [[] for _ in range(8)],
            'range': [[] for _ in range(8)],
            'bad_channels': set(),
            'flat_channels': set(),
            'noisy_channels': set(),
        }

        # Reset preprocessor for fresh start
        self.preprocessor.reset()

        start_time = time.time()
        while time.time() - start_time < duration:
            data = self.eeg_driver.get_data()
            if data is not None and data.shape[1] > 0:
                # Process filtered data
                filtered_data = self.preprocessor.process(data)

                # Collect samples (both raw and filtered)
                for ch in range(8):
                    self.signal_history[ch].extend(data[ch, :])
                    self.signal_history_filtered[ch].extend(filtered_data[ch, :])

                # Analyze every second
                if len(self.signal_history[0]) >= 250:
                    for ch in range(8):
                        samples = np.array(self.signal_history[ch])
                        metrics['mean'][ch].append(np.mean(samples))
                        metrics['std'][ch].append(np.std(samples))
                        metrics['range'][ch].append(np.ptp(samples))

                    # Check for issues
                    elapsed = int(time.time() - start_time)
                    print(f"\n[Second {elapsed}] - Raw vs Filtered Comparison")
                    print("-" * 100)
                    for ch in range(8):
                        # Raw data stats
                        raw_samples = np.array(self.signal_history[ch])
                        raw_mean = np.mean(raw_samples)
                        raw_std = np.std(raw_samples)
                        raw_ptp = np.ptp(raw_samples)

                        # Filtered data stats
                        filt_samples = np.array(self.signal_history_filtered[ch])
                        filt_mean = np.mean(filt_samples)
                        filt_std = np.std(filt_samples)
                        filt_ptp = np.ptp(filt_samples)

                        # Flags (check raw data for issues)
                        flags = []

                        # Check for flat line (dead channel)
                        if raw_std < 1.0 or raw_ptp < 5.0:
                            flags.append("FLAT")
                            metrics['flat_channels'].add(ch)

                        # Check for excessive noise
                        if raw_std > 100.0:
                            flags.append("NOISY")
                            metrics['noisy_channels'].add(ch)

                        # Check for DC offset
                        if abs(raw_mean) > 200.0:
                            flags.append("DC OFFSET")
                            metrics['bad_channels'].add(ch)

                        # Check for saturation
                        if raw_ptp > 500.0:
                            flags.append("SATURATED")
                            metrics['bad_channels'].add(ch)

                        status = " [" + ", ".join(flags) + "]" if flags else " [OK]"

                        # Display raw vs filtered
                        ch_name = ("Pz", "P3", "P4", "PO3", "PO4", "O1", "Oz", "O2")[ch]
                        print(f"  Ch{ch+1} ({ch_name}){status}")
                        print(f"    Raw:      mean={raw_mean:7.1f} µV, std={raw_std:5.1f} µV, range={raw_ptp:6.1f} µV")
                        print(f"    Filtered: mean={filt_mean:7.1f} µV, std={filt_std:5.1f} µV, range={filt_ptp:6.1f} µV")

                    # Clear history after analysis
                    for ch in range(8):
                        self.signal_history[ch].clear()
                        self.signal_history_filtered[ch].clear()

            time.sleep(0.01)

        # Summary
        print("\n" + "-"*60)
        print("SIGNAL QUALITY SUMMARY")
        print("-"*60)

        if len(metrics['flat_channels']) > 0:
            print(f"\n[WARNING] Flat channels detected: {sorted(metrics['flat_channels'])}")
            print("  Possible causes:")
            print("  - Electrode not making good contact")
            print("  - Dry electrode (needs gel/water)")
            print("  - Broken electrode wire")

        if len(metrics['noisy_channels']) > 0:
            print(f"\n[WARNING] Noisy channels detected: {sorted(metrics['noisy_channels'])}")
            print("  Possible causes:")
            print("  - Poor electrode contact")
            print("  - Muscle artifact (EMG)")
            print("  - Electrical interference (50/60 Hz)")

        if len(metrics['bad_channels']) > 0:
            print(f"\n[WARNING] Bad channels detected: {sorted(metrics['bad_channels'])}")
            print("  Possible causes:")
            print("  - Incorrect reference connection")
            print("  - Bias/ground electrode issues")
            print("  - Amplifier saturation")

        # Check for missing reference
        all_bad = metrics['flat_channels'].union(metrics['bad_channels'])
        if len(all_bad) >= 6:
            print("\n[CRITICAL] Most channels are bad!")
            print("  Likely cause: Reference electrode (SRB2) is not connected!")
            print("  Solution: Connect SRB2 to earlobe or mastoid")

        if len(all_bad) == 0 and len(metrics['noisy_channels']) == 0:
            print("\n[OK] All channels look good!")

        return metrics

    def run_full_test(self) -> bool:
        """Run complete hardware test sequence."""
        print("\n" + "="*70)
        print("SSVEP BCI HARDWARE TROUBLESHOOTING")
        print("="*70)

        try:
            # Test Arduino
            arduino_ok = self.test_arduino()

            # Test Cyton
            cyton_ok = self.test_cyton()

            if not cyton_ok:
                return False

            # Signal quality analysis
            self.analyze_signal_quality(duration=10)

            return True

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            return False

        finally:
            self.cleanup()

    def cleanup(self):
        """Disconnect all hardware."""
        print("\n" + "="*60)
        print("Cleaning up...")
        if self.eeg_driver:
            self.eeg_driver.stop_stream()
            self.eeg_driver.disconnect()
        if self.arduino and self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()
            self.arduino.disconnect()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="SSVEP BCI Hardware Troubleshooting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--cyton', '-c',
        type=str,
        default=None,
        help='Cyton serial port (e.g., COM3)'
    )

    parser.add_argument(
        '--arduino', '-a',
        type=str,
        default=None,
        help='Arduino serial port (e.g., COM4)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=10,
        help='Signal analysis duration in seconds (default: 10)'
    )

    args = parser.parse_args()

    # Create troubleshooter
    troubleshooter = HardwareTroubleshooter(
        cyton_port=args.cyton,
        arduino_port=args.arduino
    )

    # Run tests
    success = troubleshooter.run_full_test()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
