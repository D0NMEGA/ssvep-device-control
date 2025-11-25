#!/usr/bin/env python3
"""
Real-time SSVEP Monitoring and Optimization Tool

Comprehensive real-time system for:
- Hardware validation (EEG + Arduino)
- Live signal quality monitoring (raw vs filtered)
- Real-time SSVEP classification with all correlation scores
- Calibration validation (verify user is looking at correct LED)
- Performance optimization and data logging

Usage:
    # With calibration templates
    python run_realtime_monitor.py --subject YOUR_NAME --cyton COM3 --arduino COM4

    # Without calibration (standard CCA)
    python run_realtime_monitor.py --cyton COM3 --arduino COM4

    # Adjust window size for higher accuracy
    python run_realtime_monitor.py --subject YOUR_NAME --cyton COM3 --window-ms 500
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig, create_config
from models.eeg_buffer import EEGBuffer
from models.preprocessor import SSVEPPreprocessor
from models.cca_decoder import SSVEPDecoder
from models.template_cca import TemplateCCADecoder, TemplateCCAConfig
from models.calibration import CalibrationCollector
from drivers.brainflow_driver import BrainFlowDriver
from drivers.arduino_controller import ArduinoController
from drivers.data_logger import DataLogger


class RealtimeMonitor:
    """Real-time SSVEP monitoring and validation system."""

    def __init__(
        self,
        subject: str = None,
        cyton_port: str = None,
        arduino_port: str = None,
        window_ms: float = None,
        confidence_threshold: float = 0.50,
        margin_threshold: float = 0.10,
        log_data: bool = True
    ):
        self.subject = subject
        self.cyton_port = cyton_port
        self.arduino_port = arduino_port
        self.log_data = log_data

        # Create config with optional custom window size
        config_kwargs = {
            'confidence_threshold': confidence_threshold,
            'margin_threshold': margin_threshold
        }

        if window_ms is not None:
            # Convert window duration to samples
            samples = int((window_ms / 1000.0) * 250)  # 250 Hz
            config_kwargs['window_samples'] = samples
            print(f"Custom window size: {samples} samples ({window_ms} ms)")

        self.config = create_config(**config_kwargs)

        # Hardware drivers
        self.eeg_driver = None
        self.arduino = None
        self.preprocessor = SSVEPPreprocessor(self.config)

        # EEG buffer
        self.buffer = EEGBuffer(self.config)

        # Decoder (standard or template-based)
        self.decoder = None
        self.using_templates = False

        # Load calibration if available
        if subject:
            self._load_calibration()

        # Create standard decoder if no templates
        if self.decoder is None:
            print("Using standard CCA (no calibration templates)")
            self.decoder = SSVEPDecoder(self.config)

        # Logger
        self.logger = None
        if self.log_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_prefix = f"realtime_monitor_{subject or 'unknown'}_{timestamp}"
            self.logger = ClassificationLogger(
                self.config,
                log_prefix=log_prefix,
                enabled=True
            )

        # Real-time statistics
        self.stats = {
            'windows_processed': 0,
            'predictions': {freq: 0 for freq in self.config.target_frequencies},
            'correct_predictions': 0,
            'start_time': None,
            'raw_signal_quality': deque(maxlen=50),  # Last 50 windows
            'filtered_signal_quality': deque(maxlen=50),
            'correlation_history': {freq: deque(maxlen=100) for freq in self.config.target_frequencies}
        }

        # LED mapping for display
        self.led_positions = {
            8.57: "Far Left",
            10.0: "Center-Left",
            12.0: "Center-Right",
            15.0: "Far Right"
        }

    def _load_calibration(self):
        """Load calibration templates for subject."""
        sessions = CalibrationCollector.get_subject_sessions(self.subject)
        if not sessions:
            print(f"No calibration found for subject '{self.subject}'")
            print("Will use standard CCA")
            return

        # Combine all sessions for incremental learning
        try:
            cal_data = CalibrationCollector.get_combined_calibration(self.subject)
            if cal_data is None:
                print("Could not load calibration")
                return

            # Create template-based decoder
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
            print(f"✓ Loaded calibration templates for {self.subject}")
            print(f"  Sessions combined: {len(sessions)}")
            print(f"  Total epochs: {cal_data.epochs.shape[0]}")
            print(f"  Using template-based CCA (70% template, 30% standard)")

        except Exception as e:
            print(f"Error loading calibration: {e}")
            print("Will use standard CCA")

    def connect_hardware(self) -> bool:
        """Connect to Cyton and Arduino."""
        print("\n" + "="*70)
        print("HARDWARE CONNECTION")
        print("="*70)

        # Connect Arduino
        if self.arduino_port:
            print("\n[1/2] Connecting to Arduino...")
            self.arduino = ArduinoController(port=self.arduino_port)
            if not self.arduino.connect():
                print("  [FAIL] Could not connect to Arduino")
                return False
            print(f"  [OK] Arduino connected on {self.arduino.port}")
        else:
            print("\n[1/2] Arduino: Not specified (skipping)")

        # Connect Cyton
        print("\n[2/2] Connecting to Cyton...")
        self.eeg_driver = BrainFlowDriver(self.config)
        self.eeg_driver.config.serial_port = self.cyton_port

        if not self.eeg_driver.connect():
            print("  [FAIL] Could not connect to Cyton")
            return False

        print(f"  [OK] Cyton connected!")
        print(f"  Sampling rate: {self.eeg_driver.sampling_rate} Hz")

        # Start stream
        if not self.eeg_driver.start_stream():
            print("  [FAIL] Could not start stream")
            return False

        print("  [OK] Stream started")

        return True

    def test_signal_quality(self, duration: int = 5):
        """Quick signal quality test."""
        print("\n" + "="*70)
        print("SIGNAL QUALITY TEST")
        print("="*70)
        print(f"Testing for {duration} seconds...")
        print("Filters: 5-50 Hz bandpass, 60 Hz notch, smoothing\n")

        self.preprocessor.reset()

        signal_stats = {
            'raw_mean': [],
            'raw_std': [],
            'filt_mean': [],
            'filt_std': []
        }

        start_time = time.time()
        while time.time() - start_time < duration:
            data = self.eeg_driver.get_data()
            if data is not None and data.shape[1] > 0:
                # Process
                filtered = self.preprocessor.process(data)

                # Stats across all channels
                signal_stats['raw_mean'].append(np.mean(np.abs(data)))
                signal_stats['raw_std'].append(np.std(data))
                signal_stats['filt_mean'].append(np.mean(np.abs(filtered)))
                signal_stats['filt_std'].append(np.std(filtered))

            time.sleep(0.01)

        # Summary
        print("Signal Quality Summary (all channels):")
        print(f"  Raw data:      mean={np.mean(signal_stats['raw_mean']):.1f} µV, "
              f"std={np.mean(signal_stats['raw_std']):.1f} µV")
        print(f"  Filtered data: mean={np.mean(signal_stats['filt_mean']):.1f} µV, "
              f"std={np.mean(signal_stats['filt_std']):.1f} µV")

        # Check if signal looks reasonable
        filt_mean = np.mean(signal_stats['filt_mean'])
        if filt_mean < 1.0:
            print("\n  [WARNING] Signal amplitude very low - check electrode contact")
        elif filt_mean > 100.0:
            print("\n  [WARNING] Signal amplitude very high - check for artifacts")
        else:
            print("\n  [OK] Signal quality looks good")

    def run_realtime_classification(self, duration: int = 60):
        """Run real-time classification with live monitoring."""
        print("\n" + "="*70)
        print("REAL-TIME SSVEP CLASSIFICATION")
        print("="*70)
        print(f"Duration: {duration} seconds")
        print(f"Window: {self.config.window_samples} samples "
              f"({self.config.window_duration_ms:.1f} ms)")
        print(f"Thresholds: confidence={self.config.confidence_threshold:.2f}, "
              f"margin={self.config.margin_threshold:.2f}")
        if self.using_templates:
            print(f"Decoder: Template-based CCA (subject: {self.subject})")
        else:
            print(f"Decoder: Standard CCA")
        print("\nLook at one LED and hold your gaze steady...")
        print("="*70 + "\n")

        # Start Arduino LEDs
        if self.arduino and self.arduino.is_connected:
            self.arduino.start_stimulation()
            print("✓ Arduino LEDs started\n")

        # Reset everything
        self.buffer.reset()
        self.decoder.reset()
        self.preprocessor.reset()
        self.stats['start_time'] = time.time()
        self.stats['windows_processed'] = 0

        # Display header
        self._print_display_header()

        last_display_time = time.time()
        display_interval = 0.5  # Update display every 500ms

        try:
            while time.time() - self.stats['start_time'] < duration:
                # Get new data
                data = self.eeg_driver.get_data()
                if data is None or data.shape[1] == 0:
                    time.sleep(0.001)
                    continue

                # Preprocess
                filtered = self.preprocessor.process(data)

                # Add to buffer
                for i in range(data.shape[1]):
                    self.buffer.add_sample(filtered[:, i])

                    # Process window if ready
                    if self.buffer.is_ready():
                        window = self.buffer.get_window()
                        result = self.decoder.step(window)

                        # Update stats
                        self._update_stats(result, data[:, i], filtered[:, i])

                        # Log
                        if self.logger:
                            self.logger.log_classification(result)

                        # Update display periodically
                        if time.time() - last_display_time >= display_interval:
                            self._print_live_update(result)
                            last_display_time = time.time()

                        # Step buffer
                        self.buffer.step()

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        finally:
            # Stop Arduino
            if self.arduino and self.arduino.is_connected:
                self.arduino.stop_stimulation()
                self.arduino.clear_feedback()

            # Show final summary
            self._print_final_summary()

    def _update_stats(self, result, raw_sample, filtered_sample):
        """Update real-time statistics."""
        self.stats['windows_processed'] += 1

        # Signal quality
        self.stats['raw_signal_quality'].append(np.std(raw_sample))
        self.stats['filtered_signal_quality'].append(np.std(filtered_sample))

        # Correlation history
        for freq, corr in result.correlations.items():
            self.stats['correlation_history'][freq].append(corr)

        # Predictions
        if result.committed_prediction is not None:
            self.stats['predictions'][result.committed_prediction] += 1

    def _print_display_header(self):
        """Print display header."""
        print(f"{'Time':>6} │ {'Win#':>5} │ ", end="")
        for freq in self.config.target_frequencies:
            pos = self.led_positions[freq]
            print(f"{freq:.2f}Hz ({pos[:4]}) │ ", end="")
        print(f"{'Best':>6} │ {'Margin':>6} │ {'Pred':>6} │ {'Status':>10}")
        print("─" * 120)

    def _print_live_update(self, result):
        """Print live classification update."""
        elapsed = time.time() - self.stats['start_time']

        # Time and window count
        print(f"{elapsed:6.1f}s│ {self.stats['windows_processed']:5d} │ ", end="")

        # Correlation for each frequency
        for freq in self.config.target_frequencies:
            corr = result.correlations[freq]
            # Highlight if this is the best
            if freq == max(result.correlations, key=result.correlations.get):
                print(f"\033[1m{corr:6.3f}\033[0m         │ ", end="")
            else:
                print(f"{corr:6.3f}         │ ", end="")

        # Best frequency, margin, prediction
        best_freq = max(result.correlations, key=result.correlations.get)
        print(f"{best_freq:6.2f} │ {result.margin:6.3f} │ ", end="")

        if result.committed_prediction:
            print(f"{result.committed_prediction:6.2f} │ ", end="")
        else:
            print(f"{'---':>6} │ ", end="")

        # Status
        if result.committed_prediction:
            pos = self.led_positions[result.committed_prediction]
            print(f"\033[92m{pos:>10}\033[0m", end="")
        elif result.confidence_met and not result.margin_met:
            print(f"{'Low margin':>10}", end="")
        elif not result.confidence_met:
            print(f"{'Low conf':>10}", end="")
        else:
            print(f"{'Voting...':>10}", end="")

        print()  # Newline

    def _print_final_summary(self):
        """Print final performance summary."""
        elapsed = time.time() - self.stats['start_time']

        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        print(f"\nSession Duration: {elapsed:.1f} seconds")
        print(f"Windows Processed: {self.stats['windows_processed']}")
        print(f"Processing Rate: {self.stats['windows_processed']/elapsed:.1f} windows/second")

        # Prediction distribution
        print("\nPrediction Distribution:")
        total_predictions = sum(self.stats['predictions'].values())
        for freq in self.config.target_frequencies:
            count = self.stats['predictions'][freq]
            pct = (count / total_predictions * 100) if total_predictions > 0 else 0
            pos = self.led_positions[freq]
            print(f"  {freq:.2f} Hz ({pos:12s}): {count:4d} ({pct:5.1f}%)")

        # Average correlations
        print("\nAverage Correlations:")
        for freq in self.config.target_frequencies:
            if len(self.stats['correlation_history'][freq]) > 0:
                avg_corr = np.mean(self.stats['correlation_history'][freq])
                max_corr = np.max(self.stats['correlation_history'][freq])
                pos = self.led_positions[freq]
                print(f"  {freq:.2f} Hz ({pos:12s}): avg={avg_corr:.3f}, max={max_corr:.3f}")

        # Signal quality
        if len(self.stats['filtered_signal_quality']) > 0:
            avg_noise = np.mean(self.stats['filtered_signal_quality'])
            print(f"\nAverage Signal Std Dev: {avg_noise:.1f} µV")

        # Recommendations
        print("\n" + "─"*70)
        self._print_recommendations()

        # Log file location
        if self.logger:
            print(f"\nData logged to: {self.logger.filepath}")

    def _print_recommendations(self):
        """Print performance optimization recommendations."""
        print("RECOMMENDATIONS:")

        # Check if using templates
        if not self.using_templates and self.subject:
            print(f"  → Run calibration to create templates for {self.subject}")
            print(f"     Command: python run_calibration.py --subject {self.subject} --cyton {self.cyton_port}")

        # Check correlation levels
        avg_corrs = []
        for freq in self.config.target_frequencies:
            if len(self.stats['correlation_history'][freq]) > 0:
                avg_corrs.append(np.mean(self.stats['correlation_history'][freq]))

        if len(avg_corrs) > 0:
            overall_avg = np.mean(avg_corrs)
            if overall_avg < 0.4:
                print(f"  → Low correlations (avg={overall_avg:.3f})")
                print(f"     - Check electrode contact (especially occipital channels)")
                print(f"     - Reduce eye movements and blinks")
                print(f"     - Ensure good fixation on LEDs")
                if self.config.window_samples < 100:
                    print(f"     - Try larger window: --window-ms 400 or 500")
            elif overall_avg < 0.55:
                print(f"  → Moderate correlations (avg={overall_avg:.3f})")
                if not self.using_templates:
                    print(f"     - Run calibration for personalized templates")
                if self.config.window_samples < 100:
                    print(f"     - Consider larger window for higher accuracy")
            else:
                print(f"  ✓ Good correlations (avg={overall_avg:.3f})")

        # Check if predictions are concentrated
        if sum(self.stats['predictions'].values()) > 10:
            max_pred_count = max(self.stats['predictions'].values())
            total_preds = sum(self.stats['predictions'].values())
            concentration = max_pred_count / total_preds

            if concentration > 0.8:
                dominant_freq = max(self.stats['predictions'], key=self.stats['predictions'].get)
                print(f"  ✓ Stable predictions ({concentration*100:.0f}% at {dominant_freq} Hz)")
            elif concentration < 0.5:
                print(f"  → Predictions scattered across frequencies")
                print(f"     - Lower thresholds: --confidence 0.45 --margin 0.08")
                print(f"     - Or increase window size for better resolution")

    def cleanup(self):
        """Disconnect all hardware."""
        print("\n" + "="*70)
        print("Cleaning up...")

        if self.eeg_driver:
            self.eeg_driver.stop_stream()
            self.eeg_driver.disconnect()

        if self.arduino and self.arduino.is_connected:
            self.arduino.stop_stimulation()
            self.arduino.clear_feedback()
            self.arduino.disconnect()

        if self.logger:
            self.logger.close()

        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time SSVEP Monitoring and Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With calibration templates
  python run_realtime_monitor.py --subject Donovan_Santine --cyton COM3 --arduino COM4

  # Without calibration
  python run_realtime_monitor.py --cyton COM3 --arduino COM4

  # Larger window for higher accuracy
  python run_realtime_monitor.py --subject Donovan_Santine --cyton COM3 --window-ms 500

  # Quick signal quality test only
  python run_realtime_monitor.py --cyton COM3 --test-only
        """
    )

    parser.add_argument(
        '--subject', '-s',
        type=str,
        help='Subject ID for loading calibration templates'
    )

    parser.add_argument(
        '--cyton', '-c',
        type=str,
        required=True,
        help='Cyton serial port (e.g., COM3)'
    )

    parser.add_argument(
        '--arduino', '-a',
        type=str,
        help='Arduino serial port (e.g., COM4)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Monitoring duration in seconds (default: 60)'
    )

    parser.add_argument(
        '--window-ms',
        type=float,
        help='Analysis window duration in milliseconds (default: 252ms). Try 400-500ms for higher accuracy.'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.50,
        help='Confidence threshold (default: 0.50)'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=0.10,
        help='Margin threshold (default: 0.10)'
    )

    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable data logging'
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run signal quality test, no classification'
    )

    args = parser.parse_args()

    # Create monitor
    monitor = RealtimeMonitor(
        subject=args.subject,
        cyton_port=args.cyton,
        arduino_port=args.arduino,
        window_ms=args.window_ms,
        confidence_threshold=args.confidence,
        margin_threshold=args.margin,
        log_data=not args.no_log
    )

    try:
        # Connect hardware
        if not monitor.connect_hardware():
            return 1

        # Wait for buffer to fill
        print("\nWaiting for data buffer to fill...")
        time.sleep(2)

        # Run signal quality test
        monitor.test_signal_quality(duration=5)

        if args.test_only:
            print("\nSignal quality test complete (--test-only mode)")
        else:
            # Run real-time classification
            monitor.run_realtime_classification(duration=args.duration)

        return 0

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        monitor.cleanup()


if __name__ == "__main__":
    sys.exit(main())
