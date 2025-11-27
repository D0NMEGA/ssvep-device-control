#!/usr/bin/env python3
"""
SSVEP BCI using Reference TRCA Implementation (Nakanishi et al.)

This script uses the exact reference implementation from:
M. Nakanishi et al. (2018) - Enhanced TRCA for SSVEP-based BCIs

Adapted for:
- OpenBCI Cyton (8 channels)
- Arduino Mega LED control
- Real-time streaming classification

Usage:
    python mnakanishi.py models/alice_trca.pkl
    python mnakanishi.py models/model.pkl --arduino COM3 --cyton COM4
    python mnakanishi.py models/model.pkl --synthetic --num-fbs 5
"""

import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from numpy import linalg as la
from scipy.stats import pearsonr
from scipy import signal

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.drivers import BrainFlowDriver, ArduinoController, SyntheticSSVEPDriver
from ssvep_bci.buffer import EEGBuffer


# =============================================================================
# REFERENCE TRCA IMPLEMENTATION (Nakanishi et al. 2018)
# =============================================================================

def train_trca(eeg, fs, num_fbs):
    """Train TRCA model - REFERENCE IMPLEMENTATION.

    Args:
        eeg: (num_classes, num_channels, num_samples, num_trials)
        fs: Sampling frequency
        num_fbs: Number of filter banks

    Returns:
        weight: (num_classes, num_channels, num_fbs)
        template: (num_classes, num_channels, num_samples, num_fbs)
    """
    num_targs = eeg.shape[0]
    num_chans = eeg.shape[1]
    num_smpls = eeg.shape[2]

    weight = np.zeros((num_targs, num_chans, num_fbs))
    train_temp = np.zeros((num_targs, num_chans, num_smpls, num_fbs))

    for targ_i in range(num_targs):
        for fb_i in range(num_fbs):
            traindata = filterbank(eeg[targ_i, :, :, :], fs, fb_i)
            w_tmp = ftrca(traindata)
            weight[targ_i, :, fb_i] = w_tmp[:, 0]  # First component
            train_temp[targ_i, :, :, fb_i] = np.average(traindata, axis=2)

    return weight, train_temp


def train_trca_variable(eeg_by_class, fs, num_fbs):
    """Train TRCA model with varying trial counts per class.

    Args:
        eeg_by_class: List of (num_channels, num_samples, num_trials) per class
        fs: Sampling frequency
        num_fbs: Number of filter banks

    Returns:
        weight: (num_classes, num_channels, num_fbs)
        template: (num_classes, num_channels, num_samples, num_fbs)
    """
    num_targs = len(eeg_by_class)
    num_chans = eeg_by_class[0].shape[0]
    num_smpls = eeg_by_class[0].shape[1]

    weight = np.zeros((num_targs, num_chans, num_fbs))
    train_temp = np.zeros((num_targs, num_chans, num_smpls, num_fbs))

    for targ_i in range(num_targs):
        for fb_i in range(num_fbs):
            traindata = filterbank(eeg_by_class[targ_i], fs, fb_i)
            w_tmp = ftrca(traindata)
            weight[targ_i, :, fb_i] = w_tmp[:, 0]  # First component
            train_temp[targ_i, :, :, fb_i] = np.average(traindata, axis=2)

    return weight, train_temp


def test_trca(eeg, weight, template, fs, num_fbs, is_ensemble=True):
    """Test TRCA model - REFERENCE IMPLEMENTATION.

    Args:
        eeg: (num_trials, num_channels, num_samples) - test data
        weight: (num_classes, num_channels, num_fbs)
        template: (num_classes, num_channels, num_samples, num_fbs)
        fs: Sampling frequency
        num_fbs: Number of filter banks
        is_ensemble: Use ensemble weights (True = better performance)

    Returns:
        predictions: (num_trials,)
        correlations: (num_trials, num_classes)
    """
    if eeg.ndim == 2:
        # Single trial: (num_channels, num_samples)
        eeg = eeg[np.newaxis, :, :]

    num_trials = eeg.shape[0]
    num_targs = template.shape[0]

    # Filter bank coefficients (a^(-1.25) + 0.25 as per Nakanishi 2018)
    fb_coefs = pow(np.array(range(1, num_fbs + 1, 1)), -1.25) + 0.25

    outclass = np.zeros(num_trials)
    all_corr = np.zeros((num_trials, num_targs))

    for trial_i in range(num_trials):
        corr = np.zeros((num_targs, num_fbs))

        for fb_i in range(num_fbs):
            testdata = filterbank(eeg[trial_i, :, :], fs, fb_i)

            for class_i in range(num_targs):
                traindata = template[class_i, :, :, fb_i]

                if is_ensemble:
                    # Ensemble: use ALL class weights
                    w = weight[:, :, fb_i]  # (num_classes, num_channels)
                    # Average correlation across all class weights
                    corr_sum = 0
                    for w_i in range(num_targs):
                        test_proj = np.dot(w[w_i, :], testdata).flatten()
                        train_proj = np.dot(w[w_i, :], traindata).flatten()

                        r, _ = pearsonr(test_proj, train_proj)
                        corr_sum += r
                    corr[class_i, fb_i] = corr_sum / num_targs
                else:
                    # Individual: use only class-specific weight
                    w = weight[class_i, :, fb_i]
                    test_proj = np.dot(w, testdata).flatten()
                    train_proj = np.dot(w, traindata).flatten()

                    corr[class_i, fb_i], _ = pearsonr(test_proj, train_proj)

        # Weighted sum across filter banks
        rho = np.dot(fb_coefs, corr.T)
        outclass[trial_i] = np.argmax(rho)
        all_corr[trial_i, :] = rho

    return outclass.astype(int), all_corr


def ftrca(x):
    """Task-Related Component Analysis - REFERENCE IMPLEMENTATION.

    Args:
        x: (num_channels, num_samples, num_trials)

    Returns:
        V: (num_channels, num_components) - spatial filters
    """
    num_trials = x.shape[2]

    # Demean each trial
    for trial_i in range(num_trials):
        x1 = x[:, :, trial_i]
        x[:, :, trial_i] = x1 - x1.mean(axis=1, keepdims=True)

    # Inter-trial covariance S
    SX = np.sum(x, axis=2)
    S = np.dot(SX, SX.T)

    # Total covariance Q
    QX = x.reshape(x.shape[0], -1)
    Q = np.dot(QX, QX.T)

    # Solve generalized eigenvalue problem
    W, V = la.eig(np.dot(la.inv(Q), S))

    # Sort by eigenvalue (descending)
    idx = W.argsort()[::-1]
    V = V[:, idx]

    return V


def filterbank(x, fs, fb_i):
    """Apply filter bank - REFERENCE IMPLEMENTATION.

    Uses 10 filter banks as per Nakanishi et al. 2018.

    Args:
        x: (num_channels, num_samples) or (num_channels, num_samples, num_trials)
        fs: Sampling frequency
        fb_i: Filter bank index (0-9)

    Returns:
        y: Filtered data with same shape as x
    """
    num_chans = x.shape[0]
    num_smpls = x.shape[1]
    if x.ndim > 2:
        num_trials = x.shape[2]
    else:
        num_trials = 1

    nyq = fs / 2

    # Reference filter bank design
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

    Wp = [passband[fb_i] / nyq, 90 / nyq]
    Ws = [stopband[fb_i] / nyq, 100 / nyq]

    gpass = 3
    gstop = 40
    Rp = 0.5

    [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
    [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')

    if num_trials == 1:
        y = np.zeros((num_chans, num_smpls))
        for ch_i in range(num_chans):
            y[ch_i, :] = signal.filtfilt(B, A, x[ch_i, :])
    else:
        y = np.zeros((num_chans, num_smpls, num_trials))
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = signal.filtfilt(B, A, x[ch_i, :, trial_i])

    return y


# =============================================================================
# REAL-TIME BCI WITH REFERENCE TRCA
# =============================================================================

class NakanishiBCI:
    """Real-time SSVEP BCI using Nakanishi reference TRCA."""

    def __init__(self, model_path: str, arduino_port=None, cyton_port=None,
                 use_synthetic=False, num_fbs=10, is_ensemble=True):
        # Load model
        print(f"Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.config: SSVEPConfig = model_data['config']

        # Extract training data and retrain with reference TRCA
        if 'training_data' in model_data:
            print("  Retraining with reference TRCA implementation...")
            X_train = model_data['training_data']['X']
            y_train = model_data['training_data']['y']

            # Convert to reference format: list of (num_channels, num_samples, num_trials) per class
            num_classes = len(self.config.target_frequencies)
            num_channels = X_train[0].shape[0]
            num_samples = X_train[0].shape[1]

            # Group by class
            eeg_by_class = []
            for class_i in range(num_classes):
                class_trials = [X_train[i] for i in range(len(X_train)) if y_train[i] == class_i]
                if len(class_trials) == 0:
                    raise ValueError(f"No training trials for class {class_i}")
                # Stack: (num_channels, num_samples, num_trials)
                eeg_by_class.append(np.stack(class_trials, axis=2))

            # Print per-class trial counts
            print(f"  Training data per class:")
            for class_i in range(num_classes):
                print(f"    Class {class_i}: {eeg_by_class[class_i].shape[2]} trials")
            print(f"  Using {num_fbs} filter banks")
            print(f"  Ensemble mode: {is_ensemble}")

            # Train reference TRCA (handles varying trial counts)
            self.weight, self.template = train_trca_variable(eeg_by_class, self.config.fs, num_fbs)

            # Update config window size to match template
            template_samples = self.template.shape[2]
            template_duration_ms = (template_samples / self.config.fs) * 1000
            self.config.window_samples = template_samples
            print(f"  Template window: {template_duration_ms:.0f} ms ({template_samples} samples)")

        else:
            print("\nERROR: This model doesn't contain raw training data.")
            print("\nThe reference TRCA implementation needs to retrain from scratch")
            print("using the raw calibration data.")
            print("\nSolutions:")
            print("  1. Use a model from calibration.py (includes training_data)")
            print("  2. Use a model from online_training.py (includes training_data)")
            print("  3. Re-run optimize_accuracy.py (now saves training_data)")
            print("\nFor now, use classify.py with this model instead:")
            print(f"  python classify.py {model_path}")
            raise ValueError("Model must contain 'training_data' for reference TRCA")

        self.num_fbs = num_fbs
        self.is_ensemble = is_ensemble

        # Initialize components
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
        print("Reference TRCA BCI (Nakanishi et al. 2018)")
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
        """Run real-time classification."""
        print("\n" + "=" * 60)
        print("Real-Time Classification (Reference TRCA)")
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
                    self.buffer.append(data)

                # Process when ready
                if self.buffer.ready():
                    window_count += 1

                    window = self.buffer.get_window()  # (n_channels, n_samples)

                    # Predict using reference TRCA
                    pred_idx, correlations = test_trca(
                        window,
                        self.weight,
                        self.template,
                        self.config.fs,
                        self.num_fbs,
                        self.is_ensemble
                    )

                    pred_idx = pred_idx[0]  # Single prediction
                    freq = self.config.target_frequencies[pred_idx]
                    max_corr = correlations[0, pred_idx]

                    # Get margin
                    sorted_idx = np.argsort(correlations[0])[::-1]
                    if len(sorted_idx) > 1:
                        margin = correlations[0, sorted_idx[0]] - correlations[0, sorted_idx[1]]
                    else:
                        margin = max_corr

                    # Update feedback if prediction changed
                    if pred_idx != self.last_prediction:
                        print(f"[{window_count:4d}] Prediction: {freq:.2f} Hz "
                              f"(corr={max_corr:.4f}, margin={margin:.4f})")

                        if self.arduino.is_connected:
                            self.arduino.show_feedback(freq)

                        self.last_prediction = pred_idx

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
    parser = argparse.ArgumentParser(description="Reference TRCA BCI (Nakanishi et al.)")
    parser.add_argument("model", type=str,
                       help="Path to trained model (.pkl file)")
    parser.add_argument("--arduino", type=str, default=None,
                       help="Arduino serial port (auto-detect if not specified)")
    parser.add_argument("--cyton", type=str, default=None,
                       help="Cyton serial port (auto-detect if not specified)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic EEG for testing")
    parser.add_argument("--num-fbs", type=int, default=10,
                       help="Number of filter banks (default: 10, max: 10)")
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Disable ensemble mode (use individual weights)")

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    # Validate num_fbs
    if args.num_fbs < 1 or args.num_fbs > 10:
        print("Error: num_fbs must be between 1 and 10")
        return 1

    # Create BCI
    bci = NakanishiBCI(
        model_path=args.model,
        arduino_port=args.arduino,
        cyton_port=args.cyton,
        use_synthetic=args.synthetic,
        num_fbs=args.num_fbs,
        is_ensemble=not args.no_ensemble
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
