#!/usr/bin/env python3
"""
Accuracy Optimization Script

Tests multiple approaches to maximize SSVEP-BCI accuracy:
1. Quality-based trial filtering (SNR threshold)
2. Different window lengths
3. Reference TRCA implementation vs current
4. Per-session vs batch learning
5. Different filter bank configurations

Supports single-subject and multi-subject ensemble models.

Usage:
    python optimize_accuracy.py --subject donny
    python optimize_accuracy.py --subject logan
    python optimize_accuracy.py --ensemble  # Multi-subject ensemble (donny + logan)
    python optimize_accuracy.py --subject all  # Same as --ensemble

Outputs best configuration for both individual sessions and batch learning.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from datetime import datetime

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import extract_trials_from_xdf, OfflinePreprocessor
from ssvep_bci.trca import TRCA


def filter_trials_by_quality(X, y, config, snr_threshold=1.0, verbose=True):
    """Remove trials with weak SSVEP response based on FFT SNR.

    Args:
        X: Trials with shape (n_trials, n_channels, n_samples)
        y: Labels with shape (n_trials,)
        config: SSVEPConfig instance
        snr_threshold: Minimum SNR to keep trial (default: 1.0)
        verbose: Print filtering statistics

    Returns:
        (X_filtered, y_filtered, snr_values)
    """
    kept_trials = []
    kept_labels = []
    snr_values = []

    oz_idx = 6  # Oz channel index (most responsive to occipital SSVEP)
    baseline_samples = 125  # 0.5s baseline at 250Hz

    for trial_idx in range(len(X)):
        trial = X[trial_idx]
        true_freq = config.target_frequencies[y[trial_idx]]

        # FFT on Oz channel (skip baseline period)
        oz_data = trial[oz_idx, baseline_samples:]
        freqs = np.fft.rfftfreq(len(oz_data), 1/config.fs)
        fft_mag = np.abs(np.fft.rfft(oz_data))

        # Power at target frequency
        freq_idx = np.argmin(np.abs(freqs - true_freq))
        target_power = fft_mag[freq_idx]

        # Background noise (average of neighboring frequencies)
        noise_indices = [freq_idx-2, freq_idx-1, freq_idx+1, freq_idx+2]
        valid_noise_idx = [i for i in noise_indices if 0 <= i < len(fft_mag)]
        background = np.mean([fft_mag[i] for i in valid_noise_idx])

        snr = target_power / background if background > 0 else 0
        snr_values.append(snr)

        # Keep only trials with SNR above threshold
        if snr >= snr_threshold:
            kept_trials.append(trial)
            kept_labels.append(y[trial_idx])

    if verbose:
        print(f"  Quality filtering (SNR >= {snr_threshold}):")
        print(f"    Kept: {len(kept_trials)}/{len(X)} trials ({100*len(kept_trials)/len(X):.1f}%)")
        print(f"    SNR range: {np.min(snr_values):.2f} - {np.max(snr_values):.2f}")
        print(f"    SNR mean: {np.mean(snr_values):.2f} ± {np.std(snr_values):.2f}")

    return np.array(kept_trials), np.array(kept_labels), np.array(snr_values)


def evaluate_loocv(X, y, config, filter_quality=False, snr_threshold=1.0):
    """Leave-One-Out Cross-Validation for small datasets.

    Args:
        X: Preprocessed trials (n_trials, n_channels, n_samples, n_filterbanks)
        y: Labels (n_trials,)
        config: SSVEPConfig instance
        filter_quality: Whether to apply quality filtering
        snr_threshold: SNR threshold for quality filtering

    Returns:
        (accuracy, n_correct, n_total)
    """
    n_total = len(X)
    correct = 0

    for loocv_i in range(n_total):
        # Train on all except one
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[loocv_i] = False

        X_train = X[train_mask]
        y_train = y[train_mask]

        # Apply quality filtering to training data only
        if filter_quality:
            # Need to remove filterbank dimension for quality filtering
            X_train_3d = X_train[:, :, :, 0]  # Use first filterbank for SNR check
            X_train_filtered, y_train_filtered, _ = filter_trials_by_quality(
                X_train_3d, y_train, config, snr_threshold, verbose=False
            )

            # Reapply preprocessing to filtered trials
            preprocessor = OfflinePreprocessor(config)
            X_train = preprocessor.process(X_train_filtered)
            y_train = y_train_filtered

        X_test = X[loocv_i:loocv_i+1]
        y_test = y[loocv_i]

        # Train and predict
        trca = TRCA(config, n_components=1)
        trca.fit(X_train, y_train)

        y_pred, _ = trca.predict_with_correlation(X_test)

        if y_pred == y_test:
            correct += 1

    accuracy = correct / n_total
    return accuracy, correct, n_total


def test_configuration(subjects, config: SSVEPConfig, filter_quality=False,
                       snr_threshold=1.0, description=""):
    """Test a specific configuration on all calibration data.

    Args:
        subjects: Single subject name (str) or list of subject names for ensemble
        config: SSVEPConfig object
        filter_quality: Whether to filter trials by quality
        snr_threshold: SNR threshold for quality filtering
        description: Description of this test

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"{'='*70}")

    # Convert single subject to list
    if isinstance(subjects, str):
        subjects = [subjects]

    # Load all sessions from all subjects
    calib_dir = Path("calibration")
    subject_xdfs = []
    for subject in subjects:
        xdfs = sorted(calib_dir.glob(f"{subject}_*.xdf"))
        subject_xdfs.extend(xdfs)
        if len(xdfs) > 0:
            print(f"  {subject}: {len(xdfs)} session(s)")

    if len(subject_xdfs) == 0:
        print(f"No calibration data found for subjects: {subjects}")
        return None

    print(f"\nTotal sessions: {len(subject_xdfs)}")

    # Test individual sessions
    session_results = []
    all_trials = []
    all_labels = []

    for xdf_file in subject_xdfs:
        print(f"\n  Session: {xdf_file.name}")

        # Extract trials
        X, y = extract_trials_from_xdf(str(xdf_file), config,
                                       trial_duration_sec=4.0, baseline_sec=0.5)
        print(f"    Extracted: {len(X)} trials")

        # Preprocess
        preprocessor = OfflinePreprocessor(config)
        X_preprocessed = preprocessor.process(X)

        # Apply quality filtering if enabled
        if filter_quality:
            X_filtered, y_filtered, snr_vals = filter_trials_by_quality(
                X, y, config, snr_threshold, verbose=True
            )

            # Repreprocess filtered trials
            X_preprocessed = preprocessor.process(X_filtered)
            y = y_filtered

        # Train on this session alone
        if len(X_preprocessed) > 4:  # Need at least 5 trials for LOOCV
            trca_session = TRCA(config, n_components=1)
            trca_session.fit(X_preprocessed, y)

            y_pred, _ = trca_session.predict_with_correlation(X_preprocessed)
            train_acc = np.mean(y_pred == y)

            # LOOCV on this session
            loocv_acc, loocv_correct, loocv_total = evaluate_loocv(
                X_preprocessed, y, config, filter_quality=False, snr_threshold=snr_threshold
            )

            print(f"    Training acc: {train_acc*100:.1f}%")
            print(f"    LOOCV acc: {loocv_acc*100:.1f}% ({loocv_correct}/{loocv_total})")

            session_results.append({
                'session': xdf_file.name,
                'n_trials': len(X_preprocessed),
                'train_acc': train_acc,
                'loocv_acc': loocv_acc
            })

        all_trials.append(X_preprocessed)
        all_labels.append(y)

    # Batch learning (combine all sessions)
    print(f"\n  Batch Learning (All Sessions Combined):")
    X_combined = np.concatenate(all_trials, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)

    print(f"    Total trials: {len(X_combined)}")

    # Train/test split
    n_total = len(X_combined)
    n_test = int(n_total * 0.2)
    indices = np.random.permutation(n_total)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X_combined[train_idx], y_combined[train_idx]
    X_test, y_test = X_combined[test_idx], y_combined[test_idx]

    # Train final model
    trca_final = TRCA(config, n_components=1)
    trca_final.fit(X_train, y_train)

    y_pred_train, _ = trca_final.predict_with_correlation(X_train)
    y_pred_test, _ = trca_final.predict_with_correlation(X_test)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)

    print(f"    Training acc: {train_acc*100:.1f}% ({np.sum(y_pred_train == y_train)}/{len(y_train)})")
    print(f"    Test acc: {test_acc*100:.1f}% ({np.sum(y_pred_test == y_test)}/{len(y_test)})")

    # LOOCV on combined data
    print(f"    Running LOOCV on {len(X_combined)} trials...")
    loocv_acc, loocv_correct, loocv_total = evaluate_loocv(
        X_combined, y_combined, config, filter_quality=False, snr_threshold=snr_threshold
    )

    print(f"    LOOCV acc: {loocv_acc*100:.1f}% ({loocv_correct}/{loocv_total})")

    results = {
        'description': description,
        'n_sessions': len(subject_xdfs),
        'n_total_trials': len(X_combined),
        'session_results': session_results,
        'batch_train_acc': train_acc,
        'batch_test_acc': test_acc,
        'batch_loocv_acc': loocv_acc,
        'config': {
            'num_fbs': config.num_fbs,
            'window_samples': config.window_samples,
            'filter_quality': filter_quality,
            'snr_threshold': snr_threshold if filter_quality else None
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="SSVEP-BCI Accuracy Optimization")
    parser.add_argument("--subject", type=str, default="donny",
                       help="Subject name (default: donny). Use 'all' for multi-subject ensemble")
    parser.add_argument("--ensemble", action="store_true",
                       help="Create multi-subject ensemble model (combines all subjects)")

    args = parser.parse_args()

    # Handle multi-subject ensemble
    if args.ensemble or args.subject.lower() == "all":
        subjects = ["donny", "logan"]  # Add more subjects as needed
        subject_str = "ensemble_" + "_".join(subjects)
        print("="*70)
        print(f"SSVEP-BCI Multi-Subject Ensemble Optimization")
        print(f"Subjects: {', '.join(subjects)}")
        print("="*70)
    else:
        subjects = [args.subject]
        subject_str = args.subject
        print("="*70)
        print(f"SSVEP-BCI Accuracy Optimization - Subject: {args.subject}")
        print("="*70)

    all_results = []

    # Test 1: Current configuration (baseline)
    config1 = SSVEPConfig()
    config1.num_fbs = 2
    results1 = test_configuration(
        subjects, config1,
        filter_quality=False,
        description="Baseline (2 filter banks, no filtering)"
    )
    if results1:
        all_results.append(results1)

    # Test 2: Quality filtering (SNR >= 1.0)
    config2 = SSVEPConfig()
    config2.num_fbs = 2
    results2 = test_configuration(
        subjects, config2,
        filter_quality=True,
        snr_threshold=1.0,
        description="Quality Filtering (SNR >= 1.0)"
    )
    if results2:
        all_results.append(results2)

    # Test 3: Quality filtering (SNR >= 1.5)
    config3 = SSVEPConfig()
    config3.num_fbs = 2
    results3 = test_configuration(
        subjects, config3,
        filter_quality=True,
        snr_threshold=1.5,
        description="Quality Filtering (SNR >= 1.5)"
    )
    if results3:
        all_results.append(results3)

    # Test 4: More filter banks
    config4 = SSVEPConfig()
    config4.num_fbs = 3
    results4 = test_configuration(
        subjects, config4,
        filter_quality=True,
        snr_threshold=1.0,
        description="Quality Filtering + 3 Filter Banks"
    )
    if results4:
        all_results.append(results4)

    # Test 5: Different window size (1s instead of 4.5s)
    config5 = SSVEPConfig()
    config5.num_fbs = 2
    config5.window_samples = 250  # 1 second at 250Hz
    # Note: This requires retraining with different epoch extraction
    # Skipping for now as it requires modifying extract_trials_from_xdf

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Batch Learning LOOCV Accuracy")
    print("="*70)

    # Sort by LOOCV accuracy
    all_results.sort(key=lambda x: x['batch_loocv_acc'], reverse=True)

    for i, result in enumerate(all_results):
        print(f"\n{i+1}. {result['description']}")
        print(f"   LOOCV: {result['batch_loocv_acc']*100:.1f}%")
        print(f"   Test: {result['batch_test_acc']*100:.1f}%")
        print(f"   Train: {result['batch_train_acc']*100:.1f}%")
        print(f"   Trials: {result['n_total_trials']}")

    # Save best model
    if len(all_results) > 0:
        best = all_results[0]
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION: {best['description']}")
        print(f"  LOOCV Accuracy: {best['batch_loocv_acc']*100:.1f}%")
        print(f"{'='*70}")

        # Retrain with best configuration and save
        print("\nRetraining with best configuration on all data...")

        best_config = SSVEPConfig()
        best_config.num_fbs = best['config']['num_fbs']

        calib_dir = Path("calibration")

        # Load sessions from all subjects
        subject_xdfs = []
        for subject in subjects:
            xdfs = sorted(calib_dir.glob(f"{subject}_*.xdf"))
            subject_xdfs.extend(xdfs)

        all_trials = []
        all_labels = []
        all_raw_trials = []  # For mnakanishi.py (needs raw data)

        for xdf_file in subject_xdfs:
            X, y = extract_trials_from_xdf(str(xdf_file), best_config,
                                          trial_duration_sec=4.0, baseline_sec=0.5)

            # Apply quality filtering if best config uses it
            if best['config']['filter_quality']:
                X, y, _ = filter_trials_by_quality(
                    X, y, best_config,
                    snr_threshold=best['config']['snr_threshold'],
                    verbose=False
                )

            # Save raw data for mnakanishi.py
            all_raw_trials.append(X)

            preprocessor = OfflinePreprocessor(best_config)
            X_preprocessed = preprocessor.process(X)

            all_trials.append(X_preprocessed)
            all_labels.append(y)

        X_all = np.concatenate(all_trials, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        X_raw_all = np.concatenate(all_raw_trials, axis=0)

        trca_best = TRCA(best_config, n_components=1)
        trca_best.fit(X_all, y_all)

        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{subject_str}_trca_optimized_{timestamp}.pkl"

        model_data = {
            'trca': trca_best,
            'config': best_config,
            'train_accuracy': best['batch_train_acc'],
            'test_accuracy': best['batch_test_acc'],
            'loocv_accuracy': best['batch_loocv_acc'],
            'subject': subject_str,  # String representation (single or "ensemble_donny_logan")
            'subjects': subjects,  # List of subjects (for ensemble tracking)
            'is_ensemble': len(subjects) > 1,
            'timestamp': datetime.now().isoformat(),
            'n_sessions': len(subject_xdfs),
            'n_total_trials': len(X_all),
            'optimization_config': best['config'],
            'training_data': {
                'X': X_raw_all,  # Raw EEG for mnakanishi.py
                'y': y_all
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nOptimized model saved: {model_path}")
        print(f"  Use with:")
        print(f"    python classify.py {model_path}")
        print(f"    python mnakanishi.py {model_path}")
        print(f"\n  (Model now includes training data for mnakanishi.py)")


if __name__ == "__main__":
    main()
